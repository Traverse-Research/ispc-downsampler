#include "image.ispc"

// Alpha coverage model: every texel's alpha is scaled and clamped to 1, then the quad between each 2x2 block of
// texels is sampled at 4x4 bilinear subsamples. With a cutoff, coverage is the fraction of subsamples whose alpha
// is above it; without one it is the mean alpha. Images that are 1 texel wide or high sample the texels directly.
//
// Coverage only grows with the scale, so the scale that makes the downsampled image match the source is found by
// bisection. Every step is one vectorized pass over the downsampled image, which is cheaper than building a
// histogram with per-lane scatters.

// The logic for this kernel only supports 4-channel textures
const uniform int SUBSAMPLES = 4;
const uniform float MIN_SCALE = 1.0f / 256.0f;
const uniform float MAX_SCALE = 256.0f;

inline uniform bool is_1d(uniform Image image) {
    return image.size.x == 1 || image.size.y == 1;
}

inline const uniform uint32* uniform pixel_row(uniform Image image, uniform int y) {
    return (const uniform uint32*)(image.data + (uint64)y * image.row_pitch);
}

// Scaled and clamped alpha of a packed rgba pixel.
inline float scaled_alpha(uint32 pixel, uniform float scale) {
    return min((float)(int32)(pixel >> 24) * (scale / 255.0f), 1.0f);
}

// ---------------------------------------------------------------------------------------------------------------
// With a cutoff

inline uniform uint64 num_samples(uniform Image image) {
    return is_1d(image) ? (uint64)image.size.x * image.size.y
                        : (uint64)(image.size.x - 1) * (image.size.y - 1) * SUBSAMPLES * SUBSAMPLES;
}

// Visible subsamples in the row of a quad at `fy`. Along the row the bilinear alpha is linear in x.
inline uint32 visible_in_row(float a00, float a10, float a01, float a11, uniform float fy, uniform float cutoff) {
    float left = a00 + (a01 - a00) * fy;
    float slope = a10 + (a11 - a10) * fy - left;
    uint32 v0 = left + slope * 0.125f > cutoff ? 1 : 0;
    uint32 v1 = left + slope * 0.375f > cutoff ? 1 : 0;
    uint32 v2 = left + slope * 0.625f > cutoff ? 1 : 0;
    uint32 v3 = left + slope * 0.875f > cutoff ? 1 : 0;
    return (v0 + v1) + (v2 + v3);
}

// Number of subsamples with scaled alpha above the cutoff. Along a row of subsamples the bilinear alpha is linear
// in x, so every subsample is a single multiply-add from the row's left edge.
uniform uint64 count_visible(uniform Image image, uniform float scale, uniform float cutoff) {
    uniform uint64 visible = 0;
    if (is_1d(image)) {
        // A row or a column: texels are `step` bytes apart.
        uniform uint32 step = image.size.y == 1 ? 4 : image.row_pitch;
        uint32 lane_visible = 0;
        foreach (i = 0 ... image.size.x * image.size.y) {
            uint32 pixel = *((const uniform uint32* varying)(image.data + (uint64)i * step));
            lane_visible += scaled_alpha(pixel, scale) > cutoff ? 1 : 0;
        }
        return reduce_add(lane_visible);
    }
    for (uniform int y = 0; y < image.size.y - 1; y++) {
        const uniform uint32* uniform top = pixel_row(image, y);
        const uniform uint32* uniform bottom = pixel_row(image, y + 1);
        // One count per subsample row, and the 4 compares of a row summed as a tree: a single running count would
        // make all 16 additions per quad wait on each other.
        uint32 visible0 = 0, visible1 = 0, visible2 = 0, visible3 = 0;
        foreach (x = 0 ... image.size.x - 1) {
            float a00 = scaled_alpha(top[x], scale);
            float a10 = scaled_alpha(top[x + 1], scale);
            float a01 = scaled_alpha(bottom[x], scale);
            float a11 = scaled_alpha(bottom[x + 1], scale);
            // Bilinear values lie between the corners, so quads entirely above or at/below the cutoff are all or
            // nothing. Cut-out textures are mostly such quads; only evaluate subsamples where there is an edge.
            float lowest = min(min(a00, a10), min(a01, a11));
            float highest = max(max(a00, a10), max(a01, a11));
            if (all(lowest > cutoff || highest <= cutoff)) {
                visible0 += lowest > cutoff ? SUBSAMPLES * SUBSAMPLES : 0;
                continue;
            }
            visible0 += visible_in_row(a00, a10, a01, a11, 0.125f, cutoff);
            visible1 += visible_in_row(a00, a10, a01, a11, 0.375f, cutoff);
            visible2 += visible_in_row(a00, a10, a01, a11, 0.625f, cutoff);
            visible3 += visible_in_row(a00, a10, a01, a11, 0.875f, cutoff);
        }
        visible += reduce_add((visible0 + visible1) + (visible2 + visible3));
    }
    return visible;
}

// The scale whose visible count is closest to `target`, to within ~0.1%, preferring the scale closest to 1.
uniform float solve_scale_with_cutoff(uniform Image image, uniform float cutoff, uniform uint64 target) {
    uniform uint64 at_one = count_visible(image, 1.0f, cutoff);
    if (at_one == target)
        return 1.0f;

    // Bracket the target between two scales an octave apart, starting from 1, then bisect in the log domain.
    uniform float lo, hi;
    uniform uint64 lo_count, hi_count;
    if (at_one < target) {
        lo = 1.0f, lo_count = at_one;
        hi = 2.0f, hi_count = count_visible(image, hi, cutoff);
        while (hi_count < target && hi < MAX_SCALE) {
            lo = hi, lo_count = hi_count;
            hi *= 2.0f, hi_count = count_visible(image, hi, cutoff);
        }
        if (hi_count < target)
            return hi;
    } else {
        hi = 1.0f, hi_count = at_one;
        lo = 0.5f, lo_count = count_visible(image, lo, cutoff);
        while (lo_count > target && lo > MIN_SCALE) {
            hi = lo, hi_count = lo_count;
            lo *= 0.5f, lo_count = count_visible(image, lo, cutoff);
        }
        if (lo_count > target)
            return lo;
    }
    // Narrow the bracket to ~0.1%. The count is close to linear in log(scale) locally, so interpolate for the next
    // guess, but keep it away from the ends of the bracket and fall back to bisection when the bracket does not
    // shrink fast enough, so every step makes progress even where the count is flat or jumps.
    uniform float previous_width = log(hi / lo);
    for (uniform int i = 0; i < 32 && hi > lo * 1.001f; i++) {
        uniform float width = log(hi / lo);
        uniform float t = 0.5f;
        if (width < 0.5f * previous_width || i == 0) {
            uniform float span = (uniform float)((double)hi_count - (double)lo_count);
            if (span > 0.0f)
                t = clamp((uniform float)((double)target - (double)lo_count) / span, 0.1f, 0.9f);
        }
        previous_width = width;
        uniform float mid = lo * exp(t * width);
        uniform uint64 mid_count = count_visible(image, mid, cutoff);
        if (mid_count < target || (at_one > target && mid_count == target)) {
            // Below the target, or on it but further from 1 than needed: keep the part towards 1.
            lo = mid, lo_count = mid_count;
        } else {
            hi = mid, hi_count = mid_count;
        }
    }
    uniform uint64 lo_error = lo_count > target ? lo_count - target : target - lo_count;
    uniform uint64 hi_error = hi_count > target ? hi_count - target : target - hi_count;
    // On a tie take the side closer to 1.
    if (lo_error == hi_error)
        return at_one < target ? lo : hi;
    return lo_error < hi_error ? lo : hi;
}

// ---------------------------------------------------------------------------------------------------------------
// Without a cutoff

// Mean scaled alpha, and its derivative with respect to the scale. Summed over a quad's 4x4 subsamples every
// corner gets a quarter of the quad, and bilinear interpolation is linear, so this is a weighted mean of the texels:
// 1 inside, 1/2 on edges, 1/4 in corners. The derivative counts the texels that are not clamped yet.
uniform double mean_alpha(uniform Image image, uniform float scale, uniform double& slope) {
    uniform double total = 0.0, total_slope = 0.0;
    uniform bool one_d = is_1d(image);
    for (uniform int y = 0; y < image.size.y; y++) {
        const uniform uint32* uniform row = pixel_row(image, y);
        // Interior texels have weight 1: sum them two vectors at a time for independent sums, and add the edge
        // texels (weight 1/2) separately.
        float sum0 = 0.0f, sum1 = 0.0f, slope0 = 0.0f, slope1 = 0.0f;
        uniform int x_begin = one_d ? 0 : 1;
        uniform int x_end = one_d ? image.size.x : image.size.x - 1;
        uniform int x = x_begin;
        for (; x + 2 * programCount <= x_end; x += 2 * programCount) {
            float a0 = (float)(int32)(row[x + programIndex] >> 24) * (1.0f / 255.0f);
            float a1 = (float)(int32)(row[x + programCount + programIndex] >> 24) * (1.0f / 255.0f);
            sum0 += min(a0 * scale, 1.0f);
            sum1 += min(a1 * scale, 1.0f);
            slope0 += a0 * scale < 1.0f ? a0 : 0.0f;
            slope1 += a1 * scale < 1.0f ? a1 : 0.0f;
        }
        foreach (i = x ... x_end) {
            float a0 = (float)(int32)(row[i] >> 24) * (1.0f / 255.0f);
            sum0 += min(a0 * scale, 1.0f);
            slope0 += a0 * scale < 1.0f ? a0 : 0.0f;
        }
        uniform float row_sum = reduce_add(sum0 + sum1), row_slope = reduce_add(slope0 + slope1);
        if (!one_d) {
            uniform float edges[2] = { (uniform float)(int32)(row[0] >> 24) * (1.0f / 255.0f),
                (uniform float)(int32)(row[image.size.x - 1] >> 24) * (1.0f / 255.0f) };
            for (uniform int e = 0; e < 2; e++) {
                row_sum += 0.5f * min(edges[e] * scale, 1.0f);
                row_slope += edges[e] * scale < 1.0f ? 0.5f * edges[e] : 0.0f;
            }
        }
        uniform float row_weight = one_d || (y != 0 && y != image.size.y - 1) ? 1.0f : 0.5f;
        total += row_weight * row_sum;
        total_slope += row_weight * row_slope;
    }
    uniform double samples = one_d ? (double)image.size.x * image.size.y : (double)(image.size.x - 1) * (image.size.y - 1);
    slope = total_slope / samples;
    return total / samples;
}

// The scale whose mean alpha matches `target`. The mean is piecewise linear, increasing and concave in the scale,
// so Newton's method converges in a few passes: from below it never overshoots, and from above one step lands
// below. Being piecewise linear it is exact once it reaches the right piece.
uniform float solve_scale_linear(uniform Image image, uniform double target) {
    uniform float scale = 1.0f;
    for (uniform int i = 0; i < 32; i++) {
        uniform double slope;
        uniform double mean = mean_alpha(image, scale, slope);
        if (mean == target || slope <= 0.0)
            break;
        uniform float next = clamp((uniform float)(scale + (target - mean) / slope), 0.0f, MAX_SCALE);
        if (abs(next - scale) <= 1e-6f * scale)
            return next;
        scale = next;
    }
    return scale;
}

// ---------------------------------------------------------------------------------------------------------------

void apply_alpha_scale(uniform Image image, uniform float scale) {
    for (uniform int y = 0; y < image.size.y; y++) {
        uniform uint32* uniform pixels = (uniform uint32*)pixel_row(image, y);
        foreach (i = 0 ... image.size.x) {
            uint32 pixel = pixels[i];
            uint32 alpha = (int32)min(255.0f, (float)(int32)(pixel >> 24) * scale + 0.5f);
            pixels[i] = (pixel & 0x00FFFFFF) | (alpha << 24);
        }
    }
}

// Coverage of an image, the value `scale_to_coverage()` preserves: the fraction of visible subsamples with a cutoff,
// the mean alpha without one.
uniform double coverage(uniform Image image, const uniform float* uniform alpha_cutoff) {
    if (alpha_cutoff == NULL) {
        uniform double unused_slope;
        return mean_alpha(image, 1.0f, unused_slope);
    }
    return (uniform double)count_visible(image, 1.0f, *alpha_cutoff) / num_samples(image);
}

// Scales the alpha of `image` so its coverage matches `target`.
void scale_to_coverage(uniform Image image, const uniform float* uniform alpha_cutoff, uniform double target) {
    uniform float scale;
    if (alpha_cutoff == NULL) {
        scale = solve_scale_linear(image, target);
    } else {
        uniform float cutoff = *alpha_cutoff;
        // Alpha never exceeds 1 (or goes below 0), so no scale changes coverage for these cutoffs.
        if (cutoff >= 1.0f || cutoff < 0.0f)
            return;
        scale = solve_scale_with_cutoff(image, cutoff, (uniform uint64)(target * num_samples(image) + 0.5));
    }
    if (scale != 1.0f)
        apply_alpha_scale(image, scale);
}

export uniform double alpha_coverage(uniform uint32 width, uniform uint32 height, uniform uint32 row_pitch,
    uniform const uint8 data[], const uniform float* uniform alpha_cutoff) {
    uniform Image image = { (uniform uint8*)data, {width, height}, row_pitch };
    return coverage(image, alpha_cutoff);
}

export void scale_to_target_alpha_coverage(uniform uint32 width, uniform uint32 height, uniform uint32 row_pitch,
    uniform uint8 data[], const uniform float* uniform alpha_cutoff, uniform double target) {
    uniform Image image = { data, {width, height}, row_pitch };
    scale_to_coverage(image, alpha_cutoff, target);
}

export void scale_to_alpha_coverage(uniform uint32 source_width, uniform uint32 source_height, uniform const uint8 source_data[],
    uniform uint32 downsampled_width, uniform uint32 downsampled_height, uniform uint8 downsampled_image_data[],
    const uniform float* uniform alpha_cutoff) {
    uniform Image source = { (uniform uint8*)source_data, {source_width, source_height}, source_width * 4 };
    uniform Image downsampled = { downsampled_image_data, {downsampled_width, downsampled_height}, downsampled_width * 4 };
    // Cutoffs outside 0..1 are no-ops, and would count nothing.
    if (alpha_cutoff != NULL && (*alpha_cutoff >= 1.0f || *alpha_cutoff < 0.0f))
        return;
    scale_to_coverage(downsampled, alpha_cutoff, coverage(source, alpha_cutoff));
}
