#include "image.ispc"

// Alpha coverage model: every texel's alpha is scaled and clamped to 1, then the quad between each 2x2 block of
// texels is sampled at 4x4 bilinear subsamples. With a cutoff, coverage is the fraction of subsamples whose alpha
// is above it; without one it is the mean alpha. Images that are 1 texel wide or high sample the texels directly.
//
// The scale that makes the downsampled image match the source's coverage is solved directly rather than searched:
// - without a cutoff, coverage is piecewise linear in the scale, so it is solved exactly from an alpha histogram;
// - with a cutoff, every subsample becomes visible above its own threshold scale, so the answer is a quantile of
//   those thresholds, read from a fine log-scale histogram.

// The logic for this kernel only supports 4-channel textures
const uniform uint32 NUM_CHANNELS = 4;
const uniform int SUBSAMPLES = 4;
const uniform float INFINITY = 1.0f / 0.0f;

// Threshold histogram: scales in [2^-MAX_OCTAVE, 2^MAX_OCTAVE], BINS_PER_OCTAVE bins each (~0.07% apart),
// plus one bin below and one above (which also holds "never visible").
const uniform int MAX_OCTAVE = 8;
const uniform int BINS_PER_OCTAVE = 1024;
const uniform int THRESHOLD_BINS = 2 * MAX_OCTAVE * BINS_PER_OCTAVE + 2;

inline uint8 alpha_u8(uniform Image image, int x, int y) {
    return image.data[((uint64)y * image.size.x + x) * NUM_CHANNELS + 3];
}

inline float alpha(uniform Image image, int x, int y) {
    return alpha_u8(image, x, y) / 255.0f;
}

inline uniform bool is_1d(uniform Image image) {
    return image.size.x == 1 || image.size.y == 1;
}

inline uniform uint64 num_samples(uniform Image image) {
    return is_1d(image) ? (uint64)image.size.x * image.size.y
                        : (uint64)(image.size.x - 1) * (image.size.y - 1) * SUBSAMPLES * SUBSAMPLES;
}

inline uniform float subsample_offset(uniform int i) {
    return ((uniform float)i + 0.5f) / (uniform float)SUBSAMPLES;
}

// ---------------------------------------------------------------------------------------------------------------
// Without a cutoff

// Each texel's share of all subsample weight. Summed over a quad's 4x4 subsamples, every corner gets a quarter of
// the quad, so interior texels get 1, edges 1/2 and corners 1/4 (in units of whole quads).
inline float texel_weight(uniform Image image, int x, int y) {
    if (is_1d(image))
        return 1.0f;
    float wx = (x == 0 || x == image.size.x - 1) ? 0.5f : 1.0f;
    float wy = (y == 0 || y == image.size.y - 1) ? 0.5f : 1.0f;
    return wx * wy;
}

// Histogram of alpha values weighted by `texel_weight()`, normalized so that coverage(scale) is
// sum(histogram[a] * min(a * scale / 255, 1)).
void weighted_alpha_histogram(uniform Image image, uniform float histogram[256]) {
    // One histogram per program instance, so lanes never write the same bin.
    uniform float lane_histograms[256 * programCount];
    foreach (i = 0 ... 256 * programCount) {
        lane_histograms[i] = 0.0f;
    }
    foreach (y = 0 ... image.size.y, x = 0 ... image.size.x) {
        lane_histograms[alpha_u8(image, x, y) * programCount + programIndex] += texel_weight(image, x, y);
    }
    uniform float normalization = 1.0f / (is_1d(image) ? (uniform float)image.size.x * image.size.y
                                                       : (uniform float)(image.size.x - 1) * (image.size.y - 1));
    foreach (a = 0 ... 256) {
        float sum = 0.0f;
        for (uniform int lane = 0; lane < programCount; lane++) {
            sum += lane_histograms[a * programCount + lane];
        }
        histogram[a] = sum * normalization;
    }
}

uniform float linear_coverage(uniform float histogram[256]) {
    uniform float coverage = 0.0f;
    for (uniform int a = 0; a < 256; a++) {
        coverage += histogram[a] * a / 255.0f;
    }
    return coverage;
}

// Solves sum(histogram[a] * min(a * scale / 255, 1)) = coverage for scale. Alpha values clamp to 1 from the
// largest down as the scale grows, so walk those breakpoints: between them coverage is clamped + scale * linear.
uniform float solve_linear_scale(uniform float histogram[256], uniform float coverage) {
    uniform float clamped = 0.0f;
    uniform float linear = 0.0f;
    for (uniform int a = 1; a < 256; a++) {
        linear += histogram[a] * a / 255.0f;
    }
    for (uniform int a = 255; a >= 1; a--) {
        // Scales up to 255 / a keep alpha a (and everything below it) unclamped.
        uniform float max_scale = 255.0f / a;
        if (linear > 0.0f) {
            uniform float scale = (coverage - clamped) / linear;
            if (scale <= max_scale)
                return max(scale, 0.0f);
        }
        clamped += histogram[a];
        linear -= histogram[a] * a / 255.0f;
    }
    // Even with every non-zero alpha clamped to 1 the coverage is not reached.
    return 255.0f;
}

// ---------------------------------------------------------------------------------------------------------------
// With a cutoff

// The smallest scale at which sum(w[i] * min(a[i] * scale, 1)) reaches `cutoff`; the subsample is visible above it.
// Assumes 0 <= cutoff < 1. Start without clamping, which underestimates the scale, then clamp the corners that
// exceed 1 at that scale and solve again; each round clamps at least one more corner.
inline float visibility_threshold(float<4> a, uniform float<4> w, uniform float cutoff) {
    float clamped = 0.0f;
    float linear = a[0] * w[0] + a[1] * w[1] + a[2] * w[2] + a[3] * w[3];
    if (linear <= 0.0f)
        return INFINITY;
    float scale = cutoff / linear;
    bool is_clamped[4] = { false, false, false, false };
    for (uniform int round = 0; round < 4; round++) {
        bool changed = false;
        for (uniform int i = 0; i < 4; i++) {
            if (!is_clamped[i] && a[i] * scale > 1.0f) {
                is_clamped[i] = true;
                clamped += w[i];
                linear -= a[i] * w[i];
                changed = true;
            }
        }
        // Reaching the cutoff with clamped corners alone is only possible with equality at the previous scale,
        // which is then the answer.
        if (!changed || clamped >= cutoff)
            break;
        if (linear <= 1e-12f)
            return INFINITY;
        scale = max((cutoff - clamped) / linear, scale);
    }
    return scale;
}

// 2^-MAX_OCTAVE and 2^MAX_OCTAVE.
const uniform float MIN_THRESHOLD = 1.0f / 256.0f;
const uniform float MAX_THRESHOLD = 256.0f;
const uniform float LN_2 = 0.693147180559945f;

inline int threshold_bin(float threshold) {
    if (threshold < MIN_THRESHOLD)
        return 0;
    if (threshold >= MAX_THRESHOLD)
        return THRESHOLD_BINS - 1;
    int bin = 1 + (int)((log(threshold) / LN_2 + MAX_OCTAVE) * BINS_PER_OCTAVE);
    return min(bin, (int)(THRESHOLD_BINS - 2));
}

inline uniform float threshold_bin_start(uniform int bin) {
    if (bin == 0)
        return 0.0f;
    return exp(((uniform float)(bin - 1) / BINS_PER_OCTAVE - MAX_OCTAVE) * LN_2);
}

// Counts the subsamples of `image` per threshold bin.
void threshold_histogram(uniform Image image, uniform float cutoff, uniform uint32 histogram[]) {
    // One histogram per program instance, so lanes never write the same bin.
    uniform uint32* uniform lane_histograms = uniform new uniform uint32[THRESHOLD_BINS * programCount];
    foreach (i = 0 ... THRESHOLD_BINS * programCount) {
        lane_histograms[i] = 0;
    }

    if (is_1d(image)) {
        foreach (y = 0 ... image.size.y, x = 0 ... image.size.x) {
            float a = alpha(image, x, y);
            float threshold = a > 0.0f ? cutoff / a : INFINITY;
            lane_histograms[threshold_bin(threshold) * programCount + programIndex] += 1;
        }
    } else {
        foreach (y = 0 ... image.size.y - 1, x = 0 ... image.size.x - 1) {
            float<4> a = { alpha(image, x, y), alpha(image, x + 1, y), alpha(image, x, y + 1), alpha(image, x + 1, y + 1) };
            for (uniform int sy = 0; sy < SUBSAMPLES; sy++) {
                uniform float fy = subsample_offset(sy);
                for (uniform int sx = 0; sx < SUBSAMPLES; sx++) {
                    uniform float fx = subsample_offset(sx);
                    uniform float<4> w = { (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy), (1.0f - fx) * fy, fx * fy };
                    lane_histograms[threshold_bin(visibility_threshold(a, w, cutoff)) * programCount + programIndex] += 1;
                }
            }
        }
    }

    foreach (bin = 0 ... THRESHOLD_BINS) {
        uint32 sum = 0;
        for (uniform int lane = 0; lane < programCount; lane++) {
            sum += lane_histograms[bin * programCount + lane];
        }
        histogram[bin] = sum;
    }
    delete[] lane_histograms;
}

// Number of subsamples above the cutoff at scale 1. No clamping is needed since alpha is at most 1.
uniform uint64 count_visible(uniform Image image, uniform float cutoff) {
    uint64 visible = 0;
    if (is_1d(image)) {
        foreach (y = 0 ... image.size.y, x = 0 ... image.size.x) {
            visible += alpha(image, x, y) > cutoff ? 1 : 0;
        }
    } else {
        foreach (y = 0 ... image.size.y - 1, x = 0 ... image.size.x - 1) {
            float<4> a = { alpha(image, x, y), alpha(image, x + 1, y), alpha(image, x, y + 1), alpha(image, x + 1, y + 1) };
            for (uniform int sy = 0; sy < SUBSAMPLES; sy++) {
                uniform float fy = subsample_offset(sy);
                for (uniform int sx = 0; sx < SUBSAMPLES; sx++) {
                    uniform float fx = subsample_offset(sx);
                    float v = a[0] * (1.0f - fx) * (1.0f - fy) + a[1] * fx * (1.0f - fy) + a[2] * (1.0f - fx) * fy + a[3] * fx * fy;
                    visible += v > cutoff ? 1 : 0;
                }
            }
        }
    }
    return reduce_add(visible);
}

// Picks a scale at which the number of visible subsamples is closest to `target`. At the start of bin b, the
// subsamples in bins before b are visible. Several consecutive bin starts can give that count (when the bins in
// between are empty); of that range of scales, take the one closest to 1 so alpha changes as little as possible.
uniform float solve_threshold_scale(uniform uint32 histogram[], uniform uint64 target) {
    uniform uint64 best_error = 0xFFFFFFFFFFFFFFFF;
    uniform int best_first = 0;
    uniform int best_last = 0;
    uniform uint64 visible = 0;
    // The last bin start is the top of the supported range; subsamples in the last bin are never counted.
    for (uniform int bin = 0; bin < THRESHOLD_BINS; bin++) {
        uniform uint64 error = visible > target ? visible - target : target - visible;
        if (error < best_error) {
            best_error = error;
            best_first = bin;
            best_last = bin;
        } else if (error == best_error && best_last == bin - 1 && histogram[bin - 1] == 0) {
            best_last = bin;
        }
        visible += histogram[bin];
    }
    return clamp(1.0f, threshold_bin_start(best_first), threshold_bin_start(best_last));
}

// ---------------------------------------------------------------------------------------------------------------

void apply_alpha_scale(uniform Image image, uniform float scale) {
    foreach (y = 0 ... image.size.y, x = 0 ... image.size.x) {
        uint64 index = ((uint64)y * image.size.x + x) * NUM_CHANNELS + 3;
        image.data[index] = (uint8)min(255.0f, image.data[index] * scale + 0.5f);
    }
}

export void scale_to_alpha_coverage(uniform uint32 source_width, uniform uint32 source_height, uniform const uint8 source_data[],
    uniform uint32 downsampled_width, uniform uint32 downsampled_height, uniform uint8 downsampled_image_data[],
    const uniform float* uniform alpha_cutoff) {
    uniform Image source = { (uniform uint8*)source_data, {source_width, source_height} };
    uniform Image downsampled = { downsampled_image_data, {downsampled_width, downsampled_height} };

    uniform float scale = 1.0f;
    if (alpha_cutoff == NULL) {
        uniform float source_histogram[256];
        uniform float downsampled_histogram[256];
        weighted_alpha_histogram(source, source_histogram);
        weighted_alpha_histogram(downsampled, downsampled_histogram);
        scale = solve_linear_scale(downsampled_histogram, linear_coverage(source_histogram));
    } else {
        uniform float cutoff = *alpha_cutoff;
        // Alpha never exceeds 1 (or goes below 0), so no scale changes coverage for these cutoffs.
        if (cutoff >= 1.0f || cutoff < 0.0f)
            return;
        uniform double coverage = (uniform double)count_visible(source, cutoff) / num_samples(source);
        uniform uint64 target = (uniform uint64)(coverage * num_samples(downsampled) + 0.5);

        uniform uint32* uniform histogram = uniform new uniform uint32[THRESHOLD_BINS];
        threshold_histogram(downsampled, cutoff, histogram);
        scale = solve_threshold_scale(histogram, target);
        delete[] histogram;
    }
    apply_alpha_scale(downsampled, scale);
}
