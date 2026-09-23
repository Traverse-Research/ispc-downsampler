#include "weights.ispc"
#include "formats.ispc"
#include "image.ispc"
#include "math.ispc"
#include "linearization.ispc"


inline float<3> sample_3_channels(const uniform uint8* varying pixel_ptr) {
    // Memory reinterpretation to read all channels at once rather than one-by-one.
    // While testing, this proved more performant than reading one-by-one.
    const uniform uint8<3>* pixel_ptr3 = (const uniform uint8<3>*)(pixel_ptr);
    varying float<3> dst = {0.0f, 0.0f, 0.0f};
    dst = *pixel_ptr3 / 255.0f;

    return dst;
}

inline float<4> sample_4_channels(const uniform uint8* varying pixel_ptr) {
    const uniform uint8<4>* pixel_ptr4 = (const uniform uint8<4>*)(pixel_ptr);
    varying float<4> dst = {0.0f, 0.0f, 0.0f, 0.0f};
    dst = *pixel_ptr4 / 255.0f;

    return dst;
}

inline float<3> sample_3_channels_srgb(const uniform uint8* varying pixel_ptr, const uniform SrgbToLinearLut * uniform lut) {
    const uniform uint8<3>* pixel_ptr3 = (const uniform uint8<3>*)(pixel_ptr);
    uint8<3> srgb = *pixel_ptr3;
    float<3> dst = {lut->values[srgb.x], lut->values[srgb.y], lut->values[srgb.z]};
    return dst;
}

inline float<4> sample_4_channels_srgb(const uniform uint8* varying pixel_ptr, const uniform SrgbToLinearLut * uniform lut) {
    const uniform uint8<4>* pixel_ptr4 = (const uniform uint8<4>*)(pixel_ptr);
    uint8<4> srgb = *pixel_ptr4;
    // Alpha is not sRGB-encoded
    float<4> dst = {lut->values[srgb.x], lut->values[srgb.y], lut->values[srgb.z], srgb.w / 255.0f};
    return dst;
}

inline float<3> sample_3_channels(const uniform float* varying pixel_ptr) {
    const uniform float<3>* pixel_ptr3 = (const uniform float<3>*)(pixel_ptr);
    return *pixel_ptr3;
}

inline float<4> sample_4_channels(const uniform float* varying pixel_ptr) {
    const uniform float<4>* pixel_ptr4 = (const uniform float<4>*)(pixel_ptr);
    return *pixel_ptr4;
}

inline float<3> sample_normal(const uniform uint8* varying pixel_ptr, uniform NormalMapFormat normal_map_format) {
    float<3> normal = {0.0f, 0.0f, 0.0f};
    switch (normal_map_format) {
        case R8g8b8: {
            // Memory reinterpretation to read all channels at once rather than one-by-one.
            // While testing, this proved more performant than reading one-by-one.
            const uniform uint8<3>* pixel_ptr3 = (const uniform uint8<3>*)(pixel_ptr);

            normal = *pixel_ptr3 / 255.0f;
            normal = normal * 2.0f - 1.0f;
            break;
        }
        case R8g8TangentSpaceReconstructedZ: {
            // Memory reinterpretation to read all channels at once rather than one-by-one.
            // While testing, this proved more performant than reading one-by-one.
            const uniform uint8<2>* pixel_ptr2 = (const uniform uint8<2>*)(pixel_ptr);
            normal.x = pixel_ptr2->x / 255.0f;
            normal.y = pixel_ptr2->y / 255.0f;
            normal = normal * 2.0f - 1.0f;

            normal.z = sqrt(max(0.01, 1.0 - (normal.x * normal.x + normal.y * normal.y)));
            break;
        }
    }

    return normal;
}

inline void write_3_channels(varying float<3> color, uniform float* varying pixel_ptr) {
    uniform float<3>* pixel_ptr3 = (uniform float<3>*)(pixel_ptr);
    *pixel_ptr3 = color;
}

inline void write_4_channels(varying float<4> color, uniform float* varying pixel_ptr) {
    uniform float<4>* pixel_ptr4 = (uniform float<4>*)(pixel_ptr);
    *pixel_ptr4 = color;
}

inline void clean_and_write_3_channels(varying float<3> color, uniform uint8* varying pixel_ptr, uniform bool srgb) {
    // The final color is a sum of numbers that are multiplied by the weights of their respective pixels.
    // Because of their numbers, floating point precision leads to the final color being potentially outside of the 0-255 range by a slight margin.
    // This would cause an underflow/overflow, which we avoid with the clamps.

    color[0] = clamp(color[0], 0.0f, 1.0f);
    color[1] = clamp(color[1], 0.0f, 1.0f);
    color[2] = clamp(color[2], 0.0f, 1.0f);
    if (srgb) {
        color[0] = linear_to_srgb(color[0]);
        color[1] = linear_to_srgb(color[1]);
        color[2] = linear_to_srgb(color[2]);
    }
    color = color * 255.0f + 0.5f;

    pixel_ptr[0] = color[0];
    pixel_ptr[1] = color[1];
    pixel_ptr[2] = color[2];
}

inline void clean_and_write_4_channels(varying float<4> color, uniform uint8* varying pixel_ptr, uniform bool srgb) {
    color[0] = clamp(color[0], 0.0f, 1.0f);
    color[1] = clamp(color[1], 0.0f, 1.0f);
    color[2] = clamp(color[2], 0.0f, 1.0f);
    color[3] = clamp(color[3], 0.0f, 1.0f);
    if (srgb) {
        // Alpha is not sRGB-encoded
        color[0] = linear_to_srgb(color[0]);
        color[1] = linear_to_srgb(color[1]);
        color[2] = linear_to_srgb(color[2]);
    }
    color = color * 255.0f + 0.5f;

    pixel_ptr[0] = color[0];
    pixel_ptr[1] = color[1];
    pixel_ptr[2] = color[2];
    pixel_ptr[3] = color[3];
}

inline void clean_and_write_normal(varying float<3> normal, uniform uint8* varying pixel_ptr, uniform NormalMapFormat normal_map_format) {
    normal = normalize(normal);
    normal = normal * 0.5 + 0.5;

    switch (normal_map_format) {
        case R8g8b8: {
            normal[0] = clamp(normal[0], 0.0f, 1.0f);
            normal[1] = clamp(normal[1], 0.0f, 1.0f);
            normal[2] = clamp(normal[2], 0.0f, 1.0f);
            normal = normal * 255.0f + 0.5f;

            pixel_ptr[0] = (uint8)normal[0];
            pixel_ptr[1] = (uint8)normal[1];
            pixel_ptr[2] = (uint8)normal[2];
            break;
        }
        case R8g8TangentSpaceReconstructedZ: {
            normal[0] = clamp(normal[0], 0.0f, 1.0f);
            normal[1] = clamp(normal[1], 0.0f, 1.0f);
            normal = normal * 255.0f + 0.5f;

            pixel_ptr[0] = normal[0];
            pixel_ptr[1] = normal[1];
        }
    }
}

struct SourceImage {
    uniform uint32 width;
    uniform uint32 height;
    uniform const uint8* data;
    uniform uint pixel_stride;
};

struct DownsampledImage {
    uniform uint32 width;
    uniform uint32 height;
    uniform uint8* data;
    uniform uint pixel_stride;
};

struct DownsamplingContext {
    uniform const SampleWeights weights;
    uniform float* scratch_space;
    // Holds a single row of the source image decoded to linear space, for sRGB inputs
    uniform float* linear_row;
};

// Accumulates `sample` into `color`. When alpha weighting, rgb is weighted by alpha as well so fully transparent
// texels (often black) do not bleed into opaque neighbours. `unweighted` keeps the plain rgb sum as a fallback.
inline void accumulate_4_channels(varying float<4>& color, varying float<3>& unweighted, varying float<4> sample, varying float weight, uniform bool alpha_weighted) {
    if (alpha_weighted) {
        float alpha_weight = sample[3] * weight;
        color[0] += sample[0] * alpha_weight;
        color[1] += sample[1] * alpha_weight;
        color[2] += sample[2] * alpha_weight;
        color[3] += alpha_weight;
        unweighted[0] += sample[0] * weight;
        unweighted[1] += sample[1] * weight;
        unweighted[2] += sample[2] * weight;
    } else {
        color += sample * weight;
    }
}

// Number of floats per texel in the scratch space. When alpha weighting, the horizontal pass stores the unresolved
// alpha weighted rgb sum, the alpha sum and the unweighted rgb sum, so that the vertical pass can keep accumulating
// them linearly and the division only happens once at the end. Resolving in between is not exact: the negative lobes
// of the filter can leave an intermediate alpha sum at or below zero, where the unweighted fallback kicks in.
inline uniform uint scratch_channel_count(uniform PixelFormat pixel_format, uniform bool alpha_weighted) {
    return alpha_weighted ? 7 : channel_count(pixel_format);
}

inline void write_4_channels_to_scratch(varying float<4> color, varying float<3> unweighted, uniform float* varying pixel_ptr, uniform bool alpha_weighted) {
    write_4_channels(color, pixel_ptr);
    if (alpha_weighted)
        write_3_channels(unweighted, pixel_ptr + 4);
}

// Turns the alpha weighted rgb sum back into a colour.
inline float<4> resolve_4_channels(varying float<4> color, varying float<3> unweighted, uniform bool alpha_weighted) {
    if (alpha_weighted) {
        // Nothing visible under the filter, so there is nothing to weight by.
        if (color[3] > 0.5f / 255.0f) {
            color[0] /= color[3];
            color[1] /= color[3];
            color[2] /= color[3];
        } else {
            color[0] = unweighted[0];
            color[1] = unweighted[1];
            color[2] = unweighted[2];
        }
    }
    return color;
}

/// Horizontal pass of `resample_with_cached_weights()` for sRGB-encoded source images, which are
/// decoded to linear space before filtering. Every source pixel is visited by multiple filter taps,
/// so each row is decoded once up-front into `ctx->linear_row` rather than for every tap.
inline void horizontal_pass_srgb(uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx, uniform bool alpha_weighted) {
    uniform WeightCollection * uniform horizontal_weight_collection = ctx->weights.horizontal_weights;
    const uniform uint num_channels = channel_count(pixel_format);
    const uniform uint scratch_channels = scratch_channel_count(pixel_format, alpha_weighted);

    uniform SrgbToLinearLut srgb_to_linear_lut;
    init_srgb_to_linear_lut(&srgb_to_linear_lut);

    for (uniform uint32 y = 0; y < src->height; y++) {
        foreach (x = 0 ... src->width) {
            uint64 src_read_address = ((uint64)y * src->width + x) * src->pixel_stride;

            if (channel_count(pixel_format) == 3)
                write_3_channels(sample_3_channels_srgb(src->data + src_read_address, &srgb_to_linear_lut), ctx->linear_row + x * num_channels);
            if (channel_count(pixel_format) == 4)
                write_4_channels(sample_4_channels_srgb(src->data + src_read_address, &srgb_to_linear_lut), ctx->linear_row + x * num_channels);
        }

        foreach (x = 0 ... dst->width) {
            uint32 src_width_start = horizontal_weight_collection->starts[x];
            uint32 num_horizontal_weights = horizontal_weight_collection->weight_counts[x];
            float* horizontal_weights = horizontal_weight_collection->values[x];

            float<3> color3 = {0.0f, 0.0f, 0.0f};
            float<4> color4 = {0.0f, 0.0f, 0.0f, 0.0f};
            float<3> unweighted = {0.0f, 0.0f, 0.0f};
            for (uint32 i = 0; i < num_horizontal_weights; i++) {
                float weight = horizontal_weights[i];
                uint32 src_x = src_width_start + i;

                if (channel_count(pixel_format) == 3)
                    color3 += sample_3_channels(ctx->linear_row + src_x * num_channels) * weight;
                if (channel_count(pixel_format) == 4)
                    accumulate_4_channels(color4, unweighted, sample_4_channels(ctx->linear_row + src_x * num_channels), weight, alpha_weighted);
            }

            uint64 scratch_write_address = (y * dst->width + x) * scratch_channels;

            if (channel_count(pixel_format) == 3)
                write_3_channels(color3, ctx->scratch_space + scratch_write_address);
            if (channel_count(pixel_format) == 4)
                write_4_channels_to_scratch(color4, unweighted, ctx->scratch_space + scratch_write_address, alpha_weighted);
        }
    }
}

inline void resample_with_cached_weights(uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx, uniform bool alpha_weighted) {
    uniform WeightCollection * uniform vertical_weight_collection = ctx->weights.vertical_weights;
    uniform WeightCollection * uniform horizontal_weight_collection = ctx->weights.horizontal_weights;
    const uniform uint num_channels = channel_count(pixel_format);
    const uniform uint scratch_channels = scratch_channel_count(pixel_format, alpha_weighted);
    const uniform bool srgb = is_srgb(pixel_format);

    // Accumulate only along the width for each pixel, sampling from the source image
    // Results in the source image being downsampled to src->height X dst->width
    // Filtering happens in linear space: sRGB inputs are decoded to linear in the horizontal pass,
    // and encoded back to sRGB when writing the final result.
    if (srgb) {
        horizontal_pass_srgb(src, dst, pixel_format, ctx, alpha_weighted);
    } else {
        foreach_tiled(y =  0 ... src->height, x = 0 ... dst->width) {

            uint32 src_width_start = horizontal_weight_collection->starts[x];
            uint32 num_horizontal_weights = horizontal_weight_collection->weight_counts[x];
            float* horizontal_weights = horizontal_weight_collection->values[x];

            float<3> color3 = {0.0f, 0.0f, 0.0f};
            float<4> color4 = {0.0f, 0.0f, 0.0f, 0.0f};
            float<3> unweighted = {0.0f, 0.0f, 0.0f};
            for (uint32 i = 0; i < num_horizontal_weights; i++) {
                float weight = horizontal_weights[i];
                uint32 src_x = src_width_start + i;
                uint64 src_read_address = (y * src->width + src_x) * src->pixel_stride;

                if (channel_count(pixel_format) == 3)
                    color3 += sample_3_channels(src->data + src_read_address) * weight;
                if (channel_count(pixel_format) == 4)
                    accumulate_4_channels(color4, unweighted, sample_4_channels(src->data + src_read_address), weight, alpha_weighted);
            }

            uint64 scratch_write_address = (y * dst->width + x) * scratch_channels;

            // Keep the intermediate result in full precision, and don't clamp it yet: negative lobes of the
            // filter are only cancelled out after the vertical pass.
            if (channel_count(pixel_format) == 3)
                write_3_channels(color3, ctx->scratch_space + scratch_write_address);
            if (channel_count(pixel_format) == 4)
                write_4_channels_to_scratch(color4, unweighted, ctx->scratch_space + scratch_write_address, alpha_weighted);
        }
    }
    // Accumulate the scratch space data along the height
    // Downsamples the src_height X dst->width image to dst->height * dst->width
    foreach_tiled(y =  0 ... dst->height, x = 0 ... dst->width) {

        uint32 src_height_start = vertical_weight_collection->starts[y];
        uint32 num_vertical_weights = vertical_weight_collection->weight_counts[y];
        float* vertical_weights = vertical_weight_collection->values[y];
        float<3> color3 = {0.0f, 0.0f, 0.0f};
        float<4> color4 = {0.0f, 0.0f, 0.0f, 0.0f};
        float<3> unweighted = {0.0f, 0.0f, 0.0f};
        for (uint32 i = 0; i < num_vertical_weights; i++) {
            float weight = vertical_weights[i];
            uint32 scratch_y = src_height_start + i;
            uint64 scratch_read_address = (scratch_y * dst->width + x) * scratch_channels;

            if (channel_count(pixel_format) == 3)
                color3 += sample_3_channels(ctx->scratch_space + scratch_read_address) * weight;
            // The scratch space already holds alpha weighted sums, which only need to be accumulated
            if (channel_count(pixel_format) == 4)
                color4 += sample_4_channels(ctx->scratch_space + scratch_read_address) * weight;
            if (alpha_weighted)
                unweighted += sample_3_channels(ctx->scratch_space + scratch_read_address + 4) * weight;
        }

        uint64 out_write_address = (y * dst->width + x) * dst->pixel_stride;
        assert(out_write_address < dst->height * dst->width * dst->pixel_stride);
        if (channel_count(pixel_format) == 3)
            clean_and_write_3_channels(color3, dst->data + out_write_address, srgb);
        if (channel_count(pixel_format) == 4)
            clean_and_write_4_channels(resolve_4_channels(color4, unweighted, alpha_weighted), dst->data + out_write_address, srgb);
    }
}

/// scratch_space must be at least src_height * dst->width pixels big
export void resample_with_cached_weights_3(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx) {
    const uniform bool is_normal_map = false;
    assume(channel_count(pixel_format) == 3);
    resample_with_cached_weights(src, dst, pixel_format, ctx, false);
}

/// scratch_space must be at least src_height * dst->width pixels big
export void resample_with_cached_weights_4(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx) {
    assume(channel_count(pixel_format) == 4);
    resample_with_cached_weights(src, dst, pixel_format, ctx, false);
}

/// Like `resample_with_cached_weights_4`, but weights rgb by alpha.
/// scratch_space must be at least src_height * dst->width * 7 floats big
export void resample_with_cached_weights_4_alpha_weighted(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx) {
    assume(channel_count(pixel_format) == 4);
    resample_with_cached_weights(src, dst, pixel_format, ctx, true);
}

/// scratch_space must be at least src_height * dst->width pixels big
export void downsample_normal_map(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform NormalMapFormat normal_map_format) {
    float<2> ratio = { (float)src->width / (float)dst->width, (float)src->height / (float)dst->height };
    foreach_tiled(y =  0 ... dst->height, x = 0 ... dst->width) {

        // Box filter over the source texels that the destination texel covers: [x * ratio, (x + 1) * ratio).
        uint<2> start = { 0, 0 };
        uint<2> end = { 0, 0 };
        float pixel_weight = 0.0;

        {
            start.x = min((uint)floor(x * ratio.x), src->width - 1);
            start.y = min((uint)floor(y * ratio.y), src->height - 1);

            end.x = min(max((uint)ceil((x + 1) * ratio.x), start.x + 1), src->width);
            end.y = min(max((uint)ceil((y + 1) * ratio.y), start.y + 1), src->height);

            pixel_weight = (float)((end.x - start.x) * (end.y - start.y));
        }


        float<3> normal = {0.0f, 0.0f, 0.0f};
        for (uint32 i = start.x; i < end.x; i++) {
            for (uint32 j = start.y; j < end.y; j++) {

                uint64 read_address = (j * src->width + i) * src->pixel_stride;
                normal += sample_normal(src->data + read_address, normal_map_format);
            }
        }

        normal /= pixel_weight;

        uint64 out_write_address = (y * dst->width + x) * dst->pixel_stride;
        assert(out_write_address < dst->height * dst->width * dst->pixel_stride);
        clean_and_write_normal(normal, dst->data + out_write_address, normal_map_format);

        uniform uint8* pixel_ptr = (dst->data + out_write_address);

    }
}
