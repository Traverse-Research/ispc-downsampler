#include "weights.ispc"
#include "formats.ispc"
#include "image.ispc"
#include "math.ispc"
#include "linearization.ispc"


// `linear16`: the pixel is stored as linear u16 rather than u8, see `linearize_srgb()`.
inline float<3> sample_3_channels(const uniform uint8* varying pixel_ptr, uniform bool linear16) {
    varying float<3> dst = {0.0f, 0.0f, 0.0f};
    if (linear16) {
        dst = *((const uniform uint16<3>*)(pixel_ptr)) / 65535.0f;
        return dst;
    }
    // Memory reinterpretation to read all channels at once rather than one-by-one.
    // While testing, this proved more performant than reading one-by-one.
    const uniform uint8<3>* pixel_ptr3 = (const uniform uint8<3>*)(pixel_ptr);
    dst = *pixel_ptr3 / 255.0f;

    return dst;
}

inline float<4> sample_4_channels(const uniform uint8* varying pixel_ptr, uniform bool linear16) {
    varying float<4> dst = {0.0f, 0.0f, 0.0f, 0.0f};
    if (linear16) {
        dst = *((const uniform uint16<4>*)(pixel_ptr)) / 65535.0f;
        return dst;
    }
    const uniform uint8<4>* pixel_ptr4 = (const uniform uint8<4>*)(pixel_ptr);
    dst = *pixel_ptr4 / 255.0f;

    return dst;
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

// Writes the final u8 output. sRGB formats are encoded here, the only place colour leaves linear space.
inline void clean_and_write_3_channels(varying float<3> color, uniform uint8* varying pixel_ptr, uniform bool encode_srgb) {
    // Lanczos has negative lobes, so the filtered colour can over- or undershoot the 0-1 range near edges.
    // Clamping only happens here, after both passes, so the lobes of the two passes can cancel out first.
    color[0] = clamp(color[0], 0.0f, 1.0f);
    color[1] = clamp(color[1], 0.0f, 1.0f);
    color[2] = clamp(color[2], 0.0f, 1.0f);
    if (encode_srgb) {
        color[0] = linear_to_srgb(color[0]);
        color[1] = linear_to_srgb(color[1]);
        color[2] = linear_to_srgb(color[2]);
    }
    color = color * 255.0f + 0.5f;

    pixel_ptr[0] = color[0];
    pixel_ptr[1] = color[1];
    pixel_ptr[2] = color[2];
}

inline void clean_and_write_4_channels(varying float<4> color, uniform uint8* varying pixel_ptr, uniform bool encode_srgb) {
    color[0] = clamp(color[0], 0.0f, 1.0f);
    color[1] = clamp(color[1], 0.0f, 1.0f);
    color[2] = clamp(color[2], 0.0f, 1.0f);
    color[3] = clamp(color[3], 0.0f, 1.0f);
    if (encode_srgb) {
        // Alpha is always linear.
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
    // src->height * dst->width pixels of `scratch_floats()` floats each.
    uniform float* scratch_space;
};

// The horizontal pass writes unclamped float sums to the scratch space, which the vertical pass filters further.
// With alpha weighting a texel holds the alpha weighted rgb sum, the alpha sum and the unweighted rgb sum.
inline uniform uint scratch_floats(uniform uint num_channels, uniform bool alpha_weighted) {
    return alpha_weighted ? 7 : num_channels;
}

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

// Turns the alpha weighted rgb sum back into a colour, once both passes have been summed.
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

inline void resample_with_cached_weights(uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx, uniform bool alpha_weighted) {
    uniform WeightCollection * uniform vertical_weight_collection = ctx->weights.vertical_weights;
    uniform WeightCollection * uniform horizontal_weight_collection = ctx->weights.horizontal_weights;
    const uniform uint num_channels = channel_count(pixel_format);
    // sRGB is filtered in linear space: the source has already been decoded to linear u16 by `linearize_srgb()`,
    // and only the final write encodes back to sRGB u8.
    // Decoding once per texel rather than once per filter tap keeps the cost close to unorm.
    const uniform bool srgb = is_srgb(pixel_format);
    const uniform uint scratch_pixel_floats = scratch_floats(num_channels, alpha_weighted);

    // Accumulate only along the width for each pixel, sampling from the source image
    // Results in the source image being downsampled to src->height X dst->width
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
                color3 += sample_3_channels(src->data + src_read_address, srgb) * weight;
            if (channel_count(pixel_format) == 4)
                accumulate_4_channels(color4, unweighted, sample_4_channels(src->data + src_read_address, srgb), weight, alpha_weighted);
        }

        // Keep full precision and don't clamp: the negative lobes of this pass may only cancel out in the next.
        uniform float* varying scratch = ctx->scratch_space + ((uint64)y * dst->width + x) * scratch_pixel_floats;
        if (channel_count(pixel_format) == 3) {
            scratch[0] = color3[0];
            scratch[1] = color3[1];
            scratch[2] = color3[2];
        }
        if (channel_count(pixel_format) == 4) {
            scratch[0] = color4[0];
            scratch[1] = color4[1];
            scratch[2] = color4[2];
            scratch[3] = color4[3];
            if (alpha_weighted) {
                scratch[4] = unweighted[0];
                scratch[5] = unweighted[1];
                scratch[6] = unweighted[2];
            }
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
            // The horizontal pass already weighted by alpha where needed, so this is a plain weighted sum.
            const uniform float* varying scratch = ctx->scratch_space + ((uint64)scratch_y * dst->width + x) * scratch_pixel_floats;

            if (channel_count(pixel_format) == 3) {
                color3[0] += scratch[0] * weight;
                color3[1] += scratch[1] * weight;
                color3[2] += scratch[2] * weight;
            }
            if (channel_count(pixel_format) == 4) {
                color4[0] += scratch[0] * weight;
                color4[1] += scratch[1] * weight;
                color4[2] += scratch[2] * weight;
                color4[3] += scratch[3] * weight;
                if (alpha_weighted) {
                    unweighted[0] += scratch[4] * weight;
                    unweighted[1] += scratch[5] * weight;
                    unweighted[2] += scratch[6] * weight;
                }
            }
        }

        uint64 out_write_address = (y * dst->width + x) * dst->pixel_stride;
        assert(out_write_address < dst->height * dst->width * dst->pixel_stride);
        if (channel_count(pixel_format) == 3)
            clean_and_write_3_channels(color3, dst->data + out_write_address, srgb);
        if (channel_count(pixel_format) == 4)
            clean_and_write_4_channels(resolve_4_channels(color4, unweighted, alpha_weighted), dst->data + out_write_address, srgb);
    }
}

/// Decodes an sRGB image to tightly packed linear u16, which is what the resample kernels read for sRGB formats.
/// Alpha (the 4th channel) is linear already and only widened.
export void linearize_srgb(const uniform SourceImage * uniform src, uniform uint16 out[], uniform uint32 num_channels) {
    foreach (i = 0 ... src->width * src->height) {
        const uniform uint8* varying pixel_ptr = src->data + (uint64)i * src->pixel_stride;
        uniform uint16* varying out_ptr = out + (uint64)i * num_channels;
        out_ptr[0] = SRGB_TO_LINEAR16[pixel_ptr[0]];
        out_ptr[1] = SRGB_TO_LINEAR16[pixel_ptr[1]];
        out_ptr[2] = SRGB_TO_LINEAR16[pixel_ptr[2]];
        if (num_channels == 4) {
            out_ptr[3] = (uint16)pixel_ptr[3] * 257;
        }
    }
}

/// scratch_space must hold src_height * dst->width * 3 floats
export void resample_with_cached_weights_3(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx) {
    const uniform bool is_normal_map = false;
    assume(channel_count(pixel_format) == 3);
    resample_with_cached_weights(src, dst, pixel_format, ctx, false);
}

/// scratch_space must hold src_height * dst->width * 4 floats
export void resample_with_cached_weights_4(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx) {
    assume(channel_count(pixel_format) == 4);
    resample_with_cached_weights(src, dst, pixel_format, ctx, false);
}

/// Like `resample_with_cached_weights_4`, but weights rgb by alpha.
/// scratch_space must hold src_height * dst->width * 7 floats
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
