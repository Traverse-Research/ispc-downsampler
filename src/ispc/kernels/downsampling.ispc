#include "weights.ispc"
#include "formats.ispc"
#include "image.ispc"
#include "math.ispc"

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

inline void clean_and_write_3_channels(varying float<3> color, uniform uint8* varying pixel_ptr) {
    // The final color is a sum of numbers that are multiplied by the weights of their respective pixels.
    // Because of their numbers, floating point precision leads to the final color being potentially outside of the 0-255 range by a slight margin.
    // This would cause an underflow/overflow, which we avoid with the clamps.

    color[0] = clamp(color[0], 0.0f, 1.0f);
    color[1] = clamp(color[1], 0.0f, 1.0f);
    color[2] = clamp(color[2], 0.0f, 1.0f);
    color *= 255.0f;

    pixel_ptr[0] = color[0];
    pixel_ptr[1] = color[1];
    pixel_ptr[2] = color[2];
}

inline void clean_and_write_4_channels(varying float<4> color, uniform uint8* varying pixel_ptr) {
    color[0] = clamp(color[0], 0.0f, 1.0f);
    color[1] = clamp(color[1], 0.0f, 1.0f);
    color[2] = clamp(color[2], 0.0f, 1.0f);
    color[3] = clamp(color[3], 0.0f, 1.0f);
    color *= 255.0f;

    pixel_ptr[0] = color[0];
    pixel_ptr[1] = color[1];
    pixel_ptr[2] = color[2];
    pixel_ptr[3] = color[3];
}

struct SourceImage {
    uniform PixelFormat pixel_format;
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
    uniform uint8* scratch_space;
};

inline void resample_with_cached_weights(uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform DownsamplingContext * uniform ctx) {
    uniform WeightCollection * uniform vertical_weight_collection = ctx->weights.vertical_weights;
    uniform WeightCollection * uniform horizontal_weight_collection = ctx->weights.horizontal_weights;
    const uniform uint num_channels = channel_count(src->pixel_format);

    // Accumulate only along the width for each pixel, sampling from the source image
    // Results in the source image being downsampled to src->height X dst->width
    foreach_tiled(y =  0 ... src->height, x = 0 ... dst->width) {

        uint32 src_width_start = horizontal_weight_collection->starts[x];
        uint32 num_horizontal_weights = horizontal_weight_collection->weight_counts[x];
        float* horizontal_weights = horizontal_weight_collection->values[x];

        float<3> color3 = {0.0f, 0.0f, 0.0f};
        float<4> color4 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32 i = 0; i < num_horizontal_weights; i++) {
            float weight = horizontal_weights[i];
            uint32 src_x = src_width_start + i;
            uint64 src_read_address = (y * src->width + src_x) * num_channels;

            if (channel_count(src->pixel_format) == 3)
                color3 += sample_3_channels(src->data + src_read_address) * weight;
            if (channel_count(src->pixel_format) == 4)
                color4 += sample_4_channels(src->data + src_read_address) * weight;
        }

        uint64 scratch_write_address = (y * dst->width + x) * num_channels;

        if (channel_count(src->pixel_format) == 3)
            clean_and_write_3_channels(color3, ctx->scratch_space + scratch_write_address);
        if (channel_count(src->pixel_format) == 4)
            clean_and_write_4_channels(color4, ctx->scratch_space + scratch_write_address);
    }
    // Accumulate the scratch space data along the height
    // Downsamples the src_height X dst->width image to target_height * dst->width
    foreach_tiled(y =  0 ... dst->height, x = 0 ... dst->width) {

        uint32 src_height_start = vertical_weight_collection->starts[y];
        uint32 num_vertical_weights = vertical_weight_collection->weight_counts[y];
        float* vertical_weights = vertical_weight_collection->values[y];
        float<3> color3 = {0.0f, 0.0f, 0.0f};
        float<4> color4 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32 i = 0; i < num_vertical_weights; i++) {
            float weight = vertical_weights[i];
            uint32 scratch_y = src_height_start + i;
            uint64 scratch_read_address = (scratch_y * dst->width + x) * num_channels;

            if (channel_count(src->pixel_format) == 3)
                color3 += sample_3_channels(ctx->scratch_space + scratch_read_address) * weight;
            if (channel_count(src->pixel_format) == 4)
                color4 += sample_4_channels(ctx->scratch_space + scratch_read_address) * weight;
        }

        uint64 out_write_address = (y * dst->width + x) * num_channels;
        assert(out_write_address < dst->height * dst->width * num_channels);
        if (channel_count(src->pixel_format) == 3)
            clean_and_write_3_channels(color3, dst->data + out_write_address);
        if (channel_count(src->pixel_format) == 4)
            clean_and_write_4_channels(color4, dst->data + out_write_address);
    }
}

/// scratch_space must be at least src_height * dst->width pixels big
export void resample_with_cached_weights_3(uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform DownsamplingContext * uniform ctx) {
    const uniform bool is_normal_map = false;
    assume(channel_count(src->pixel_format) == 3);
    resample_with_cached_weights(src, dst, ctx);
}

/// scratch_space must be at least src_height * dst->width pixels big
export void resample_with_cached_weights_4(uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform DownsamplingContext * uniform ctx) {
    const uniform bool is_normal_map = false;
    assume(channel_count(src->pixel_format) == 4);
    resample_with_cached_weights(src, dst, ctx);
}

/// scratch_space must be at least src_height * dst->width pixels big
export void downsample_normal_map(uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst) {
    assume(channel_count(src->pixel_format) == 3);
    float<2> ratio = { (float)src->width / (float)dst->width, (float)src->height / (float)dst->height };
    int<2> ceiled = { floor(ratio.x), floor(ratio.y) };
    foreach_tiled(y =  0 ... dst->height, x = 0 ... dst->width) {

        uint<2> start = { 0, 0 };
        uint<2> end = { 0, 0 };
        float pixel_weight = 0.0;

        {
            start.x = max((int)x * ratio.x - ceiled.x, 0);
            start.y = max((int)y * ratio.y - ceiled.y, 0);

            end.x = min((int)x * ratio.x + ceiled.x, src->width);
            end.y = min((int)y * ratio.y + ceiled.y, src->height);

            pixel_weight = (float)((end.x - start.x) * (end.y - start.y));
        }


        float<3> color3 = {0.0f, 0.0f, 0.0f};
        for (uint32 i = start.x; i < end.x; i++) {
            for (uint32 j = start.y; j < end.y; j++) {

                uint64 read_address = (j * src->width + i) * src->pixel_stride;
                color3 += sample_3_channels(src->data + read_address);
            }
        }

        color3 /= pixel_weight;
        color3 = color3 * 2.0f - 1.0f;
        color3 = normalize(color3);
        color3 = color3 * 0.5f + 0.5f;

        uint64 out_write_address = (y * dst->width + x) * dst->pixel_stride;
        assert(out_write_address < dst->height * dst->width * dst->pixel_stride);
        clean_and_write_3_channels(color3, dst->data + out_write_address);
    }
}
