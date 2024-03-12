#include "math.ispc"
#include "formats.ispc"
#include "image.ispc"

const uniform float M_PI = 3.14159265358979;

static inline varying float clean(varying float t)
{
    const varying float EPSILON = .0000125f;
    if (abs(t) < EPSILON)
        return 0.0f;
    return t;
}

static inline varying float sinc(varying float x)
{
    x = x * M_PI;

    // if ((x < 0.01f) && (x > -0.01f))
    //     return 1.0f + x * x * (-1.0f / 6.0f + x * x * 1.0f / 120.0f);

    return sin(x) / x;
}

static inline varying float lanczos3_filter(varying float t, uniform float filter_scale)
{
    t = abs(t);

    if (t < filter_scale)
        return clean(sinc(t) * sinc(t / filter_scale));
    else
        return 0.0f;
}

struct WeightDimensions {
    float src_center;
    float src_start;
    float src_end;
};

export void calculate_weight_variables(uniform float filter_scale, uniform uint32 src, uniform uint32 target, uniform WeightDimensions out_variables[]) {
    uniform float ratio = src / target;

    uniform float filter_radius = ceil(ratio * filter_scale);

    foreach(pixel = 0 ... target) {
        WeightDimensions vars;

        vars.src_center = (pixel + 0.5) * ratio - 0.5;

        float start = ceil(vars.src_center - filter_radius);
        vars.src_start = min(max(start, 0), src - 1);

        float end = floor(vars.src_center + filter_radius);
        end = min(max(end, 0), src - 1);
        vars.src_end = max(end, (float)vars.src_start);

        out_variables[pixel] = vars;
    }
}

export void calculate_weights(uniform float image_scale, uniform float filter_scale, uniform const WeightDimensions * uniform vars, uniform float * uniform weights) {
    uniform float start = vars->src_start;
    uniform float end = vars->src_end;
    uniform float center = vars->src_center;

    // Calculate the Lanczos3 weight of each pixel
    foreach(i = 0 ... end - start + 1) {
        weights[i] = lanczos3_filter((start + (float)i - center) / image_scale, filter_scale);
    }

    // The needs to be done in a normal loop because we cannot apply vectorization to it.
    float sum = 0.0;
    for(int i = 0; i < end - start + 1; i++) {
        sum += weights[i];
    }

    // Normalize the weights, such that their sum is 1.0f
    foreach(i = 0 ... end - start + 1) {
        weights[i] /= sum;
    }
}

struct WeightCollection {
    uniform const uint32* starts;
    uniform const uint32* weight_counts;
    uniform const float* const* values;
};

struct SampleWeights {
    uniform const WeightCollection* vertical_weights;
    uniform const WeightCollection* horizontal_weights;
};

float<3> sample_3_channels(const uniform uint8* varying pixel_ptr, uniform PixelFormat pixel_format) {
    // Memory reinterpretation to read all channels at once rather than one-by-one.
    // While testing, this proved more performant than reading one-by-one.
    const uniform uint8<3>* pixel_ptr3 = (const uniform uint8<3>*)(pixel_ptr);
    varying float<3> dst = {0.0f, 0.0f, 0.0f};
    dst = *pixel_ptr3 / 255.0f;

    if (is_snorm(pixel_format)) {
        dst = dst * 2.0f - 1.0f;
    }
    return dst;
}

float<4> sample_4_channels(const uniform uint8* varying pixel_ptr) {
    const uniform uint8<4>* pixel_ptr4 = (const uniform uint8<4>*)(pixel_ptr);
    varying float<4> dst = {0.0f, 0.0f, 0.0f, 0.0f};
    dst = *pixel_ptr4 / 255.0f;

    return dst;
}

void clean_and_write_3_channels(varying float<3> color, uniform uint8* varying pixel_ptr, uniform PixelFormat pixel_format) {
    // The final color is a sum of numbers that are multiplied by the weights of their respective pixels.
    // Because of their numbers, floating point precision leads to the final color being potentially outside of the 0-255 range by a slight margin.
    // This would cause an underflow/overflow, which we avoid with the clamps.
    if (is_snorm(pixel_format)) {
        // Assumption: Snorm formats are only used for normal maps, so we normalize them to preserve vector length.
        color = normalize(color);
        color = color * 0.5f + 0.5f;
    } else {
        color[0] = clamp(color[0], 0.0f, 1.0f);
        color[1] = clamp(color[1], 0.0f, 1.0f);
        color[2] = clamp(color[2], 0.0f, 1.0f);
    }
    color *= 255.0f;

    pixel_ptr[0] = color[0];
    pixel_ptr[1] = color[1];
    pixel_ptr[2] = color[2];
}

void clean_and_write_4_channels(varying float<4> color, uniform uint8* varying pixel_ptr) {
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


inline void resample_with_cached_weights(uniform uint32 src_width, uniform uint32 src_height, uniform uint32 target_width, uniform uint32 target_height, uniform uint32 pixel_stride,
    uniform const SampleWeights * uniform cached_weights, uniform uint8 scratch_space[], uniform const uint8 src_data[], uniform uint8 out_data[], uniform PixelFormat pixel_format) {
    uniform WeightCollection * uniform vertical_weight_collection = cached_weights->vertical_weights;
    uniform WeightCollection * uniform horizontal_weight_collection = cached_weights->horizontal_weights;
    const uniform uint num_channels = channel_count(pixel_format);

    // Accumulate only along the width for each pixel, sampling from the source image
    // Results in the source image being downsampled to src_height X target_width
    foreach_tiled(y =  0 ... src_height, x = 0 ... target_width) {

        uint32 src_width_start = horizontal_weight_collection->starts[x];
        uint32 num_horizontal_weights = horizontal_weight_collection->weight_counts[x];
        float* horizontal_weights = horizontal_weight_collection->values[x];

        float<3> color3 = {0.0f, 0.0f, 0.0f};
        float<4> color4 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32 i = 0; i < num_horizontal_weights; i++) {
            float weight = horizontal_weights[i];
            uint32 src_x = src_width_start + i;
            uint64 src_read_address = (y * src_width + src_x) * num_channels;

            if (channel_count(pixel_format) == 3)
                color3 += sample_3_channels(src_data + src_read_address, pixel_format) * weight;
            if (channel_count(pixel_format) == 4)
                color4 += sample_4_channels(src_data + src_read_address) * weight;
        }

        uint64 scratch_write_address = (y * target_width + x) * num_channels;

        if (channel_count(pixel_format) == 3)
            clean_and_write_3_channels(color3, scratch_space + scratch_write_address, pixel_format);
        if (channel_count(pixel_format) == 4)
            clean_and_write_4_channels(color4, scratch_space + scratch_write_address);
    }
    // Accumulate the scratch space data along the height
    // Downsamples the src_height X target_width image to target_height * target_width
    foreach_tiled(y =  0 ... target_height, x = 0 ... target_width) {

        uint32 src_height_start = vertical_weight_collection->starts[y];
        uint32 num_vertical_weights = vertical_weight_collection->weight_counts[y];
        float* vertical_weights = vertical_weight_collection->values[y];
        float<3> color3 = {0.0f, 0.0f, 0.0f};
        float<4> color4 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32 i = 0; i < num_vertical_weights; i++) {
            float weight = vertical_weights[i];
            uint32 scratch_y = src_height_start + i;
            uint64 scratch_read_address = (scratch_y * target_width + x) * num_channels;

            uniform uint8<3>* varying scratch_pixel_ptr = (uniform uint8<3>* varying)(scratch_space + scratch_read_address);
            uint8<3> scratch_color = *scratch_pixel_ptr;

            if (channel_count(pixel_format) == 3)
                color3 += sample_3_channels(scratch_space + scratch_read_address, pixel_format) * weight;
            if (channel_count(pixel_format) == 4)
                color4 += sample_4_channels(scratch_space + scratch_read_address) * weight;
        }

        uint64 out_write_address = (y * target_width + x) * num_channels;
        assert(out_write_address < target_height * target_width * num_channels);
        if (channel_count(pixel_format) == 3)
            clean_and_write_3_channels(color3, out_data + out_write_address, pixel_format);
        if (channel_count(pixel_format) == 4)
            clean_and_write_4_channels(color4, out_data + out_write_address);
    }
}

/// scratch_space must be at least src_height * target_width pixels big
export void resample_with_cached_weights_3(uniform uint32 src_width, uniform uint32 src_height, uniform uint32 target_width, uniform uint32 target_height, uniform uint32 pixel_stride,
    uniform const SampleWeights * uniform cached_weights, uniform uint8 scratch_space[], uniform const uint8 src_data[], uniform uint8 out_data[], uniform PixelFormat pixel_format) {
    assume(channel_count(pixel_format) == 3);
    resample_with_cached_weights(src_width, src_height, target_width, target_height, pixel_stride, cached_weights, scratch_space, src_data, out_data, pixel_format);
}

/// scratch_space must be at least src_height * target_width pixels big
export void resample_with_cached_weights_4(uniform uint32 src_width, uniform uint32 src_height, uniform uint32 target_width, uniform uint32 target_height, uniform uint32 pixel_stride,
    uniform const SampleWeights * uniform cached_weights, uniform uint8 scratch_space[], uniform const uint8 src_data[], uniform uint8 out_data[], uniform PixelFormat pixel_format) {
    assume(channel_count(pixel_format) == 4);
    resample_with_cached_weights(src_width, src_height, target_width, target_height, pixel_stride, cached_weights, scratch_space, src_data, out_data, pixel_format);
}
