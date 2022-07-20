#define M_PI 3.14159265358979

static inline float clean(float t)
{
    const float EPSILON = .0000125f;
    if (abs(t) < EPSILON)
        return 0.0f;
    return (float)t;
}

static inline float sinc(float x)
{
    x = (x * M_PI);

    // if ((x < 0.01f) && (x > -0.01f))
    //     return 1.0f + x * x * (-1.0f / 6.0f + x * x * 1.0f / 120.0f);

    return sin(x) / x;
}

static inline float lanczos3_filter(float t, uniform float filter_scale)
{
    t = abs(t);

    if (t < filter_scale)
        return clean(sinc(t) * sinc(t / filter_scale));
    else
        return 0.0f;
}

struct WeightVariables {
    float src_center;
    float src_start;
    float src_end;
};

export void calculate_weight_variables(uniform float filter_scale, uniform uint32 src, uniform uint32 target, uniform WeightVariables out_variables[]) {
    uniform float ratio = src / target;

    uniform float filter_radius = ceil(ratio * filter_scale);

    foreach(pixel = 0 ... target) {
        WeightVariables vars;

        vars.src_center = (pixel + 0.5) * ratio - 0.5;

        float start = ceil(vars.src_center - filter_radius);
        vars.src_start = min(max(start, 0), src - 1);

        float end = floor(vars.src_center + filter_radius);
        end = min(max(end, 0), src - 1);
        vars.src_end = max(end, (float)vars.src_start);

        out_variables[pixel] = vars;
    }
}

export void calculate_weights(uniform float image_scale, uniform float filter_scale, uniform const WeightVariables * uniform vars, uniform float * uniform weights) {

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

struct Cache {
    uniform WeightCollection vertical_weights;
    uniform WeightCollection horizontal_weights;
};

uint8<3> sample_3_channels(varying uint64 read_address, const uniform uint8* varying src) {
    // Pointer hacks to read all channels at once rather than one-by-one, despite technically reading uint8<3> to a uint8<4>.
    // While testing, this proved more performant than reading one-by-one.
    const uniform uint8* pixel_ptr = src + read_address;
    const uniform uint8<3>* pixel_ptr3 = (const uniform uint8<3>*)(pixel_ptr);
    varying uint8<3> dst = {0, 0, 0};
    dst = *pixel_ptr3;
    return dst;
}

void clean_and_write_3_channels(varying float<3> color, varying uint64 write_address, uniform uint8* varying dst) {
    // The final color is a sum of numbers that are multiplied by the weights of their respective pixels.
    // Because of their numbers, floating point precision leads to the final color being potentially outside of the 0-255 range by a slighty margin.
    // This would cause an underflow/overflow, which we avoid with the clamps.
    color.x = clamp(color.x, 0.0f, 255.0f);
    color.y = clamp(color.y, 0.0f, 255.0f);
    color.z = clamp(color.z, 0.0f, 255.0f);

    // Pointer hacks to write all channels at once rather than one-by-one, despite technically writing uint8<4> to a uint8<3>.
    // While testing, this proved more performant than writing one-by-one.
    uniform uint8* pixel_ptr = dst + write_address;
    uniform uint8<3>* pixel_ptr3 = (uniform uint8<3>*)(pixel_ptr);
    *pixel_ptr3 = color;
}

uint8<4> sample_4_channels(varying uint64 read_address, const uniform uint8* varying src) {
    const uniform uint8* pixel_ptr = src + read_address;
    const uniform uint8<4>* pixel_ptr4 = (const uniform uint8<4>*)(pixel_ptr);
    varying uint8<4> dst = {0, 0, 0, 0};
    dst = *pixel_ptr4;
    return dst;
}

void clean_and_write_4_channels(varying float<4> color, varying uint64 write_address, uniform uint8* varying dst) {
    color.x = clamp(color.x, 0.0f, 255.0f);
    color.y = clamp(color.y, 0.0f, 255.0f);
    color.z = clamp(color.z, 0.0f, 255.0f);
    color.w = clamp(color.w, 0.0f, 255.0f);

    uniform uint8* pixel_ptr = dst + write_address;
    uniform uint8<4>* pixel_ptr4 = (uniform uint8<4>*)(pixel_ptr);
    *pixel_ptr4 = color;
}

/// scratch_space must be at least src_height * target_width pixels big
export void resample_with_cache_3(uniform uint32 src_width, uniform uint32 src_height, uniform uint32 target_width, uniform uint32 target_height,
    uniform const Cache * uniform cache, uniform uint8 scratch_space[], uniform const uint8 src_data[], uniform uint8 out_data[]) {
    uniform const uint32 num_channels = 3;
    uniform WeightCollection * uniform vertical_weight_collection = &cache->vertical_weights;
    uniform WeightCollection * uniform horizontal_weight_collection = &cache->horizontal_weights;

    // Accumulate only along the width for each pixel, sampling from the source image
    // Results in the source image being downsampled to src_height X target_width
    foreach_tiled(y =  0 ... src_height, x = 0 ... target_width) {
        uint32 src_width_start = horizontal_weight_collection->starts[x];
        uint32 num_horizontal_weights = horizontal_weight_collection->weight_counts[x];
        float* horizontal_weights = horizontal_weight_collection->values[x];

        float<3> color = {0.0f, 0.0f, 0.0f};
        for (uint32 i = 0; i < num_horizontal_weights; i++) {
            float weight = horizontal_weights[i];
            uint32 src_x = src_width_start + i;
            uint64 src_read_address = (y * src_width + src_x) * num_channels;

            color += sample_3_channels(src_read_address, src_data) * weight;
        }

        uint64 scratch_write_address = (y * target_width + x) * num_channels;

        clean_and_write_3_channels(color, scratch_write_address, scratch_space);
    }

    // Accumulate the scratch space data along the height
    // Downsamples the src_height X target_width image to target_height * target_width
    foreach_tiled(y =  0 ... target_height, x = 0 ... target_width) {

        uint32 src_height_start = vertical_weight_collection->starts[y];
        uint32 num_vertical_weights = vertical_weight_collection->weight_counts[y];
        float* vertical_weights = vertical_weight_collection->values[y];

        float<3> color = {0.0f, 0.0f, 0.0f};
        for (uint32 i = 0; i < num_vertical_weights; i++) {
            float weight = vertical_weights[i];
            uint32 scratch_y = src_height_start + i;
            uint64 scratch_read_address = (scratch_y * target_width + x) * num_channels;

            uniform uint8<3>* varying scratch_pixel_ptr = (uniform uint8<3>* varying)(scratch_space + scratch_read_address);
            uint8<3> scratch_color = *scratch_pixel_ptr;

            color += sample_3_channels(scratch_read_address, scratch_space) * weight;
        }

        uint64 out_write_address = (y * target_height + x) * num_channels;

        clean_and_write_3_channels(color, out_write_address, out_data);
    }
}

/// scratch_space must be at least src_height * target_width pixels big
export void resample_with_cache_4(uniform uint32 src_width, uniform uint32 src_height, uniform uint32 target_width, uniform uint32 target_height,
    uniform const Cache * uniform cache, uniform uint8 scratch_space[], uniform const uint8 src_data[], uniform uint8 out_data[]) {
    uniform const uint32 num_channels = 4;

    uniform WeightCollection * uniform vertical_weight_collection = &cache->vertical_weights;
    uniform WeightCollection * uniform horizontal_weight_collection = &cache->horizontal_weights;

    // Accumulate only along the width for each pixel, sampling from the source image
    // Results in the source image being downsampled to src_height X target_width
    foreach_tiled(y =  0 ... src_height, x = 0 ... target_width) {
        uint32 src_width_start = horizontal_weight_collection->starts[x];
        uint32 num_horizontal_weights = horizontal_weight_collection->weight_counts[x];
        float* horizontal_weights = horizontal_weight_collection->values[x];

        float<4> color = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32 i = 0; i < num_horizontal_weights; i++) {
            float weight = horizontal_weights[i];
            uint32 src_x = src_width_start + i;
            uint64 src_read_address = (y * src_width + src_x) * num_channels;

            color += sample_4_channels(src_read_address, src_data) * weight;
        }

        uint64 scratch_write_address = (y * target_width + x) * num_channels;

        clean_and_write_4_channels(color, scratch_write_address, scratch_space);
    }

    // Accumulate the scratch space data along the height
    // Downsamples the src_height X target_width image to target_height * target_width
    foreach_tiled(y =  0 ... target_height, x = 0 ... target_width) {

        uint32 src_height_start = vertical_weight_collection->starts[y];
        uint32 num_vertical_weights = vertical_weight_collection->weight_counts[y];
        float* vertical_weights = vertical_weight_collection->values[y];

        float<4> color = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32 i = 0; i < num_vertical_weights; i++) {
            float weight = vertical_weights[i];
            uint32 scratch_y = src_height_start + i;
            uint64 scratch_read_address = (scratch_y * target_width + x) * num_channels;

            uniform uint8<3>* varying scratch_pixel_ptr = (uniform uint8<3>* varying)(scratch_space + scratch_read_address);
            uint8<3> scratch_color = *scratch_pixel_ptr;

            color += sample_4_channels(scratch_read_address, scratch_space) * weight;
        }

        uint64 out_write_address = (y * target_height + x) * num_channels;

        clean_and_write_4_channels(color, out_write_address, out_data);
    }
}
