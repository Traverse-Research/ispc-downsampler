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

static inline float lanczos3_filter(float t)
{
    t = abs(t);

    if (t < 3.0)
        return clean(sinc(t) * sinc(t / 3.0));
    else
        return 0.0f;
}

struct Image {
    uniform const uint8* data;
    uniform int<2> size;
};

static inline int<2> uv_to_pixel_id(int<2> image_size, float<2> uv) {
    float<2> pixel_coord = uv * image_size;
    int<2> i = {floor(pixel_coord.x + 0.5), floor(pixel_coord.y + 0.5)};
    i.x = clamp(i.x, 0, image_size.x - 1);
    i.y = clamp(i.y, 0, image_size.y - 1);
    return i;
}

static inline float frac(float f) {
    float absf = abs(f);
    return absf - floor(absf);
}

static inline float byte_to_float(uint b) {
    //return floatbits(0x3f800000 | (b << (23 - 8))) - 1.0;
    return (float)b;
}

static inline uint8<4> resample_internal(uniform Image src_image, float<2> uv, uniform float<2> target_size, uniform float<2> inv_target_size, uniform uint8 num_channels) {
    float<2> center_pixel = uv * target_size;
    center_pixel.x = frac(center_pixel.x);
    center_pixel.y = frac(center_pixel.y);
    float<2> center = uv - (center_pixel - 0.5) * inv_target_size;
    float<2> offset = (uv - center) * target_size;
    const float inv_255 = rcp(255.0);

    float<4> col = { 0, 0, 0, 0 };
    float weight = 0.0;

    for(uniform int x = -3; x < 3; x++){
        for(uniform int y = -3; y < 3; y++){

            float wx = lanczos3_filter((uniform float)x - offset.x);
            float wy = lanczos3_filter((uniform float)y - offset.y);
            float w = wx * wy;
            float<2> texel_offset = {x, y};
            float<2> texel_uv = center + texel_offset * inv_target_size;
            int<2> pixel_coord = uv_to_pixel_id(src_image.size, texel_uv);

            int addr = (pixel_coord.x + pixel_coord.y * src_image.size.x) * num_channels;

            float<4> texel;

            if (num_channels == 3) {
                texel.x = byte_to_float(src_image.data[addr + 0]) * inv_255;
                texel.y = byte_to_float(src_image.data[addr + 1]) * inv_255;
                texel.z = byte_to_float(src_image.data[addr + 2]) * inv_255;
            } else if (num_channels == 4) {
                texel.x = byte_to_float(src_image.data[addr + 0]) * inv_255;
                texel.y = byte_to_float(src_image.data[addr + 1]) * inv_255;
                texel.z = byte_to_float(src_image.data[addr + 2]) * inv_255;
                texel.w = byte_to_float(src_image.data[addr + 3]) * inv_255;
            }

            col += w * texel;
            weight += w;
        }
    }
    col /= weight;
    return col * 255;
}

export void resample(uniform uint32 width, uniform uint32 height, uniform uint32 stride, uniform uint8 num_channels, uniform uint32 target_width, uniform uint32 target_height, uniform const uint8 src_data[], uniform uint8 out_data[]) {
    uniform Image src = {src_data, {width, height}};
    uniform float<2> target_size = {(float)target_width, (float)target_height};
    uniform float<2> inv_target_size = {1.0 / target_width, 1.0 / target_height};

    foreach_tiled (y = 0 ... target_height, x = 0 ... target_width) {
        float yf = y / target_size.y;
        float xf = x / target_size.x;

        float<2> uv = { xf, yf };

        uint8<4> s = resample_internal(src, uv, target_size, inv_target_size, num_channels);

        for (uniform int i = 0; i < num_channels; i++)
            out_data[(x + y * target_width) * num_channels + i] = s[i];
    }
}

struct WeightVariables {
    float src_center;
    float src_start;
    float src_end;
};

export void calculate_weight_variables(uniform uint src, uniform uint target, uniform WeightVariables out_variables[]) {
    uniform float ratio = src / target;

    uniform float filter_radius = ceil(ratio * 3);

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

export void calculate_weights(uniform float scale, uniform const WeightVariables * uniform vars, uniform float * uniform weights) {

    uniform float start = vars->src_start;
    uniform float end = vars->src_end;
    uniform float center = vars->src_center;

    foreach(i = 0 ... end - start + 1) {
        weights[i] = lanczos3_filter((start + (float)i - center) / scale);
    }
    float sum = 0.0;
    for(int i = 0; i < end - start + 1; i++) {
        sum += weights[i];
    }

    foreach(i = 0 ... end - start + 1) {
        weights[i] /= sum;
    }
}

struct WeightCollection {
    uniform const uint* starts;
    uniform const uint* weight_counts;
    uniform const float* const* values;
};

struct Cache {
    uniform WeightCollection vertical_weights;
    uniform WeightCollection horizontal_weights;
};

uint8<4> sample_3_channels(varying uint64 read_address, const uniform uint8* varying src) {
    const uniform uint8* pixel_ptr = src + read_address;
    const uniform uint8<3>* pixel_ptr3 = (const uniform uint8<3>*)(pixel_ptr);
    varying uint8<4> dst = {0, 0, 0, 0};
    varying uint8<3>* varying dst_ptr = (varying uint8<3>* varying)&dst;
    *dst_ptr = *pixel_ptr3;
    return dst;
}

void clean_and_write_3_channels(varying float<4> color, varying write_address, uniform uint8* varying dst) {
    color.x = clamp(color.x, 0.0f, 255.0f);
    color.y = clamp(color.y, 0.0f, 255.0f);
    color.z = clamp(color.z, 0.0f, 255.0f);

    uniform uint8* pixel_ptr = dst + write_address;
    uniform uint8<3>* pixel_ptr3 = (uniform uint8<3>*)(pixel_ptr);
    varying float<3>* varying col_ptr3 = (varying float<3>* varying)&color;
    *pixel_ptr3 = *col_ptr3;
}

uint8<4> sample_4_channels(varying uint64 read_address, const uniform uint8* varying src) {
    const uniform uint8* pixel_ptr = src + read_address;
    const uniform uint8<4>* pixel_ptr4 = (const uniform uint8<4>*)(pixel_ptr);
    varying uint8<4> dst = {0, 0, 0, 0};
    dst = *pixel_ptr4;
    return dst;
}

void clean_and_write_4_channels(varying float<4> color, varying write_address, uniform uint8* varying dst) {
    color.x = clamp(color.x, 0.0f, 255.0f);
    color.y = clamp(color.y, 0.0f, 255.0f);
    color.z = clamp(color.z, 0.0f, 255.0f);
    color.w = clamp(color.w, 0.0f, 255.0f);

    uniform uint8* pixel_ptr = dst + write_address;
    uniform uint8<4>* pixel_ptr4 = (uniform uint8<4>*)(pixel_ptr);
    *pixel_ptr4 = color;
}

/// scratch_space must be at least src_height * target_width pixels big
export void resample_with_cache(uniform uint src_width, uniform uint src_height, uniform uint target_width, uniform uint target_height, uniform uint8 num_channels,
    uniform const Cache * uniform cache, uniform uint8 scratch_space[], uniform const uint8 src_data[], uniform uint8 out_data[]) {

    uniform WeightCollection * uniform vertical_weight_collection = &cache->vertical_weights;
    uniform WeightCollection * uniform horizontal_weight_collection = &cache->horizontal_weights;

    // Accumulate only along the width for each pixel, sampling from the source image
    // Results in the source image being downsampled to src_height X target_width
    foreach_tiled(y =  0 ... src_height, x = 0 ... target_width) {
        uint src_width_start = horizontal_weight_collection->starts[x];
        uint num_horizontal_weights = horizontal_weight_collection->weight_counts[x];
        float* horizontal_weights = horizontal_weight_collection->values[x];

        float<4> color = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = 0; i < num_horizontal_weights; i++) {
            float weight = horizontal_weights[i];
            uint src_x = src_width_start + i;
            uint64 src_read_address = (y * src_width + src_x) * num_channels;

            color += sample_3_channels(src_read_address, src_data) * weight;
        }

        uint64 scratch_write_address = (y * target_width + x) * num_channels;

        clean_and_write_3_channels(color, scratch_write_address, scratch_space);
    }

    // Accumulate the scratch space data along the height
    // Downsamples the src_height X target_width image to target_height * target_width
    foreach_tiled(y =  0 ... target_height, x = 0 ... target_width) {

        uint src_height_start = vertical_weight_collection->starts[y];
        uint num_vertical_weights = vertical_weight_collection->weight_counts[y];
        float* vertical_weights = vertical_weight_collection->values[y];

        float<4> color = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = 0; i < num_vertical_weights; i++) {
            float weight = vertical_weights[i];
            uint scratch_y = src_height_start + i;
            uint64 scratch_read_address = (scratch_y * target_width + x) * num_channels;

            uniform uint8<3>* varying scratch_pixel_ptr = (uniform uint8<3>* varying)(scratch_space + scratch_read_address);
            uint8<3> scratch_color = *scratch_pixel_ptr;

            color += sample_3_channels(scratch_read_address, scratch_space) * weight;
        }

        uint64 out_write_address = (y * target_height + x) * num_channels;
        clean_and_write_3_channels(color, out_write_address, out_data);
    }
}
