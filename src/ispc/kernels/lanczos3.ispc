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

static inline uint8<4> resample_internal(uniform Image src_image, uniform float<2> inv_src_size, float<2> uv, uniform float<2> target_size, uniform float<2> inv_target_size, uniform uint8 num_channels) {
    float<2> center_pixel = uv * target_size;
    center_pixel.x = frac(center_pixel.x);
    center_pixel.y = frac(center_pixel.y);
    float<2> center = uv - (center_pixel - 0.5) * inv_target_size;
    float<2> offset = (uv - center) * target_size;
    const float inv_255 = rcp(255.0);

    float<4> col = { 0, 0, 0, 0 };
    float weight = 0.0;

    for (uniform int x = -3; x <= 3; x++) {
        for (uniform int y = -3; y <= 3; y++) {
            float wx = lanczos3_filter((uniform float)x - offset.x);
            float wy = lanczos3_filter((uniform float)y - offset.y);
            float w = wx * wy;
            float<2> texel_offset = {x, y};
            float<2> texel_uv = center + texel_offset * inv_src_size;
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
    uniform float<2> inv_src_size = 1.0f / src.size;

    foreach_tiled (y = 0 ... target_height, x = 0 ... target_width) {
        float yf = y / target_size.y;
        float xf = x / target_size.x;

        float<2> uv = { xf, yf };

        uint8<4> s = resample_internal(src, inv_src_size, uv, target_size, inv_target_size, num_channels);

        for (uniform int i = 0; i < num_channels; i++)
            out_data[(x + y * target_width) * num_channels + i] = s[i];
    }
}
