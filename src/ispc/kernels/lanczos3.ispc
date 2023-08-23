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

static inline float frac(float f) {
    float absf = abs(f);
    return absf - floor(absf);
}

static inline float byte_to_float(uint b) {
    //return floatbits(0x3f800000 | (b << (23 - 8))) - 1.0;
    return (float)b;
}

static inline uint8<4> resample_internal(uniform Image src_image, float<2> uv, uniform uint8 num_channels) {
    float<4> col = { 0, 0, 0, 0 };
    float weight = 0.0;

    for (uniform int x = -3; x <= 3; x++) {
        for (uniform int y = -3; y <= 3; y++) {
            const float lanczos3_offset = 0.5;

            float wx = lanczos3_filter((uniform float)x + lanczos3_offset);
            float wy = lanczos3_filter((uniform float)y + lanczos3_offset);
            float w = wx * wy;
            float<2> texel_offset = {x, y};
            int<2> src_coord = uv * src_image.size + texel_offset;
            // TODO: Let the user specify a boundary mode!
            // https://github.com/Traverse-Research/ispc-downsampler/issues/25#issuecomment-1584915050
            src_coord.x = clamp(src_coord.x, 0, src_image.size.x - 1);
            src_coord.y = clamp(src_coord.y, 0, src_image.size.y - 1);

            int addr = (src_coord.x + src_coord.y * src_image.size.x) * num_channels;

            float<4> texel;

            const float inv_255 = rcp(255.0);

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
    uniform float<2> inv_target_size = 1.0f / target_size;

    foreach_tiled (y = 0 ... target_height, x = 0 ... target_width) {
        float<2> uv = {x, y};
        // Use the center of each pixel, not the top-left:
        uv += 0.5f;
        // Convert to uniform space:
        uv *= inv_target_size;

        uint8<4> s = resample_internal(src, uv, num_channels);

        for (uniform int i = 0; i < num_channels; i++)
            out_data[(x + y * target_width) * num_channels + i] = s[i];
    }
}
