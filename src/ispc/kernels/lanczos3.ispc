#define M_PI 3.14159265358979

static float clean(float t)
{
    const float EPSILON = .0000125f;
    if (abs(t) < EPSILON)
        return 0.0f;
    return (float)t;
}

static float sinc(float x)
{
    x = (x * M_PI);

    if ((x < 0.01f) && (x > -0.01f))
        return 1.0f + x * x * (-1.0f / 6.0f + x * x * 1.0f / 120.0f);

    return sin(x) / x;
}

static float lanczos3_filter(float t)
{
    if (t < 0.0f)
        t = -t;

    if (t < 3.0f)
        return clean(sinc(t) * sinc(t / 3.0f));
    else
        return (0.0f);
}
struct Image {
    uint8* data;
    int<2> size;
};
static int<2> uv_to_pixel_id(int<2> image_size, float<2> uv) {
    float<2> pixel_coord = uv * image_size;
    int<2> i = {floor(pixel_coord.x + 0.5), floor(pixel_coord.y + 0.5)};
    i.x = max(0, min(image_size.x - 1, i.x));
    i.y = max(0, min(image_size.y - 1, i.y));
    return i;
}
static float frac(float f) {
    float absf = abs(f);
    return absf - floor(absf);
}
static uint8<4> resample_internal(Image src_image, float<2> uv, float<2> invTargetResolution) {

    float<2> center_pixel = uv / invTargetResolution;
    center_pixel.x = frac(center_pixel.x);
    center_pixel.y = frac(center_pixel.y);
    float<2> center = uv - (center_pixel - 0.5) * invTargetResolution;
    float<2> offset = (uv - center) / invTargetResolution;
    
    float<4> col = { 0, 0, 0, 0 };
    float weight = 0.0;
    for(uniform int x = -3; x < 3; x++){
        for(uniform int y = -3; y < 3; y++){
            
            float wx = lanczos3_filter((float)x - offset.x);
            float wy = lanczos3_filter((float)y - offset.y);
            float w = wx * wy;
            float<2> texel_offset = {x, y};
            float<2> texel_uv = center + texel_offset * invTargetResolution;
            int<2> pixel_coord = uv_to_pixel_id(src_image.size, texel_uv);
            float<4> texel;
            for (int i = 0; i < 4; i++) 
                texel[i] = (float)src_image.data[(pixel_coord.x + pixel_coord.y * src_image.size.x) * 4 + i] / 255.0f;
            col += w * texel;
            weight += w;
        }
    }
    col /= weight;
    return col * 255;
}

export void resample(uniform uint width, uniform uint height, uniform uint stride, uniform uint target_width, uniform uint target_height, uniform const uint8 src_data[], uniform uint8 out_data[]) {
    uniform Image src = {src_data, {width, height}};
    uniform Image dst = {out_data, {target_width, target_height}};
    foreach_tiled (y = 0 ... target_height, x = 0 ... target_width) {

        float yf = y / (float)target_height;
        float xf = x / (float)target_width;

        float<2> uv = { xf, yf };

        int<2> target_size = {target_width, target_height};

        int<2> dst_pixel_id = uv_to_pixel_id(target_size, uv);
        
// src.data[x + y * stride]
        //float r = srgb8_to_float(src.data[x + y * stride]);
        //float r_filtered = lanczos3_filter(r);
        //dst.data[x + y * target_width] = (uint8)float_to_srgb8(r_filtered);
        uint8<4> s = resample_internal(src, uv, 1.0 / target_size);
        for (int i = 0; i < 4; i++) 
            dst.data[(x + y * target_width) * 4 + i] = s[i]; 
    }
}