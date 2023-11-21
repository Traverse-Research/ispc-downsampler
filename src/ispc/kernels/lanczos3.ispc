#include "image.ispc"

const uniform float M_PI = 3.14159265358979f;
const uniform float GAMMA = 2.2f;
const uniform float DEGAMMA = 1.0f / GAMMA;

struct Parameters {
    // Whether to linearize the input before downsampling.  Assumes the input has a gamma of 1/2.2
    // that needs to be linearized by applying exponent 2.2.
    bool degamma;
    // Whether to apply gamma (make the output nonlinear) to make it compatible with typical CRTs
    // that have a gamma of 2.2, by giving linear values a gamma of 1/2.2.
    bool gamma;
};

static inline uniform float clean(uniform float t)
{
    const uniform float EPSILON = .0000125f;
    if (abs(t) < EPSILON)
        return 0.0f;
    return t;
}

static inline uniform float sinc(uniform float x)
{
    x = x * M_PI;

    // if ((x < 0.01f) && (x > -0.01f))
    //     return 1.0f + x * x * (-1.0f / 6.0f + x * x * 1.0f / 120.0f);

    return sin(x) / x;
}

static inline uniform float lanczos3_filter(uniform float t)
{
    t = abs(t);

    if (t < 3.0)
        return clean(sinc(t) * sinc(t / 3.0));
    else
        return 0.0f;
}

static inline float byte_to_float(uint8 b/*, uniform bool degamma*/) {
    const uniform float inv_255 = rcp(255.0);
    // floatbits(0x3f800000 | (b << (23 - 8))) - 1.0;
    return (float)b * inv_255;
}

static inline uint8 float_to_byte(float d, uniform bool gamma) {
    if (gamma) {
        d = pow(d, DEGAMMA);
    }
    int b = d * 255;
    return clamp(b, 0, 255);
}

template<typename IT>
static float<4> sample_image(const uniform IT &image, const int<2> coord, const uniform uint8 num_channels) {
    return 0.0f;
}

template<>
static float<4> sample_image<Image>(const uniform Image &image, const int<2> coord, const uniform uint8 num_channels) {
    float<4> col = 0.0;
    int x = clamp(coord.x, 0, image.size.x - 1);
    int y = clamp(coord.y, 0, image.size.y - 1);
    int addr = (x + y * image.size.x) * num_channels;

    col[0] = byte_to_float(image.data[addr + 0]);
    col[1] = byte_to_float(image.data[addr + 1]);
    col[2] = byte_to_float(image.data[addr + 2]);
    if (num_channels == 4)
        col[3] = byte_to_float(image.data[addr + 3]);

    return col;
}

template<>
static float<4> sample_image<FloatImage>(const uniform FloatImage &image, const int<2> coord, const uniform uint8 num_channels) {
    float<4> col = 0.0;
    int x = clamp(coord.x, 0, image.size.x - 1);
    int y = clamp(coord.y, 0, image.size.y - 1);
    int addr = (x + y * image.size.x) * num_channels;

    col[0] = image.data[addr + 0];
    col[1] = image.data[addr + 1];
    col[2] = image.data[addr + 2];
    if (num_channels == 4)
        col[3] = image.data[addr + 3];

    return col;
}

template<typename IT>
static inline float<4> resample_internal(const uniform IT src_image, const float<2> uv, const uniform uint8 num_channels) {
    float<4> col = 0.0;
    uniform float weight = 0.0;
    // Truncate floating point coordinate to integer:
    const int<2> src_coord = uv * src_image.size;

    // Sample the lanczos3 filter from -2.5 to 2.5.
    // Pixel 0, 0 in the source will be at the bottom-right of the center of the target pixel,
    // hence reading source pixels in the range [-3, 2] will correctly read pixels to the left, top,
    // right and bottom of the target pixel.
    for (uniform int x = -3; x < 3; x++) {
        for (uniform int y = -3; y < 3; y++) {
            const uniform float wx = lanczos3_filter((uniform float)x + 0.5);
            const uniform float wy = lanczos3_filter((uniform float)y + 0.5);
            const uniform float w = wx * wy;
            const uniform int<2> texel_offset = {x, y};

            int<2> c = src_coord + texel_offset;
            // TODO: Let the user specify a boundary mode!
            // https://github.com/Traverse-Research/ispc-downsampler/issues/25#issuecomment-1584915050
            // TODO: For some obscure reason this must happen in sample_image() or the whole thing segfaults because
            // values become <0 !?!?
            // c.x = clamp(c.x, 0, src_image.size.x - 1);
            // c.y = clamp(c.y, 0, src_image.size.y - 1);

            weight += w;
            col += w * sample_image<IT>(src_image, c, num_channels);
        }
    }
    col /= weight;
    return col;
}

export void resample(
    uniform const Parameters &params,
    uniform const Image &src,
    uniform FloatImage &degamma,
    uniform Image &dst,
    // Passed separately because it should be the same between input and output:
    uniform uint8 num_channels
) {
    const uniform float<2> inv_target_size = 1.0f / dst.size;

    if (params.degamma) {
        foreach_tiled(y = 0 ... src.size.y, x = 0 ... src.size.x)
        {
            uint p = (x + y * src.size.x) * num_channels;
            for (uniform int i = 0; i < num_channels; i++) {
                uint c = p + i;
                degamma.data[c] = pow(byte_to_float(src.data[c]), GAMMA);
            }
        }
    }

    foreach_tiled (y = 0 ... dst.size.y, x = 0 ... dst.size.x) {
        float<2> uv = {x, y};
        // Use the center of each pixel, not the top-left:
        uv += 0.5f;
        // Convert to uniform space:
        uv *= inv_target_size;

        float<4> col;
        if (params.degamma)
            col = resample_internal(degamma, uv, num_channels);
        else
            col = resample_internal(src, uv, num_channels);

        for (uniform int i = 0; i < num_channels; i++)
            dst.data[(x + y * dst.size.x) * num_channels + i] = float_to_byte(col[i], params.gamma);
    }
}
