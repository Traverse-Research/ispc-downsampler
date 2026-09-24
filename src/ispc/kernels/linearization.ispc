#pragma once

// sRGB transfer functions as specified by IEC 61966-2-1, operating on normalized [0, 1] values.

inline uniform float srgb_to_linear(uniform float srgb) {
    if (srgb <= 0.04045f)
        return srgb / 12.92f;
    return pow((srgb + 0.055f) / 1.055f, 2.4f);
}

inline varying float linear_to_srgb(varying float linear) {
    if (linear <= 0.0031308f)
        return linear * 12.92f;
    return 1.055f * pow(linear, 1.0f / 2.4f) - 0.055f;
}

// Lookup table to convert 8-bit sRGB-encoded values to normalized linear values.
// Much cheaper than calling `pow()` for every filter tap.
struct SrgbToLinearLut {
    uniform float values[256];
};

inline void init_srgb_to_linear_lut(uniform SrgbToLinearLut * uniform lut) {
    for (uniform int i = 0; i < 256; i++)
        lut->values[i] = srgb_to_linear((uniform float)i / 255.0f);
}
