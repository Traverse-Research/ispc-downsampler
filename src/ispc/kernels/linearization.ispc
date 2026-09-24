#pragma once

// sRGB transfer functions (IEC 61966-2-1). Filtering must happen on linear values, see
// https://github.com/Traverse-Research/ispc-downsampler/issues/25.
// Encoding back to sRGB uses a table built on the Rust side, see `srgb_encode_table()`.

// sRGB 0..1 to linear 0..1. The curve above the linear segment is a degree 6 polynomial fit (max error 1.4e-5), which
// is cheap enough to evaluate for every loaded value and accurate enough that every 8-bit sRGB value survives a
// round trip through the encode table.
inline float srgb_to_linear(float srgb) {
    float curve = -0.0489606597f;
    curve = curve * srgb + 0.207285017f;
    curve = curve * srgb - 0.409055501f;
    curve = curve * srgb + 0.700847626f;
    curve = curve * srgb + 0.516486168f;
    curve = curve * srgb + 0.0324432105f;
    curve = curve * srgb + 0.000945259468f;
    return srgb <= 0.04045f ? srgb * (1.0f / 12.92f) : curve;
}
