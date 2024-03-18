#pragma once
enum PixelFormat {
    Rgba8Unorm = 0,
    Rgb8Unorm,
    Rgba8Snorm,
    Rgb8Snorm,
};

inline uniform bool is_snorm(uniform PixelFormat format) {
    switch (format) {
        case Rgba8Snorm:
        case Rgb8Snorm:
            return true;
        default:
            return false;
    }
}

inline uniform uint32 channel_count(uniform PixelFormat format) {
    switch (format) {
        case Rgba8Unorm:
        case Rgba8Snorm:
            return 4;
        case Rgb8Unorm:
        case Rgb8Snorm:
            return 3;
    }
}

enum NormalMapFormat {
    R8g8b8 = 0,
    R8g8Encoded,
};
