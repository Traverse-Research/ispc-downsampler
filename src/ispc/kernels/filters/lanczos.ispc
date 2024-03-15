#ifndef ISPC_LANCZOS
#define ISPC_LANCZOS
#include "../math.ispc"
#include "../weights.ispc"

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

export void calculate_weights_lanczos(uniform float image_scale, uniform float filter_scale, uniform const WeightDimensions * uniform dimensions, uniform float * uniform weights) {
    uniform float start = dimensions->src_start;
    uniform float end = dimensions->src_end;
    uniform float center = dimensions->src_center;

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

#endif
