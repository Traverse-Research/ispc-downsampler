#include "weights.ispc"
export void calculate_weight_dimensions(uniform float filter_scale, uniform uint32 src, uniform uint32 target, uniform WeightDimensions out_dimensions[]) {
    uniform float ratio = src / target;

    uniform float filter_radius = ceil(ratio * filter_scale);

    foreach(pixel = 0 ... target) {
        WeightDimensions vars;

        vars.src_center = (pixel + 0.5) * ratio - 0.5;

        float start = ceil(vars.src_center - filter_radius);
        vars.src_start = min(max(start, 0), src - 1);

        float end = floor(vars.src_center + filter_radius);
        end = min(max(end, 0), src - 1);
        vars.src_end = max(end, (float)vars.src_start);

        out_dimensions[pixel] = vars;
    }
}
