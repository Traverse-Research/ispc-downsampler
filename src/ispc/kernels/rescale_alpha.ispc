#include "image.ispc"

// The logic for this kernel only supports 4-channel textures
const uint8 NUM_CHANNELS = 4;

uint64 get_alpha_index(uniform Image image, uint64 x, uint64 y) {
    uint64 pixel_index = x + y * image.size.x;
    return pixel_index * NUM_CHANNELS + 3;
}

float get_alpha(uniform Image image, uint64 x, uint64 y) {
    return (float)image.data[get_alpha_index(image, x, y)] / 255.0f;
}

uniform float calculate_scaled_alpha_coverage(uniform Image image, const uniform float* uniform alpha_cutoff, uniform float scale) {
    // Edge-case for if the texture is downsampled to 1x1
    if (image.size.x == 1 && image.size.y == 1) {
        return 0.0f;
    }
    float coverage = 0.0f;
    const uniform int subsample_factor = 4;

    for (uint64 x = 0; x < image.size.x - 1; x++) {
        for (uint64 y = 0; y < image.size.y - 1; y++) {
            float top_left = min((get_alpha(image, x, y) * scale), 1.0);
            float top_right = min((get_alpha(image, x + 1, y) * scale), 1.0);
            float bottom_left = min((get_alpha(image, x, y + 1) * scale), 1.0);
            float bottom_right = min((get_alpha(image, x + 1, y + 1) * scale), 1.0);

            float texel_coverage = 0.0;
            for (int sy = 0; sy < subsample_factor; sy++) {
                float fy = ((float)sy + 0.5) / (float)subsample_factor;
                for (int sx = 0; sx < subsample_factor; sx++) {
                    float fx = ((float)sx + 0.5) / (float)subsample_factor;
                    float alpha = top_left * (1.0 - fx) * (1.0 - fy)
                        + top_right * fx * (1.0 - fy)
                        + bottom_left * (1.0 - fx) * fy
                        + bottom_right * fx * fy;

                    if (alpha_cutoff) {
                        if (alpha > *alpha_cutoff) {
                            texel_coverage += 1.0;
                        }
                    }
                    else {
                        texel_coverage += alpha;
                    }
                }
            }
            coverage += texel_coverage / (float)(subsample_factor * subsample_factor);
        }
    }

    return reduce_add(coverage) / (float)((image.size.x - 1) * (image.size.y - 1));
}

uniform float calculate_alpha_coverage(uniform Image image, uniform float* uniform alpha_cutoff) {
    return calculate_scaled_alpha_coverage(image, alpha_cutoff, 1.0f);
}

const uniform float INFINITY = 1.0f / 0.0f;

/**
Computes the scaling factor needed for the texture's alpha such that the desired
coverage is best approximated
Ported version of implementation in https://github.com/castano/nvidia-texture-tools/.
**/
uniform float find_alpha_scale_for_coverage(uniform Image image, uniform float desired_coverage, const uniform float* uniform alpha_cutoff)  {
    // This range of potential scale values is an estimate. Especially the upper bound
    // may not be sufficient for some use-cases, depending on the downscale
    // compared to mip 0
    // TO-DO: Figure out if this should be exposed
    uniform float alpha_scale_range_start = 0.0;
    uniform float alpha_scale_range_end = 8.0;

    uniform float alpha_scale = 1.0;

    // The search is done heuristically, so a tested alpha scale value
    // does not directly map to the resulting alpha coverage. To ensure
    // we do not overwrite the best result, store the best result so far
    // and use that instead of the last found scale
    uniform float best_abs_diff = INFINITY;
    uniform float best_alpha_scale;

    // 10-step binary search for the alpha multiplier that best matches
    // the desired alpha coverage
    for (int i = 0; i < 10; i++) {
        uniform float current_coverage = calculate_scaled_alpha_coverage(image, alpha_cutoff, alpha_scale);
        uniform float coverage_diff = current_coverage - desired_coverage;

        if (abs(coverage_diff) < best_abs_diff) {
            best_abs_diff = abs(coverage_diff);
            best_alpha_scale = alpha_scale;
        }

        if (current_coverage < desired_coverage) {
            alpha_scale_range_start = alpha_scale;
        } else if (current_coverage > desired_coverage) {
            alpha_scale_range_end = alpha_scale;
        } else {
            break;
        }
        alpha_scale = (alpha_scale_range_start + alpha_scale_range_end) / 2.0;
    }
    return best_alpha_scale;
}

void apply_alpha_scale(uniform Image image, uniform float scale) {
    for (uint64 y = 0; y < image.size.y; y++) {
        for (uint64 x = 0; x < image.size.x; x++) {
            uint64 pixel_index = x + y * image.size.x;
            uint64 alpha_value_index = pixel_index * NUM_CHANNELS + 3;

            float original_alpha = get_alpha(image, x, y);
            image.data[get_alpha_index(image, x, y)] = (uint8)min(255.0f, (float)original_alpha * scale * 255.0f);
        }
    }
}

export void scale_to_alpha_coverage(uniform uint32 source_width, uniform uint32 source_height, uniform const uint8 source_data[],
    uniform uint32 downsampled_width, uniform uint32 downsampled_height, uniform uint8 downsampled_image_data[],
    const uniform float* uniform alpha_cutoff) {
    uniform Image source = { source_data, {source_width, source_height} };
    uniform Image downsampled = { downsampled_image_data, {downsampled_width, downsampled_height} };

    uniform float coverage = calculate_alpha_coverage(source, alpha_cutoff);
    uniform float scale = find_alpha_scale_for_coverage(downsampled, coverage, alpha_cutoff);
    apply_alpha_scale(downsampled, scale);
}
