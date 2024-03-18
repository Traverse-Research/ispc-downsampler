#pragma once
struct WeightDimensions {
    float src_center;
    float src_start;
    float src_end;
};

struct WeightCollection {
    uniform const uint32* starts;
    uniform const uint32* weight_counts;
    uniform const float* const* values;
};

struct SampleWeights {
    uniform const WeightCollection* vertical_weights;
    uniform const WeightCollection* horizontal_weights;
};
