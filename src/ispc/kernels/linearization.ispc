float srgb_to_linear(uint8 srgb) {
    return pow(((float)srgb / 255.0), 2.2f);
}

uint8 linear_to_srgb(float linear) {
    return (uint8)(pow(linear, 1.0f / 2.2f) * 255.0f);
}

export void linearize(uniform uint width, uniform uint height, uniform uint row_pitch_in_bytes, uniform uint8 num_channels, uniform const uint8 src_data[], uniform uint8 out_data[]) {
    // Assumes we always have 8-bit channels
    const uniform int CHANNEL_SIZE = 1;
    uniform int pixel_stride = num_channels * CHANNEL_SIZE;
    foreach(halfX = 0 ... width, halfY = 0 ... height) {
        int x = halfX;
        int y = halfY;
        int linear_index = y * row_pitch_in_bytes + x * num_channels;

        for (int channel = 0; channel < num_channels; channel++) {
            int byte_offset = (linear_index + channel) * CHANNEL_SIZE;
            int s_indices[1] = { byte_offset };
            // int s_indices[4] = { byte_offset, byte_offset + pixel_stride, byte_offset + row_pitch_in_bytes, byte_offset + pixel_stride + row_pitch_in_bytes };

            float sample = 0.0f;
            for (int i = 0; i < 1; i++) {
                sample += srgb_to_linear(src_data[s_indices[i]]);
            }

            out_data[s_indices[0]] = (uint8)(sample * 255.0f);
            // sample /= 4.0f;
            // float srgb_sample = linear_to_srgb(sample);
            // for (int i = 0; i < 1; i++) {
            //     out_data[s_indices[i]] = srgb_sample;

            // }
        }
    }
}
