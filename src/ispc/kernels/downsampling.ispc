#include "weights.ispc"
#include "formats.ispc"
#include "image.ispc"
#include "math.ispc"
#include "linearization.ispc"

// Separable resampling in two passes over blocks of output rows:
//
// 1. Vertical: every output row has a single set of weights, so it is a weighted sum of whole source rows. Lanes
//    run along the row, so every tap is a contiguous vector load with a uniform weight.
// 2. Horizontal: the vertical pass stores its result transposed (per source column, per plane, the rows of the
//    block are contiguous). Every output column then again has a single set of weights, and lanes run over the
//    rows of the block, so every tap is a contiguous vector load too.
//
// The only non-contiguous accesses are one store per intermediate value and one per output value, instead of a
// gather per filter tap. Blocks keep the intermediate small enough to stay in cache.
//
// The intermediate is unclamped f32: Lanczos has negative lobes, which may only cancel out after both passes.
// Unorm values stay in 0..255 and are scaled once on output; sRGB values are decoded to linear 0..1 on load.

struct SourceImage {
    uniform uint32 width;
    uniform uint32 height;
    uniform const uint8* data;
    uniform uint pixel_stride;
    // Bytes from one row to the next.
    uniform uint row_pitch;
};

struct DownsampledImage {
    uniform uint32 width;
    uniform uint32 height;
    uniform uint8* data;
    uniform uint pixel_stride;
    // Bytes from one row to the next.
    uniform uint row_pitch;
};

struct DownsamplingContext {
    uniform const SampleWeights weights;
    // 65536 entries: linear value quantized to 16 bits to 8-bit sRGB. Only needed for sRGB formats.
    uniform const uint8* srgb_encode;
    // Optional: sRGB outputs also store their linear value here, as 16-bit RGBA in rows `linear_row_pitch` bytes
    // apart, for the next level of a mip chain to read instead of the rounded sRGB bytes.
    uniform uint16* linear;
    uniform uint32 linear_row_pitch;
};

// How the vertical pass reads source pixels into planes of the intermediate.
enum SourceKind {
    // Tightly packed channels, filtered independently: every byte is its own plane.
    SOURCE_FLAT_U8,
    // Same, but rgb is decoded from sRGB to linear (alpha, the 4th channel, is linear already).
    SOURCE_FLAT_SRGB8,
    // Packed RGBA, turned into 8 planes: premultiplied rgb, alpha, and unweighted rgba (as a fallback for when
    // nothing under the filter is visible).
    SOURCE_ALPHA_WEIGHTED_U8,
    SOURCE_ALPHA_WEIGHTED_SRGB8,
    // Packed RG normals, decoded to xyz planes since reconstructing z is not linear.
    SOURCE_NORMAL_RG8,
    // Linear RGBA as 16-bit unorm, the precision a mip chain keeps between sRGB levels; like the sRGB kinds otherwise.
    SOURCE_FLAT_LINEAR16,
    SOURCE_ALPHA_WEIGHTED_LINEAR16,
};

// How the horizontal pass turns planes into output pixels.
enum OutputKind {
    OUTPUT_ALBEDO,
    OUTPUT_ALBEDO_ALPHA_WEIGHTED,
    OUTPUT_NORMAL_RGB8,
    OUTPUT_NORMAL_RG8,
};

// Rows per block; the horizontal pass spreads them over its lanes.
inline uniform int block_rows() {
    return 2 * programCount;
}

// 1 / sqrt(v) to ~23 bits: the hardware estimate plus one Newton step. The estimate alone (~12 bits) is off by up
// to 0.05 of an 8-bit step, which rounds differently in up to 15% of texels and compounds down a mip chain.
inline float rsqrt_precise(float v) {
    float r = rsqrt_fast(v);
    return r * (1.5f - 0.5f * v * r * r);
}

// z of a unit normal from x and y, at least 0.1. A plain `sqrt`: as exact, and faster here than `t * rsqrt_precise(t)`.
inline float reconstruct_z(float x, float y) {
    return sqrt(max(0.01f, 1.0f - (x * x + y * y)));
}

inline float decode_snorm8(float value) {
    // Subtract first, which is exact: `value * (2 / 255) - 1` cancels near 128 and leaves small components with
    // enough error to round texels near a boundary the wrong way.
    return (value - 127.5f) * (1.0f / 127.5f);
}

// Stores v0..v3, the values of block rows b .. b + 3, for the element `e` of every active lane. They are contiguous
// per element, so transpose in registers and store 16 bytes per element rather than scattering every value.
inline void store_4_rows(uniform float* uniform block, int e, uniform int b, float v0, float v1, float v2, float v3) {
    uniform const int B = block_rows();
    uniform float transposed[4 * programCount];
    soa_to_aos4(v0, v1, v2, v3, transposed);
    foreach_active (l) {
        uniform int element = extract(e, l);
        *((uniform float<4>* uniform)(block + (uint64)element * B + b)) = *((uniform float<4>* uniform)(transposed + 4 * l));
    }
}

// Fills `union_weights` for output rows y0 + b .. y0 + b + 3 (repeating the last row past `rows`) over the union of
// their source rows, as [source row][4 output rows] with zeros where an output row does not use a source row.
// Returns the first source row; `union_rows` receives the count.
inline uniform uint32 union_weights_4_rows(uniform const WeightCollection* uniform weights, uniform int y0, uniform int b,
    uniform int rows, uniform float* uniform union_weights, uniform uint32& union_rows) {
    uniform uint32 first = 0xFFFFFFFF, last = 0;
    for (uniform int k = 0; k < 4; k++) {
        uniform int y = y0 + min(b + k, rows - 1);
        first = min(first, weights->starts[y]);
        last = max(last, weights->starts[y] + weights->weight_counts[y]);
    }
    union_rows = last - first;
    foreach (i = 0 ... union_rows * 4) {
        union_weights[i] = 0.0f;
    }
    for (uniform int k = 0; k < 4; k++) {
        uniform int y = y0 + min(b + k, rows - 1);
        uniform uint32 offset = weights->starts[y] - first;
        for (uniform uint32 i = 0; i < weights->weight_counts[y]; i++) {
            union_weights[(offset + i) * 4 + k] = weights->values[y][i];
        }
    }
    return first;
}

// Adds `w[k] * x` to `sum_k` for the 4 output rows, skipping the zero weights of rows that do not use this source
// row (common for short filters such as boxes). The branches are uniform and repeat for every element.
#define ACCUMULATE_4(w, x, sum0, sum1, sum2, sum3) \
    sum0 += w[0] * (x);        \
    sum1 += w[1] * (x);        \
    sum2 += w[2] * (x);        \
    sum3 += w[3] * (x);

// Channel of element `j + step` given the channel of element `j`. A per-lane `%` by a runtime value would be a
// scalar division for every lane, so the loops advance the channel instead.
inline int advance_channel(int channel, uniform int step, uniform int channels) {
    int next = channel + step % channels;
    return next >= channels ? next - channels : next;
}

// Loads element j of a source row. sRGB rgb is decoded to linear once per load, which is cheaper than a table.
inline float load_flat(uniform SourceKind kind, const uniform SourceImage* uniform src, uniform uint32 row, uniform uint32 elements,
    int j, bool is_alpha) {
    if (kind == SOURCE_FLAT_LINEAR16 || kind == SOURCE_ALPHA_WEIGHTED_LINEAR16) {
        const uniform uint16* uniform row16 = (const uniform uint16*)(src->data + (uint64)row * src->row_pitch);
        return (float)(int32)row16[j] * (1.0f / 65535.0f);
    }
    float value = (float)(src->data + (uint64)row * src->row_pitch)[j];
    if (kind == SOURCE_FLAT_U8 || kind == SOURCE_ALPHA_WEIGHTED_U8)
        return value;
    value *= 1.0f / 255.0f;
    return is_alpha ? value : srgb_to_linear(value);
}

// Source rows decoded to float planes (the block layout: `planes` floats per pixel), for source kinds whose
// decoding costs more than a load. Groups of 4 output rows share most of their source rows, so decoding on every
// load would repeat the work about 2.5 times; this decodes every row once. Rows go to slot `row % slots`, and there
// are at least as many slots as the largest union of source rows of a group, so a group never evicts its own rows.
struct RowCache {
    uniform float* uniform rows;
    uniform int32* uniform slot_row;
    // Slot of every row of the current union.
    uniform uint32* uniform union_slots;
    uniform uint32 slots;
    uniform uint32 elements;
    // Floats between rows: `elements` plus a cache line. Rows are often a power of two in size, which would put all
    // of them in the same cache sets and make the rows of a group evict each other.
    uniform uint32 row_stride;
};

// Decodes source row `y` into `out`, `planes` floats per pixel.
inline void decode_row(uniform SourceKind kind, const uniform SourceImage* uniform src, uniform uint32 y, uniform int channels,
    uniform float* uniform out) {
    const uniform uint8* uniform row = src->data + (uint64)y * src->row_pitch;
    if (kind == SOURCE_FLAT_LINEAR16 || kind == SOURCE_ALPHA_WEIGHTED_LINEAR16) {
        const uniform uint16* uniform row16 = (const uniform uint16*)row;
        uniform uint32 elements = src->width * channels;
        uniform uint32 j0 = 0;
        for (; j0 + programCount <= elements; j0 += programCount) {
            out[j0 + programIndex] = (float)(int32)row16[j0 + programIndex] * (1.0f / 65535.0f);
        }
        if (j0 + programIndex < elements) {
            out[j0 + programIndex] = (float)(int32)row16[j0 + programIndex] * (1.0f / 65535.0f);
        }
    } else if (kind == SOURCE_FLAT_SRGB8 || kind == SOURCE_ALPHA_WEIGHTED_SRGB8) {
        // Linear rgba; alpha weighting premultiplies when summing, which is cheap compared to decoding.
        uniform uint32 elements = src->width * channels;
        int channel = programIndex % channels;
        // Full vectors unmasked, then the tail.
        uniform uint32 j0 = 0;
        for (; j0 + programCount <= elements; j0 += programCount) {
            float value = row[j0 + programIndex] * (1.0f / 255.0f);
            out[j0 + programIndex] = channel >= 3 ? value : srgb_to_linear(value);
            channel = advance_channel(channel, programCount, channels);
        }
        if (j0 + programIndex < elements) {
            float value = row[j0 + programIndex] * (1.0f / 255.0f);
            out[j0 + programIndex] = channel >= 3 ? value : srgb_to_linear(value);
        }
    } else if (kind == SOURCE_ALPHA_WEIGHTED_U8 || kind == SOURCE_ALPHA_WEIGHTED_SRGB8) {
        // Lanes over rgba elements, 4 per pixel. Every pixel becomes 8 planes: premultiplied rgb and alpha, then
        // the unweighted rgba. Both are 4 contiguous floats per pixel, so store them as 16 byte chunks.
        uniform uint32 elements = src->width * 4;
        int channel = programIndex & 3;
        int alpha_lane = programIndex | 3;
        for (uniform uint32 j0 = 0; j0 < elements; j0 += programCount) {
            int j = j0 + programIndex;
            // Unorm stays in 0..255 like the other unorm paths; sRGB is decoded to linear 0..1.
            float value = 0.0f;
            if (j < elements)
                value = row[j];
            if (kind == SOURCE_ALPHA_WEIGHTED_SRGB8) {
                value *= 1.0f / 255.0f;
                if (channel != 3)
                    value = srgb_to_linear(value);
            }
            float alpha = shuffle(value, alpha_lane);
            uniform float premultiplied[programCount], unweighted[programCount];
            premultiplied[programIndex] = channel == 3 ? value : value * alpha;
            unweighted[programIndex] = value;
            uniform uint32 pixels = min((uniform uint32)programCount, elements - j0) / 4;
            for (uniform uint32 i = 0; i < pixels; i++) {
                uniform float* uniform pixel_out = out + (uint64)(j0 / 4 + i) * 8;
                *((uniform float<4>* uniform)pixel_out) = *((uniform float<4>* uniform)(premultiplied + 4 * i));
                *((uniform float<4>* uniform)(pixel_out + 4)) = *((uniform float<4>* uniform)(unweighted + 4 * i));
            }
        }
    } else if (kind == SOURCE_NORMAL_RG8) {
        const uniform uint16* uniform row16 = (const uniform uint16*)row;
        // xyz interleaved per pixel: transpose in registers for full vector stores, except in the tail.
        uniform uint32 x0 = 0;
        for (; x0 + programCount <= src->width; x0 += programCount) {
            uint16 p = row16[x0 + programIndex];
            float nx = decode_snorm8((float)(p & 0xFF));
            float ny = decode_snorm8((float)(p >> 8));
            float nz = reconstruct_z(nx, ny);
            soa_to_aos3(nx, ny, nz, out + 3 * x0);
        }
        foreach (x = x0 ... src->width) {
            uint16 p = row16[x];
            float nx = decode_snorm8((float)(p & 0xFF));
            float ny = decode_snorm8((float)(p >> 8));
            out[3 * x + 0] = nx;
            out[3 * x + 1] = ny;
            out[3 * x + 2] = reconstruct_z(nx, ny);
        }
    }
}

// Vertical pass: filters source rows into `block`, laid out as [source column * planes + plane][block row].
//
// Four output rows at a time, over the union of their source rows: every source vector is loaded and decoded once
// and feeds the sums of all 4 rows, which are then stored together.
inline void vertical_pass(uniform const SourceImage* uniform src, uniform const WeightCollection* uniform weights,
    uniform int y0, uniform int rows, uniform float* uniform block, uniform float* uniform union_weights,
    uniform SourceKind kind, uniform int channels, uniform RowCache* uniform cache) {
    for (uniform int b = 0; b < rows; b += 4) {
        uniform uint32 union_rows;
        uniform uint32 first = union_weights_4_rows(weights, y0, b, rows, union_weights, union_rows);

        if (cache != NULL) {
            // Decoded rows: a plain weighted sum of floats, two vectors per iteration for 8 independent sums.
            for (uniform uint32 r = 0; r < union_rows; r++) {
                uniform uint32 y = first + r;
                uniform uint32 slot = y % cache->slots;
                uniform float* uniform decoded = cache->rows + (uint64)slot * cache->row_stride;
                if (cache->slot_row[slot] != (int32)y) {
                    decode_row(kind, src, y, channels, decoded);
                    cache->slot_row[slot] = y;
                }
                cache->union_slots[r] = slot;
            }
            uniform uint32 elements = cache->elements;
            if (kind == SOURCE_ALPHA_WEIGHTED_SRGB8 || kind == SOURCE_ALPHA_WEIGHTED_LINEAR16) {
                // Cached rows hold linear rgba; element j = 4 * x + c goes to planes 8 * x + c (premultiplied)
                // and 8 * x + 4 + c (unweighted).
                int channel = programIndex & 3;
                int alpha_lane = programIndex | 3;
                for (uniform uint32 j0 = 0; j0 < elements; j0 += programCount) {
                    int j = j0 + programIndex;
                    if (j < elements) {
                        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f, u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
                        for (uniform uint32 r = 0; r < union_rows; r++) {
                            float x = (cache->rows + (uint64)cache->union_slots[r] * cache->row_stride)[j];
                            float alpha = shuffle(x, alpha_lane);
                            float premultiplied = channel == 3 ? x : x * alpha;
                            const uniform float* uniform w = union_weights + r * 4;
                            ACCUMULATE_4(w, premultiplied, p0, p1, p2, p3)
                            ACCUMULATE_4(w, x, u0, u1, u2, u3)
                        }
                        int e = 2 * j - channel;
                        store_4_rows(block, e, b, p0, p1, p2, p3);
                        store_4_rows(block, e + 4, b, u0, u1, u2, u3);
                    }
                }
                continue;
            }
            uniform uint32 j0 = 0;
            for (; j0 + 2 * programCount <= elements; j0 += 2 * programCount) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
                for (uniform uint32 r = 0; r < union_rows; r++) {
                    const uniform float* uniform row = cache->rows + (uint64)cache->union_slots[r] * cache->row_stride + j0;
                    float x = row[programIndex];
                    float z = row[programCount + programIndex];
                    const uniform float* uniform w = union_weights + r * 4;
                    ACCUMULATE_4(w, x, a0, a1, a2, a3)
                    ACCUMULATE_4(w, z, c0, c1, c2, c3)
                }
                store_4_rows(block, j0 + programIndex, b, a0, a1, a2, a3);
                store_4_rows(block, j0 + programCount + programIndex, b, c0, c1, c2, c3);
            }
            for (; j0 < elements; j0 += programCount) {
                int j = j0 + programIndex;
                if (j < elements) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    for (uniform uint32 r = 0; r < union_rows; r++) {
                        float x = (cache->rows + (uint64)cache->union_slots[r] * cache->row_stride)[j];
                        const uniform float* uniform w = union_weights + r * 4;
                        ACCUMULATE_4(w, x, a0, a1, a2, a3)
                    }
                    store_4_rows(block, j, b, a0, a1, a2, a3);
                }
            }
            continue;
        }

        if (kind == SOURCE_FLAT_U8 || kind == SOURCE_FLAT_SRGB8 || kind == SOURCE_FLAT_LINEAR16) {
            // Every element is its own plane. Two vectors per iteration for 8 independent sums.
            uniform uint32 elements = src->width * channels;
            uniform uint32 j0 = 0;
            int channel = programIndex % channels;
            for (; j0 + 2 * programCount <= elements; j0 += 2 * programCount) {
                int j = j0 + programIndex;
                bool alpha_x = channel >= 3, alpha_z = advance_channel(channel, programCount, channels) >= 3;
                channel = advance_channel(channel, 2 * programCount, channels);
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
                for (uniform uint32 r = 0; r < union_rows; r++) {
                    float x = load_flat(kind, src, first + r, elements, j, alpha_x);
                    float z = load_flat(kind, src, first + r, elements, j + programCount, alpha_z);
                    const uniform float* uniform w = union_weights + r * 4;
                    ACCUMULATE_4(w, x, a0, a1, a2, a3)
                    ACCUMULATE_4(w, z, c0, c1, c2, c3)
                }
                store_4_rows(block, j, b, a0, a1, a2, a3);
                store_4_rows(block, j + programCount, b, c0, c1, c2, c3);
            }
            for (; j0 < elements; j0 += programCount) {
                int j = j0 + programIndex;
                bool alpha_x = channel >= 3;
                channel = advance_channel(channel, programCount, channels);
                if (j < elements) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    for (uniform uint32 r = 0; r < union_rows; r++) {
                        float x = load_flat(kind, src, first + r, elements, j, alpha_x);
                        const uniform float* uniform w = union_weights + r * 4;
                        ACCUMULATE_4(w, x, a0, a1, a2, a3)
                    }
                    store_4_rows(block, j, b, a0, a1, a2, a3);
                }
            }
        } else if (kind == SOURCE_ALPHA_WEIGHTED_U8 || kind == SOURCE_ALPHA_WEIGHTED_SRGB8
            || kind == SOURCE_ALPHA_WEIGHTED_LINEAR16) {
            // Lanes over RGBA elements, 4 per pixel. Every pixel gets 8 planes: premultiplied rgb and alpha, then
            // the unweighted rgba (for when nothing under the filter is visible; its alpha plane is unused).
            // Element j = 4 * x + c goes to planes 8 * x + c and 8 * x + 4 + c.
            uniform uint32 elements = src->width * 4;
            int channel = programIndex & 3;
            int alpha_lane = programIndex | 3;
            for (uniform uint32 j0 = 0; j0 < elements; j0 += programCount) {
                int j = j0 + programIndex;
                if (j < elements) {
                    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f, u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
                    for (uniform uint32 r = 0; r < union_rows; r++) {
                        float x = load_flat(kind, src, first + r, elements, j, channel == 3);
                        float alpha = shuffle(x, alpha_lane);
                        float premultiplied = channel == 3 ? x : x * alpha;
                        const uniform float* uniform w = union_weights + r * 4;
                        ACCUMULATE_4(w, premultiplied, p0, p1, p2, p3)
                        ACCUMULATE_4(w, x, u0, u1, u2, u3)
                    }
                    int e = 2 * j - channel;
                    store_4_rows(block, e, b, p0, p1, p2, p3);
                    store_4_rows(block, e + 4, b, u0, u1, u2, u3);
                }
            }
        } else if (kind == SOURCE_NORMAL_RG8) {
            // Lanes over pixels, decoded to xyz once per loaded texel since reconstructing z is not linear.
            foreach (x = 0 ... src->width) {
                float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
                float y0_ = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
                float z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;
                for (uniform uint32 r = 0; r < union_rows; r++) {
                    const uniform uint16* uniform row16 = (const uniform uint16*)(src->data + (uint64)(first + r) * src->row_pitch);
                    uint16 p = row16[x];
                    float nx = decode_snorm8((float)(p & 0xFF));
                    float ny = decode_snorm8((float)(p >> 8));
                    float nz = reconstruct_z(nx, ny);
                    const uniform float* uniform w = union_weights + r * 4;
                    ACCUMULATE_4(w, nx, x0, x1, x2, x3)
                    ACCUMULATE_4(w, ny, y0_, y1, y2, y3)
                    ACCUMULATE_4(w, nz, z0, z1, z2, z3)
                }
                store_4_rows(block, 3 * x + 0, b, x0, x1, x2, x3);
                store_4_rows(block, 3 * x + 1, b, y0_, y1, y2, y3);
                store_4_rows(block, 3 * x + 2, b, z0, z1, z2, z3);
            }
        }
    }
}

// Signed conversion: the value is 0..255, and float to unsigned takes several instructions without AVX-512.
inline uint32 to_unorm8(float value) {
    return (int32)(clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// Linear 0..1 to 8-bit sRGB through `srgb_encode`, indexed by the value quantized to 16 bits. That is exact on
// that grid (so every sRGB value survives a round trip) and far cheaper than pow().
inline uint32 to_srgb8(float value, const uniform uint8* uniform srgb_encode) {
    return srgb_encode[(int)(clamp(value, 0.0f, 1.0f) * 65535.0f + 0.5f)];
}

inline uint32 pack_rgb(float r, float g, float b, const uniform uint8* uniform srgb_encode) {
    // The filtered colour can over- or undershoot 0..1 because of the negative lobes, so clamp only here.
    if (srgb_encode != NULL)
        return to_srgb8(r, srgb_encode) | (to_srgb8(g, srgb_encode) << 8) | (to_srgb8(b, srgb_encode) << 16);
    return to_unorm8(r) | (to_unorm8(g) << 8) | (to_unorm8(b) << 16);
}

// A unit length component to its rounded unorm byte, `x * 127.5 + 127.5` rounded. Scaling first and adding 128 once
// rounds only at the add: `x * 0.5 + 0.5` then `* 255` rounds twice, and flips texels that are within a float step
// of a rounding boundary (flat areas of normal maps are full of them).
inline uint32 normal_to_unorm8(float scaled) {
    return (int32)min(scaled + 128.0f, 255.0f);
}

inline uint32 pack_normal(float x, float y, float z) {
    float scale = rsqrt_precise(x * x + y * y + z * z) * 127.5f;
    return normal_to_unorm8(x * scale) | (normal_to_unorm8(y * scale) << 8) | (normal_to_unorm8(z * scale) << 16);
}

// Stores the low `size` bytes of `packed` (4 bytes per channel slot, the rest padding) in one go where possible.
// With 3 byte pixels a 4 byte store spills into the next pixel, which is only written later in the same row, so
// this is fine except in the last column.
inline void store_pixel(uniform uint8* varying pixel, uint32 packed, uniform int size, uniform uint32 pixel_stride, uniform bool last_column) {
    if (pixel_stride >= 4 || (size == 3 && !last_column)) {
        *((uniform uint32* varying)pixel) = packed;
    } else if (size == 2) {
        *((uniform uint16* varying)pixel) = (uint16)packed;
    } else {
        pixel[0] = (uint8)packed;
        pixel[1] = (uint8)(packed >> 8);
        pixel[2] = (uint8)(packed >> 16);
    }
}

// Linear 0..1 quantized to 16 bits, the grid `to_srgb8()` encodes from.
inline uint32 to_unorm16(float value) {
    return (int32)(clamp(value, 0.0f, 1.0f) * 65535.0f + 0.5f);
}

// Stores linear rgb and the 8-bit alpha as 16-bit RGBA, for the next level of a mip chain.
inline void store_linear(uniform uint16* varying linear, float r, float g, float b, uint32 alpha8) {
    uint64 value = (uint64)to_unorm16(r) | ((uint64)to_unorm16(g) << 16) | ((uint64)to_unorm16(b) << 32)
        | ((uint64)(alpha8 * 257) << 48);
    *((uniform uint64* varying)linear) = value;
}

// Turns the filtered planes of one output pixel into its stored format. With `linear` set, albedo also keeps its
// linear value there (16-bit RGBA, rows `linear_pitch` bytes apart).
inline void write_output(uniform DownsampledImage* uniform dst, int b, uniform int y0, uniform uint32 xo, uniform uint64 row_pitch,
    float s0, float s1, float s2, float s3, float s4, float s5, float s6,
    uniform int output_channels, uniform OutputKind output, const uniform uint8* uniform srgb_encode, uniform float scale,
    uniform uint8* uniform linear, uniform uint64 linear_pitch) {
    uniform uint8* varying pixel = dst->data + (uint64)(y0 + b) * row_pitch + (uint64)xo * dst->pixel_stride;
    uniform uint16* varying linear_pixel = (uniform uint16* varying)(linear + (uint64)(y0 + b) * linear_pitch + (uint64)xo * 8);
    uniform bool last_column = xo == dst->width - 1;
    if (output == OUTPUT_ALBEDO) {
        float r = s0 * scale, g = s1 * scale, bl = s2 * scale;
        // A 4th byte is either alpha or padding, which albedo leaves at 0.
        uint32 alpha8 = output_channels == 4 ? to_unorm8(s3 * scale) : 0;
        store_pixel(pixel, pack_rgb(r, g, bl, srgb_encode) | (alpha8 << 24), output_channels, dst->pixel_stride, last_column);
        if (linear != NULL)
            store_linear(linear_pixel, r, g, bl, alpha8);
    } else if (output == OUTPUT_ALBEDO_ALPHA_WEIGHTED) {
        // s0..2 are alpha weighted sums in source units squared, s3 the alpha sum. If nothing under the
        // filter is visible there is nothing to weight by, so fall back to the unweighted sums.
        float alpha = s3 * scale;
        float r, g, bl;
        if (alpha > 0.5f / 255.0f) {
            float to_colour = scale / s3;
            r = s0 * to_colour, g = s1 * to_colour, bl = s2 * to_colour;
        } else {
            r = s4 * scale, g = s5 * scale, bl = s6 * scale;
        }
        uint32 alpha8 = to_unorm8(alpha);
        store_pixel(pixel, pack_rgb(r, g, bl, srgb_encode) | (alpha8 << 24), 4, dst->pixel_stride, last_column);
        if (linear != NULL)
            store_linear(linear_pixel, r, g, bl, alpha8);
    } else if (output == OUTPUT_NORMAL_RGB8) {
        // The planes hold raw bytes; decoding is linear so it commutes with the (normalized) filter.
        // Padding after the 3 channels is documented to be 255.
        uint32 packed = pack_normal(decode_snorm8(s0), decode_snorm8(s1), decode_snorm8(s2)) | 0xFF000000;
        store_pixel(pixel, packed, 3, dst->pixel_stride, last_column);
    } else if (output == OUTPUT_NORMAL_RG8) {
        uint32 packed = pack_normal(s0, s1, s2) & 0xFFFF;
        store_pixel(pixel, packed | 0xFFFF0000, 2, dst->pixel_stride, last_column);
    }
}

// Horizontal pass: filters `block` along the source columns and writes `rows` output rows starting at `y0`.
// `planes` is the number of planes per source column; `scale` maps source units to 0..1.
inline void horizontal_pass(uniform DownsampledImage* uniform dst, uniform const WeightCollection* uniform weights,
    uniform int y0, uniform int rows, uniform const float* uniform block, uniform int planes, uniform int output_channels,
    uniform OutputKind output, const uniform uint8* uniform srgb_encode, uniform float scale,
    uniform uint8* uniform linear, uniform uint64 linear_pitch) {
    uniform const int B = block_rows();
    uniform const uint64 row_pitch = dst->row_pitch;
    for (uniform uint32 xo = 0; xo < dst->width; xo++) {
        uniform uint32 start = weights->starts[xo];
        uniform uint32 count = weights->weight_counts[xo];
        const uniform float* uniform w = weights->values[xo];

        // Both halves of the block at once, for twice the independent sums per tap. Rows past `rows` are read
        // (the block is always allocated in full) but never stored.
        uniform int b_half = programCount;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f;
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f, t4 = 0.0f, t5 = 0.0f, t6 = 0.0f;
        const uniform float* uniform column = block + (uint64)start * planes * B;
        for (uniform uint32 i = 0; i < count; i++) {
            uniform float wi = w[i];
            s0 += wi * column[0 * B + programIndex];
            t0 += wi * column[0 * B + b_half + programIndex];
            if (planes > 1) {
                s1 += wi * column[1 * B + programIndex];
                t1 += wi * column[1 * B + b_half + programIndex];
            }
            if (planes > 2) {
                s2 += wi * column[2 * B + programIndex];
                t2 += wi * column[2 * B + b_half + programIndex];
            }
            if (planes > 3) {
                s3 += wi * column[3 * B + programIndex];
                t3 += wi * column[3 * B + b_half + programIndex];
            }
            if (planes > 4) {
                s4 += wi * column[4 * B + programIndex];
                t4 += wi * column[4 * B + b_half + programIndex];
                s5 += wi * column[5 * B + programIndex];
                t5 += wi * column[5 * B + b_half + programIndex];
                s6 += wi * column[6 * B + programIndex];
                t6 += wi * column[6 * B + b_half + programIndex];
            }
            column += planes * B;
        }
        // Full blocks (all but the last) need no mask, which keeps the stores simple.
        if (rows == B) {
            write_output(dst, programIndex, y0, xo, row_pitch, s0, s1, s2, s3, s4, s5, s6, output_channels, output, srgb_encode, scale, linear, linear_pitch);
            write_output(dst, programIndex + b_half, y0, xo, row_pitch, t0, t1, t2, t3, t4, t5, t6, output_channels, output, srgb_encode, scale, linear, linear_pitch);
        } else {
            if (programIndex < rows)
                write_output(dst, programIndex, y0, xo, row_pitch, s0, s1, s2, s3, s4, s5, s6, output_channels, output, srgb_encode, scale, linear, linear_pitch);
            if (programIndex + b_half < rows)
                write_output(dst, programIndex + b_half, y0, xo, row_pitch, t0, t1, t2, t3, t4, t5, t6, output_channels, output, srgb_encode, scale, linear, linear_pitch);
        }
    }
}

inline void resample(uniform const SourceImage* uniform src, uniform DownsampledImage* uniform dst,
    uniform const DownsamplingContext* uniform ctx, uniform SourceKind kind, uniform int channels, uniform int planes,
    uniform int output_channels, uniform OutputKind output, uniform bool srgb, uniform float scale) {
    uniform const int B = block_rows();
    uniform float* uniform block = uniform new uniform float[(uint64)src->width * planes * B];
    // The union of the source rows of 4 output rows is at most the whole source.
    uniform float* uniform union_weights = uniform new uniform float[(uint64)src->height * 4];

    // Decode sRGB rows once, unless the filter is so wide (e.g. down to a few texels) that keeping its source rows
    // around would take too much memory; then decode on every load instead. Cheaper decoding (unorm alpha
    // weighting, normals) is faster to redo on load than to read back as floats.
    uniform RowCache cache_storage;
    uniform RowCache* uniform cache = NULL;
    if (kind == SOURCE_FLAT_SRGB8 || kind == SOURCE_ALPHA_WEIGHTED_SRGB8) {
        uniform uint32 slots = 0;
        for (uniform int y0 = 0; y0 < dst->height; y0 += B) {
            uniform int rows = min(B, (uniform int)dst->height - y0);
            for (uniform int b = 0; b < rows; b += 4) {
                uniform uint32 first = 0xFFFFFFFF, last = 0;
                for (uniform int k = 0; k < 4; k++) {
                    uniform int y = y0 + min(b + k, rows - 1);
                    first = min(first, ctx->weights.vertical_weights->starts[y]);
                    last = max(last, ctx->weights.vertical_weights->starts[y] + ctx->weights.vertical_weights->weight_counts[y]);
                }
                slots = max(slots, last - first);
            }
        }
        uniform uint32 elements = src->width * channels;
        uniform uint32 row_stride = elements + 16;
        if ((uint64)slots * row_stride * 4 <= 32 * 1024 * 1024) {
            cache_storage.slots = slots;
            cache_storage.elements = elements;
            cache_storage.row_stride = row_stride;
            cache_storage.rows = uniform new uniform float[(uint64)slots * row_stride];
            cache_storage.slot_row = uniform new uniform int32[slots];
            cache_storage.union_slots = uniform new uniform uint32[slots];
            for (uniform uint32 i = 0; i < slots; i++)
                cache_storage.slot_row[i] = -1;
            cache = &cache_storage;
        }
    }

    for (uniform int y0 = 0; y0 < dst->height; y0 += B) {
        uniform int rows = min(B, (uniform int)dst->height - y0);
        vertical_pass(src, ctx->weights.vertical_weights, y0, rows, block, union_weights, kind, channels, cache);
        horizontal_pass(dst, ctx->weights.horizontal_weights, y0, rows, block, planes, output_channels, output,
            srgb ? ctx->srgb_encode : NULL, scale, srgb ? (uniform uint8* uniform)ctx->linear : NULL, ctx->linear_row_pitch);
    }
    delete[] block;
    delete[] union_weights;
    if (cache != NULL) {
        delete[] cache->rows;
        delete[] cache->slot_row;
        delete[] cache->union_slots;
    }
}

// Rows must be tightly packed, with `pixel_stride` (at most 8) bytes per pixel. Bytes past the format's channels are
// padding: filtered along but not written.
export void resample_with_cached_weights_3(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx) {
    uniform int stride = src->pixel_stride;
    if (is_srgb(pixel_format))
        resample(src, dst, ctx, SOURCE_FLAT_SRGB8, stride, stride, 3, OUTPUT_ALBEDO, true, 1.0f);
    else
        resample(src, dst, ctx, SOURCE_FLAT_U8, stride, stride, 3, OUTPUT_ALBEDO, false, 1.0f / 255.0f);
}

export void resample_with_cached_weights_4(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx) {
    uniform int stride = src->pixel_stride;
    if (is_srgb(pixel_format))
        resample(src, dst, ctx, SOURCE_FLAT_SRGB8, stride, stride, 4, OUTPUT_ALBEDO, true, 1.0f);
    else
        resample(src, dst, ctx, SOURCE_FLAT_U8, stride, stride, 4, OUTPUT_ALBEDO, false, 1.0f / 255.0f);
}

/// Like `resample_with_cached_weights_4`, but weights rgb by alpha. The source must have a pixel stride of 4.
export void resample_with_cached_weights_4_alpha_weighted(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform PixelFormat pixel_format, uniform DownsamplingContext * uniform ctx) {
    if (is_srgb(pixel_format))
        resample(src, dst, ctx, SOURCE_ALPHA_WEIGHTED_SRGB8, 4, 8, 4, OUTPUT_ALBEDO_ALPHA_WEIGHTED, true, 1.0f);
    else
        resample(src, dst, ctx, SOURCE_ALPHA_WEIGHTED_U8, 4, 8, 4, OUTPUT_ALBEDO_ALPHA_WEIGHTED, false, 1.0f / 255.0f);
}

/// sRGB RGBA from 16-bit linear RGBA, the next level of a mip chain.
export void resample_linear16(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst,
    uniform DownsamplingContext * uniform ctx, uniform bool alpha_weighted) {
    if (alpha_weighted)
        resample(src, dst, ctx, SOURCE_ALPHA_WEIGHTED_LINEAR16, 4, 8, 4, OUTPUT_ALBEDO_ALPHA_WEIGHTED, true, 1.0f);
    else
        resample(src, dst, ctx, SOURCE_FLAT_LINEAR16, 4, 4, 4, OUTPUT_ALBEDO, true, 1.0f);
}

/// Box filters a normal map with the (normalized) box weights in `ctx`, and renormalizes the result.
/// The source must be tightly packed.
// Exact 2:1 box filter for normal maps, which is what every mip level of a power of two texture is. The generic
// two passes are dominated by their intermediate for a filter this short. Here every output row is computed
// straight from its two source rows, in two loops: one sums the 2x2 texels of every output texel into planar
// x, y, z rows, the other normalizes and stores them. Each is short and independent per iteration; combined, an
// iteration is one long dependency chain the core cannot overlap with the next. The arithmetic matches the generic
// box path (sums scaled by 0.25, which is exact, the same decoding and the same `pack_normal()`), so both give the
// same bytes.
//
// All loops run whole vectors unmasked and only mask the last one: masked loads and stores are much slower.

// 2^23: integers below it are exact in the mantissa of `NORMAL_MAGIC + integer`, which converts ints to floats
// with an OR and a subtract instead of a conversion instruction, which some CPUs (Zen 2) only run on one pipe.
static const uniform float NORMAL_MAGIC = 8388608.0f;

inline float int_to_float(int32 value) {
    // `value` must be within 0 .. 2^23.
    return floatbits(value | intbits(NORMAL_MAGIC)) - NORMAL_MAGIC;
}

// Normalizes and stores the (x, y, z) sums of `programCount` output texels starting at `x0`: `pixel_size` 2 (xy),
// 3 (xyz) or 4 (xyz and 255 padding). `bytes` is scratch for interleaving Rgb8.
inline void store_normals(float x, float y, float z, uniform uint8* uniform out, uniform uint32 x0,
    uniform uint32 count, uniform int pixel_size, uniform int32 bytes[]) {
    uint32 packed = pack_normal(x, y, z);
    if (count == programCount) {
        if (pixel_size == 4) {
            ((uniform uint32* uniform)out)[x0 + programIndex] = packed | 0xFF000000;
        } else if (pixel_size == 2) {
            ((uniform uint16* uniform)out)[x0 + programIndex] = (uint16)packed;
        } else {
            // Interleave in registers, then narrow and store contiguously rather than scattering every byte.
            soa_to_aos3((int32)(packed & 0xFF), (int32)((packed >> 8) & 0xFF), (int32)(packed >> 16), bytes);
            for (uniform int i = 0; i < 3; i++) {
                out[3 * x0 + i * programCount + programIndex] = (uint8)bytes[i * programCount + programIndex];
            }
        }
    } else if (programIndex < count) {
        uniform uint8* varying pixel = out + (uint64)(x0 + programIndex) * pixel_size;
        pixel[0] = (uint8)packed;
        pixel[1] = (uint8)(packed >> 8);
        if (pixel_size >= 3)
            pixel[2] = (uint8)(packed >> 16);
        if (pixel_size == 4)
            pixel[3] = 255;
    }
}

// Rgb8 sums of 2 * programCount source texels starting at element 6 * x0 of the row sums, as normals in the
// units of the generic box path (averages decoded to snorm).
inline void rgb8_pair_sums(const uniform int32 rows[], uniform uint32 x0, float& x, float& y, float& z) {
    int even = 2 * programIndex, odd = 2 * programIndex + 1;
    int32 ar, ag, ab, br, bg, bb;
    aos_to_soa3(rows + 6 * x0, &ar, &ag, &ab);
    aos_to_soa3(rows + 6 * x0 + 3 * programCount, &br, &bg, &bb);
    x = decode_snorm8(int_to_float(shuffle(ar, br, even) + shuffle(ar, br, odd)) * 0.25f);
    y = decode_snorm8(int_to_float(shuffle(ag, bg, even) + shuffle(ag, bg, odd)) * 0.25f);
    z = decode_snorm8(int_to_float(shuffle(ab, bb, even) + shuffle(ab, bb, odd)) * 0.25f);
}

// RGBX: texels in 32-bit lanes, `t0`/`b0` texels 0..programCount-1 of the top/bottom row and `t1`/`b1` the next.
// Rows are summed first, per channel two at a time in 16-bit fields (4 texels never exceed 1020), then adjacent
// texels are paired with an in-lane swap. Only compacting the even lanes crosses 128-bit lanes: cross-lane permutes
// run on few pipes on some CPUs (Zen 2), so they are kept to a minimum.
inline void rgbx8_pair_sums(uint32 t0, uint32 t1, uint32 b0, uint32 b1, float& x, float& y, float& z) {
    int swap = programIndex ^ 1, even = 2 * programIndex;
    // [x z] and [y a] fields of both rows, then of both texels of each pair (valid in even lanes).
    uint32 xz0 = (t0 & 0x00FF00FF) + (b0 & 0x00FF00FF), xz1 = (t1 & 0x00FF00FF) + (b1 & 0x00FF00FF);
    uint32 ya0 = ((t0 >> 8) & 0x00FF00FF) + ((b0 >> 8) & 0x00FF00FF), ya1 = ((t1 >> 8) & 0x00FF00FF) + ((b1 >> 8) & 0x00FF00FF);
    xz0 += shuffle((int32)xz0, swap);
    xz1 += shuffle((int32)xz1, swap);
    ya0 += shuffle((int32)ya0, swap);
    ya1 += shuffle((int32)ya1, swap);
    uint32 xz = shuffle((int32)xz0, (int32)xz1, even), ya = shuffle((int32)ya0, (int32)ya1, even);
    x = decode_snorm8(int_to_float(xz & 0xFFFF) * 0.25f);
    y = decode_snorm8(int_to_float(ya & 0xFFFF) * 0.25f);
    z = decode_snorm8(int_to_float(xz >> 16) * 0.25f);
}

// Rg8: z is reconstructed per texel, which is not linear, so every texel is decoded to floats first. The two
// texels of an output texel in a row are one 32-bit word, `top` and `bottom`. Sums are scaled like the box weights.
inline void rg8_pair_sums(uint32 top, uint32 bottom, float& x, float& y, float& z) {
    float tx0 = decode_snorm8(int_to_float(top & 0xFF)), ty0 = decode_snorm8(int_to_float((top >> 8) & 0xFF));
    float tx1 = decode_snorm8(int_to_float((top >> 16) & 0xFF)), ty1 = decode_snorm8(int_to_float(top >> 24));
    float ux0 = decode_snorm8(int_to_float(bottom & 0xFF)), uy0 = decode_snorm8(int_to_float((bottom >> 8) & 0xFF));
    float ux1 = decode_snorm8(int_to_float((bottom >> 16) & 0xFF)), uy1 = decode_snorm8(int_to_float(bottom >> 24));
    // Like the vertical box pass (both rows weighted 1/2), then the horizontal one (both columns 1/2).
    float cx0 = 0.5f * tx0 + 0.5f * ux0, cx1 = 0.5f * tx1 + 0.5f * ux1;
    float cy0 = 0.5f * ty0 + 0.5f * uy0, cy1 = 0.5f * ty1 + 0.5f * uy1;
    float cz0 = 0.5f * reconstruct_z(tx0, ty0) + 0.5f * reconstruct_z(ux0, uy0);
    float cz1 = 0.5f * reconstruct_z(tx1, ty1) + 0.5f * reconstruct_z(ux1, uy1);
    x = 0.5f * cx0 + 0.5f * cx1;
    y = 0.5f * cy0 + 0.5f * cy1;
    z = 0.5f * cz0 + 0.5f * cz1;
}

// Packed Rgb8 in 32-bit lanes without byte shuffles: every lane takes 12 source bytes (4 texels, 2 output texels)
// as three words, splits them into even and odd bytes in 16-bit fields, and sums both rows and both texels of each
// output texel with a few masks and adds. Writes the normals of the lane's even and odd output texel (the averages
// decoded to snorm, the same arithmetic as the box path; the 0.25 is exact) to `even` and `odd`, x y z planes
// `plane` floats apart.
inline void rgb8_2x2_sums(const uniform uint8* uniform top, const uniform uint8* uniform bottom,
    uniform float even[], uniform float odd[], uniform uint32 plane) {
    // Words w0 = [r0 g0 b0 r1], w1 = [g1 b1 r2 g2], w2 = [b2 r3 g3 b3], lowest byte first.
    int32 t0, t1, t2, u0, u1, u2;
    aos_to_soa3((uniform int32* uniform)top, &t0, &t1, &t2);
    aos_to_soa3((uniform int32* uniform)bottom, &u0, &u1, &u2);
    // Even and odd bytes of both rows, summed per 16-bit field (at most 510).
    uint32 e0 = (t0 & 0x00FF00FF) + (u0 & 0x00FF00FF), o0 = ((t0 >> 8) & 0x00FF00FF) + ((u0 >> 8) & 0x00FF00FF);
    uint32 e1 = (t1 & 0x00FF00FF) + (u1 & 0x00FF00FF), o1 = ((t1 >> 8) & 0x00FF00FF) + ((u1 >> 8) & 0x00FF00FF);
    uint32 e2 = (t2 & 0x00FF00FF) + (u2 & 0x00FF00FF), o2 = ((t2 >> 8) & 0x00FF00FF) + ((u2 >> 8) & 0x00FF00FF);
    // e0 = [r0 b0], o0 = [g0 r1], e1 = [g1 r2], o1 = [b1 g2], e2 = [b2 g3], o2 = [r3 b3].
    even[programIndex] = decode_snorm8(int_to_float((e0 & 0xFFFF) + (o0 >> 16)) * 0.25f);
    even[plane + programIndex] = decode_snorm8(int_to_float((o0 & 0xFFFF) + (e1 & 0xFFFF)) * 0.25f);
    even[2 * plane + programIndex] = decode_snorm8(int_to_float((e0 >> 16) + (o1 & 0xFFFF)) * 0.25f);
    odd[programIndex] = decode_snorm8(int_to_float((e1 >> 16) + (o2 & 0xFFFF)) * 0.25f);
    odd[plane + programIndex] = decode_snorm8(int_to_float((o1 >> 16) + (e2 >> 16)) * 0.25f);
    odd[2 * plane + programIndex] = decode_snorm8(int_to_float((e2 & 0xFFFF) + (o2 >> 16)) * 0.25f);
}

inline void normal_map_2x2(const uniform SourceImage* uniform src, uniform DownsampledImage* uniform dst,
    uniform NormalMapFormat normal_map_format) {
    uniform uint32 width = dst->width;
    uniform uint32 stride = src->pixel_stride;
    uniform int pixel_size = normal_map_format == R8g8b8 ? stride : 2;
    // Planar sums of the current output row.
    uniform float* uniform sums = uniform new uniform float[3 * (width + programCount)];
    uniform float* uniform sx = sums;
    uniform float* uniform sy = sums + width + programCount;
    uniform float* uniform sz = sums + 2 * (width + programCount);
    // Rgb8: per-byte sums of the two source rows, padded for the last deinterleave.
    uniform int32* uniform rows = uniform new uniform int32[src->width * 3 + 6 * programCount];
    foreach (i = src->width * 3 ... src->width * 3 + 6 * programCount) {
        rows[i] = 0;
    }
    uniform int32 bytes[3 * programCount];
    // Rgb8: per-lane normals and packed results of the whole groups of a row.
    uniform float* uniform pair_normals = uniform new uniform float[6 * (width / 2 + programCount)];
    uniform uint32* uniform packed_even = uniform new uniform uint32[width / 2 + programCount];
    uniform uint32* uniform packed_odd = uniform new uniform uint32[width / 2 + programCount];
    // Whole vectors of output texels, and the rest.
    uniform uint32 whole = width / programCount * programCount;

    for (uniform uint32 yo = 0; yo < dst->height; yo++) {
        const uniform uint8* uniform top = src->data + (uint64)(2 * yo) * src->row_pitch;
        const uniform uint8* uniform bottom = top + src->row_pitch;
        uniform uint8* uniform out = dst->data + (uint64)yo * dst->row_pitch;

        if (normal_map_format == R8g8b8 && stride == 3) {
            // Whole groups of 2 * programCount output texels, in three short loops over the row rather than one
            // long chain per group, which the core cannot overlap with the next: sums, normalize and pack, and
            // interleave and store. The rest goes through the row sums below.
            uniform uint32 groups = width / (2 * programCount);
            uniform uint32 lanes = groups * programCount;
            // Planar x y z of the even and odd output texels of every lane.
            uniform float* uniform even = pair_normals;
            uniform float* uniform odd = pair_normals + 3 * lanes;
            for (uniform uint32 g = 0; g < groups; g++) {
                rgb8_2x2_sums(top + g * 12 * programCount, bottom + g * 12 * programCount,
                    even + g * programCount, odd + g * programCount, lanes);
            }
            for (uniform uint32 i = 0; i < lanes; i += programCount) {
                uint32 j = i + programIndex;
                packed_even[j] = pack_normal(even[j], even[lanes + j], even[2 * lanes + j]);
                packed_odd[j] = pack_normal(odd[j], odd[lanes + j], odd[2 * lanes + j]);
            }
            uniform uint16* uniform out16 = (uniform uint16* uniform)out;
            for (uniform uint32 i = 0; i < lanes; i += programCount) {
                // [xa ya za xb yb zb] of every lane as three 16-bit halves, interleaved and stored contiguously.
                uint32 a = packed_even[i + programIndex], b = packed_odd[i + programIndex];
                soa_to_aos3((int32)(a & 0xFFFF), (int32)((a >> 16) | ((b & 0xFF) << 8)), (int32)(b >> 8), bytes);
                for (uniform int k = 0; k < 3; k++) {
                    out16[3 * i + k * programCount + programIndex] = (uint16)bytes[k * programCount + programIndex];
                }
            }
            uniform uint32 done = groups * 2 * programCount;
            if (done == width)
                continue;
            uniform uint32 elements = src->width * 3;
            uniform uint32 j0 = 0;
            for (; j0 + programCount <= elements; j0 += programCount) {
                rows[j0 + programIndex] = (int32)top[j0 + programIndex] + (int32)bottom[j0 + programIndex];
            }
            foreach (j = j0 ... elements) {
                rows[j] = (int32)top[j] + (int32)bottom[j];
            }
            for (uniform uint32 x0 = done; x0 < width; x0 += programCount) {
                float x, y, z;
                rgb8_pair_sums(rows, x0, x, y, z);
                sx[x0 + programIndex] = x;
                sy[x0 + programIndex] = y;
                sz[x0 + programIndex] = z;
            }
            for (uniform uint32 x0 = done; x0 < width; x0 += programCount) {
                store_normals(sx[x0 + programIndex], sy[x0 + programIndex], sz[x0 + programIndex], out, x0,
                    min((uniform uint32)programCount, width - x0), pixel_size, bytes);
            }
            continue;
        } else if (normal_map_format == R8g8b8) {
            const uniform uint32* uniform top32 = (const uniform uint32*)top;
            const uniform uint32* uniform bottom32 = (const uniform uint32*)bottom;
            uniform uint32 x0 = 0;
            for (; x0 < whole; x0 += programCount) {
                uniform uint32 texel = 2 * x0;
                float x, y, z;
                rgbx8_pair_sums(top32[texel + programIndex], top32[texel + programCount + programIndex],
                    bottom32[texel + programIndex], bottom32[texel + programCount + programIndex], x, y, z);
                sx[x0 + programIndex] = x;
                sy[x0 + programIndex] = y;
                sz[x0 + programIndex] = z;
            }
            if (x0 < width) {
                uint32 t0 = 0, t1 = 0, b0 = 0, b1 = 0;
                int first = 2 * x0 + programIndex, second = 2 * x0 + programCount + programIndex;
                if (first < src->width) {
                    t0 = top32[first];
                    b0 = bottom32[first];
                }
                if (second < src->width) {
                    t1 = top32[second];
                    b1 = bottom32[second];
                }
                float x, y, z;
                rgbx8_pair_sums(t0, t1, b0, b1, x, y, z);
                sx[x0 + programIndex] = x;
                sy[x0 + programIndex] = y;
                sz[x0 + programIndex] = z;
            }
        } else {
            // One 32-bit word per output texel and row: its two texels.
            const uniform uint32* uniform top32 = (const uniform uint32*)top;
            const uniform uint32* uniform bottom32 = (const uniform uint32*)bottom;
            uniform uint32 x0 = 0;
            for (; x0 < whole; x0 += programCount) {
                float x, y, z;
                rg8_pair_sums(top32[x0 + programIndex], bottom32[x0 + programIndex], x, y, z);
                sx[x0 + programIndex] = x;
                sy[x0 + programIndex] = y;
                sz[x0 + programIndex] = z;
            }
            if (x0 < width) {
                uint32 t = 0, u = 0;
                if (x0 + programIndex < width) {
                    t = top32[x0 + programIndex];
                    u = bottom32[x0 + programIndex];
                }
                float x, y, z;
                rg8_pair_sums(t, u, x, y, z);
                sx[x0 + programIndex] = x;
                sy[x0 + programIndex] = y;
                sz[x0 + programIndex] = z;
            }
        }

        uniform uint32 x0 = 0;
        for (; x0 < whole; x0 += programCount) {
            store_normals(sx[x0 + programIndex], sy[x0 + programIndex], sz[x0 + programIndex], out, x0,
                programCount, pixel_size, bytes);
        }
        if (x0 < width) {
            store_normals(sx[x0 + programIndex], sy[x0 + programIndex], sz[x0 + programIndex], out, x0,
                width - x0, pixel_size, bytes);
        }
    }
    delete[] sums;
    delete[] rows;
    delete[] pair_normals;
    delete[] packed_even;
    delete[] packed_odd;
}

export void downsample_normal_map(const uniform SourceImage * uniform src, uniform DownsampledImage * uniform dst, uniform NormalMapFormat normal_map_format, uniform DownsamplingContext * uniform ctx) {
    uniform uint32 pixel_size = normal_map_format == R8g8b8 ? 3 : 2;
    uniform bool packed = src->pixel_stride == pixel_size && dst->pixel_stride == pixel_size;
    uniform bool rgbx = normal_map_format == R8g8b8 && src->pixel_stride == 4 && dst->pixel_stride == 4;
    if (src->width == 2 * dst->width && src->height == 2 * dst->height && (packed || rgbx)) {
        normal_map_2x2(src, dst, normal_map_format);
        return;
    }
    if (normal_map_format == R8g8b8)
        resample(src, dst, ctx, SOURCE_FLAT_U8, src->pixel_stride, src->pixel_stride, 3, OUTPUT_NORMAL_RGB8, false, 1.0f);
    else
        resample(src, dst, ctx, SOURCE_NORMAL_RG8, 2, 3, 2, OUTPUT_NORMAL_RG8, false, 1.0f);
}
