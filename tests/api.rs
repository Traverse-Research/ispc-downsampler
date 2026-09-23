//! Property tests for the public API. They check invariants (sizes, energy conservation, equivalences between
//! code paths) rather than golden images, so they hold across filter tweaks that don't change behaviour.

use ispc_downsampler::{
    downsample, downsample_normal_map, downsample_normal_map_into, downsample_with_alpha_weighting,
    downsample_with_custom_scale, scale_alpha_to_original_coverage, AlbedoFormat, Image,
    ImagePixelFormat, NormalMapFormat,
};

const ALBEDO_FORMATS: [AlbedoFormat; 6] = [
    AlbedoFormat::Rgb8Unorm,
    AlbedoFormat::Rgb8Snorm,
    AlbedoFormat::Srgb8,
    AlbedoFormat::Rgba8Unorm,
    AlbedoFormat::Rgba8Snorm,
    AlbedoFormat::Srgba8,
];

const NORMAL_FORMATS: [NormalMapFormat; 2] = [
    NormalMapFormat::Rgb8,
    NormalMapFormat::Rg8TangentSpaceReconstructedZ,
];

/// (source, target) pairs: halving, odd ratios, non-square, non-power-of-two, tiny and 1-pixel-wide images.
const SHAPES: [((u32, u32), (u32, u32)); 12] = [
    ((64, 64), (32, 32)),
    ((64, 64), (16, 16)),
    ((64, 64), (1, 1)),
    ((64, 64), (63, 63)),
    ((64, 32), (16, 8)),
    ((32, 64), (8, 16)),
    ((100, 60), (33, 20)),
    ((37, 23), (12, 7)),
    ((2, 2), (1, 1)),
    ((3, 3), (1, 1)),
    ((1, 8), (1, 4)),
    ((8, 1), (4, 1)),
];

/// Deterministic noise, so failures are reproducible without a rand dependency.
fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn mean(data: &[u8], stride: usize, channel: usize) -> f64 {
    let values = data.iter().skip(channel).step_by(stride);
    values.clone().map(|&v| v as f64).sum::<f64>() / values.count() as f64
}

fn assert_close(a: &[u8], b: &[u8], tolerance: u8, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0);
    assert!(
        worst <= tolerance,
        "{what}: off by {worst} (tolerance {tolerance})"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Albedo

#[test]
fn output_size_matches_target() {
    for format in ALBEDO_FORMATS {
        let channels = format.num_channel_in_memory();
        for ((sw, sh), (tw, th)) in SHAPES {
            let data = noise((sw * sh) as usize * channels, sw * 31 + sh);
            let out = downsample(&Image::new(&data, sw, sh, format), tw, th);
            assert_eq!(
                out.len(),
                (tw * th) as usize * channels,
                "{format:?} {sw}x{sh} -> {tw}x{th}"
            );
        }
    }
}

#[test]
fn constant_image_stays_constant_for_every_format_and_shape() {
    for format in ALBEDO_FORMATS {
        let channels = format.num_channel_in_memory();
        for ((sw, sh), (tw, th)) in SHAPES {
            for value in [0u8, 1, 77, 128, 254, 255] {
                let data = vec![value; (sw * sh) as usize * channels];
                let out = downsample(&Image::new(&data, sw, sh, format), tw, th);
                assert!(
                    out.iter().all(|&v| v == value),
                    "{format:?} {sw}x{sh} -> {tw}x{th}: {value} became {:?}",
                    out.iter().find(|&&v| v != value)
                );
            }
        }
    }
}

#[test]
fn constant_image_stays_constant_for_every_filter_scale() {
    for scale in [0.25f32, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0] {
        let data = vec![173u8; 48 * 48 * 3];
        let src = Image::new(&data, 48, 48, AlbedoFormat::Rgb8Unorm);
        for target in [24, 16, 5, 1] {
            let out = downsample_with_custom_scale(&src, target, target, scale);
            assert!(
                out.iter().all(|&v| v == 173),
                "scale {scale}, target {target}"
            );
        }
    }
}

#[test]
fn average_brightness_is_preserved() {
    for ((sw, sh), (tw, th)) in SHAPES {
        let data = noise((sw * sh) as usize * 3, 7);
        let out = downsample(&Image::new(&data, sw, sh, AlbedoFormat::Rgb8Unorm), tw, th);
        for channel in 0..3 {
            let (before, after) = (mean(&data, 3, channel), mean(&out, 3, channel));
            // Noise at 1x1 is dominated by which pixels land in the centre lobe, so allow more there.
            let tolerance = if tw * th <= 4 { 40.0 } else { 3.0 };
            assert!(
                (before - after).abs() <= tolerance,
                "{sw}x{sh} -> {tw}x{th} channel {channel}: mean {before:.2} -> {after:.2}"
            );
        }
    }
}

#[test]
fn downsample_uses_filter_scale_3() {
    let data = noise(64 * 64 * 4, 3);
    let src = Image::new(&data, 64, 64, AlbedoFormat::Rgba8Unorm);
    assert_eq!(
        downsample(&src, 20, 20),
        downsample_with_custom_scale(&src, 20, 20, 3.0)
    );
}

#[test]
fn channels_are_filtered_independently() {
    // Filtering one channel must not depend on the others, so RGB and the RGB of opaque RGBA must agree.
    let rgb = noise(40 * 40 * 3, 11);
    let rgba = rgb
        .chunks(3)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect::<Vec<_>>();
    let out_rgb = downsample(&Image::new(&rgb, 40, 40, AlbedoFormat::Rgb8Unorm), 13, 13);
    let out_rgba = downsample(&Image::new(&rgba, 40, 40, AlbedoFormat::Rgba8Unorm), 13, 13);
    let out_rgba_rgb = out_rgba
        .chunks(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect::<Vec<_>>();
    assert_eq!(out_rgb, out_rgba_rgb);
    assert!(out_rgba.chunks(4).all(|p| p[3] == 255));

    // A single non-zero channel stays isolated.
    let red_only = rgb.chunks(3).flat_map(|p| [p[0], 0, 0]).collect::<Vec<_>>();
    let out = downsample(
        &Image::new(&red_only, 40, 40, AlbedoFormat::Rgb8Unorm),
        13,
        13,
    );
    assert!(out.chunks(3).all(|p| p[1] == 0 && p[2] == 0));
    assert!(out
        .chunks(3)
        .map(|p| p[0])
        .eq(out_rgb.chunks(3).map(|p| p[0])));
}

#[test]
fn mirrored_input_gives_mirrored_output() {
    let (w, h) = (32u32, 32u32);
    let data = noise((w * h * 3) as usize, 5);
    let mirrored = data
        .chunks((w * 3) as usize)
        .flat_map(|row| row.chunks(3).rev().flatten().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let out = downsample(&Image::new(&data, w, h, AlbedoFormat::Rgb8Unorm), 16, 16);
    let out_mirrored = downsample(
        &Image::new(&mirrored, w, h, AlbedoFormat::Rgb8Unorm),
        16,
        16,
    );
    let unmirrored = out_mirrored
        .chunks(16 * 3)
        .flat_map(|row| row.chunks(3).rev().flatten().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_close(&out, &unmirrored, 1, "horizontal mirror");
}

#[test]
fn pixel_stride_matches_tightly_packed() {
    for (format, packed_channels) in [
        (AlbedoFormat::Rgb8Unorm, 3),
        (AlbedoFormat::Rgba8Unorm, 4),
        (AlbedoFormat::Srgb8, 3),
        (AlbedoFormat::Srgba8, 4),
    ] {
        for stride in [packed_channels + 1, 8] {
            let (w, h) = (24u32, 18u32);
            let padded = noise((w * h) as usize * stride, stride as u32);
            let packed = padded
                .chunks(stride)
                .flat_map(|p| p[..packed_channels].to_vec())
                .collect::<Vec<_>>();

            let expected = downsample(&Image::new(&packed, w, h, format), 8, 6);
            let out = downsample(
                &Image::new_with_pixel_stride(&padded, w, h, format, stride),
                8,
                6,
            );

            assert_eq!(out.len(), 8 * 6 * stride, "{format:?} stride {stride}");
            for (o, e) in out.chunks(stride).zip(expected.chunks(packed_channels)) {
                assert_eq!(&o[..packed_channels], e, "{format:?} stride {stride}");
                assert!(
                    o[packed_channels..].iter().all(|&v| v == 0),
                    "padding written"
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "same resolution")]
fn same_size_panics() {
    let data = vec![0u8; 8 * 8 * 3];
    downsample(&Image::new(&data, 8, 8, AlbedoFormat::Rgb8Unorm), 8, 8);
}

#[test]
#[should_panic(expected = "upsample")]
fn upsampling_width_panics() {
    let data = vec![0u8; 8 * 8 * 3];
    downsample(&Image::new(&data, 8, 8, AlbedoFormat::Rgb8Unorm), 16, 4);
}

#[test]
#[should_panic(expected = "upsample")]
fn upsampling_height_panics() {
    let data = vec![0u8; 8 * 8 * 3];
    downsample(&Image::new(&data, 8, 8, AlbedoFormat::Rgb8Unorm), 4, 16);
}

#[test]
#[should_panic(expected = "filter_scale")]
fn zero_filter_scale_panics() {
    let data = vec![0u8; 8 * 8 * 3];
    downsample_with_custom_scale(&Image::new(&data, 8, 8, AlbedoFormat::Rgb8Unorm), 4, 4, 0.0);
}

#[test]
#[should_panic(expected = "stride")]
fn stride_below_pixel_size_panics() {
    let data = vec![0u8; 8 * 8 * 4];
    downsample(
        &Image::new_with_pixel_stride(&data, 8, 8, AlbedoFormat::Rgba8Unorm, 3),
        4,
        4,
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Alpha weighting

#[test]
fn alpha_weighting_does_not_change_alpha() {
    for ((sw, sh), (tw, th)) in SHAPES {
        let data = noise((sw * sh * 4) as usize, 13);
        let src = Image::new(&data, sw, sh, AlbedoFormat::Rgba8Unorm);
        let plain = downsample_with_custom_scale(&src, tw, th, 3.0);
        let weighted = downsample_with_alpha_weighting(&src, tw, th, 3.0);
        assert!(
            plain
                .chunks(4)
                .map(|p| p[3])
                .eq(weighted.chunks(4).map(|p| p[3])),
            "{sw}x{sh} -> {tw}x{th}"
        );
    }
}

#[test]
fn alpha_weighting_matches_plain_on_opaque_image() {
    let data = noise(40 * 40 * 4, 17)
        .chunks(4)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect::<Vec<_>>();
    let src = Image::new(&data, 40, 40, AlbedoFormat::Rgba8Unorm);
    for scale in [1.0, 3.0] {
        let plain = downsample_with_custom_scale(&src, 13, 13, scale);
        let weighted = downsample_with_alpha_weighting(&src, 13, 13, scale);
        assert_close(&plain, &weighted, 1, "opaque image");
    }
}

#[test]
fn alpha_weighting_falls_back_to_plain_when_fully_transparent() {
    let data = noise(32 * 32 * 4, 19)
        .chunks(4)
        .flat_map(|p| [p[0], p[1], p[2], 0])
        .collect::<Vec<_>>();
    let src = Image::new(&data, 32, 32, AlbedoFormat::Rgba8Unorm);
    assert_eq!(
        downsample_with_custom_scale(&src, 8, 8, 3.0),
        downsample_with_alpha_weighting(&src, 8, 8, 3.0)
    );
}

#[test]
fn alpha_weighting_ignores_colour_under_transparent_texels() {
    // Opaque texels are one colour; the colour under transparent texels is noise and must not leak through.
    let (w, h) = (64u32, 64u32);
    let garbage = noise((w * h * 3) as usize, 23);
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            if (x / 5 + y / 3) % 3 == 0 {
                data.extend([200, 90, 30, 255]);
            } else {
                data.extend([garbage[i], garbage[i + 1], garbage[i + 2], 0]);
            }
        }
    }
    let src = Image::new(&data, w, h, AlbedoFormat::Rgba8Unorm);
    for target in [32, 16, 7, 1] {
        let out = downsample_with_alpha_weighting(&src, target, target, 3.0);
        for p in out.chunks(4).filter(|p| p[3] >= 8) {
            assert_close(
                &p[..3],
                &[200, 90, 30],
                2,
                &format!("visible texel at {target}"),
            );
        }
    }
}

#[test]
fn alpha_weighting_keeps_constant_image_constant() {
    for alpha in [1u8, 50, 255] {
        let data = [30u8, 140, 220, alpha].repeat(48 * 48);
        let src = Image::new(&data, 48, 48, AlbedoFormat::Rgba8Unorm);
        let out = downsample_with_alpha_weighting(&src, 17, 11, 3.0);
        assert!(
            out.chunks(4).all(|p| p == [30, 140, 220, alpha]),
            "alpha {alpha}"
        );
    }
}

#[test]
#[should_panic(expected = "no alpha channel")]
fn alpha_weighting_requires_alpha() {
    let data = vec![0u8; 8 * 8 * 3];
    downsample_with_alpha_weighting(&Image::new(&data, 8, 8, AlbedoFormat::Rgb8Unorm), 4, 4, 3.0);
}

// ---------------------------------------------------------------------------------------------------------------
// Alpha coverage

/// Cut-out alpha with soft edges, like a foliage texture.
fn cutout(w: u32, h: u32) -> Vec<u8> {
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - w as f32 / 2.0, y as f32 - h as f32 / 2.0);
            let d = (dx * dx + dy * dy).sqrt() / (w.min(h) as f32 / 2.0);
            let wave = ((x as f32 * 0.9).sin() * (y as f32 * 0.7).cos() * 0.3) + 0.8 - d;
            data.extend([60, 160, 40, (wave * 4.0 * 255.0).clamp(0.0, 255.0) as u8]);
        }
    }
    data
}

fn coverage(data: &[u8], cutoff: f32) -> f32 {
    let n = data.len() / 4;
    data.chunks(4)
        .filter(|p| p[3] as f32 / 255.0 > cutoff)
        .count() as f32
        / n as f32
}

#[test]
fn alpha_coverage_only_changes_alpha() {
    let data = cutout(64, 64);
    let src = Image::new(&data, 64, 64, AlbedoFormat::Rgba8Unorm);
    let down = downsample(&src, 16, 16);
    for cutoff in [Some(0.5), None] {
        let scaled = scale_alpha_to_original_coverage(
            &src,
            &Image::new(&down, 16, 16, AlbedoFormat::Rgba8Unorm),
            cutoff,
        );
        assert_eq!(scaled.len(), down.len());
        assert!(
            scaled
                .chunks(4)
                .zip(down.chunks(4))
                .all(|(a, b)| a[..3] == b[..3]),
            "rgb changed with cutoff {cutoff:?}"
        );
    }
}

#[test]
fn alpha_coverage_is_preserved_through_a_mip_chain() {
    let size = 128;
    let data = cutout(size, size);
    let original = coverage(&data, 0.5);
    let mut pixels = data.clone();
    let mut s = size;
    while s > 4 {
        let src = Image::new(&pixels, s, s, AlbedoFormat::Rgba8Unorm);
        let down = downsample(&src, s / 2, s / 2);
        pixels = scale_alpha_to_original_coverage(
            &src,
            &Image::new(&down, s / 2, s / 2, AlbedoFormat::Rgba8Unorm),
            Some(0.5),
        );
        s /= 2;
        // Quantisation to s² texels limits how close coverage can get at small sizes.
        let tolerance = 0.03 + 2.0 / s as f32;
        let now = coverage(&pixels, 0.5);
        assert!(
            (now - original).abs() <= tolerance,
            "{s}x{s}: coverage {original:.3} -> {now:.3}"
        );
    }
}

#[test]
fn alpha_coverage_scaling_is_monotonic() {
    // Scaling alpha by a single factor must keep the order of alpha values.
    let data = cutout(64, 64);
    let src = Image::new(&data, 64, 64, AlbedoFormat::Rgba8Unorm);
    let down = downsample(&src, 32, 32);
    let scaled = scale_alpha_to_original_coverage(
        &src,
        &Image::new(&down, 32, 32, AlbedoFormat::Rgba8Unorm),
        Some(0.5),
    );
    let mut pairs = down
        .chunks(4)
        .map(|p| p[3])
        .zip(scaled.chunks(4).map(|p| p[3]))
        .collect::<Vec<_>>();
    pairs.sort();
    assert!(
        pairs.windows(2).all(|w| w[0].1 <= w[1].1),
        "alpha order changed"
    );
    assert!(
        pairs
            .iter()
            .all(|&(before, after)| before != 0 || after == 0),
        "0 alpha became visible"
    );
}

#[test]
#[should_panic(expected = "no alpha channel")]
fn alpha_coverage_requires_alpha() {
    let data = vec![0u8; 8 * 8 * 3];
    let small = vec![0u8; 4 * 4 * 3];
    scale_alpha_to_original_coverage(
        &Image::new(&data, 8, 8, AlbedoFormat::Rgb8Unorm),
        &Image::new(&small, 4, 4, AlbedoFormat::Rgb8Unorm),
        None,
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Normal maps

fn decode(format: NormalMapFormat, p: &[u8]) -> [f32; 3] {
    let x = p[0] as f32 / 255.0 * 2.0 - 1.0;
    let y = p[1] as f32 / 255.0 * 2.0 - 1.0;
    let z = match format {
        NormalMapFormat::Rgb8 => p[2] as f32 / 255.0 * 2.0 - 1.0,
        NormalMapFormat::Rg8TangentSpaceReconstructedZ => (1.0 - x * x - y * y).max(0.0).sqrt(),
    };
    [x, y, z]
}

fn encode(n: [f32; 3], channels: usize) -> Vec<u8> {
    n[..channels]
        .iter()
        .map(|v| ((v * 0.5 + 0.5) * 255.0).round() as u8)
        .collect()
}

/// Random unit normals in the upper hemisphere.
fn random_normals(count: usize, channels: usize, seed: u32) -> Vec<u8> {
    noise(count * 2, seed)
        .chunks(2)
        .flat_map(|r| {
            let (x, y) = (
                r[0] as f32 / 255.0 * 1.2 - 0.6,
                r[1] as f32 / 255.0 * 1.2 - 0.6,
            );
            let z = (1.0 - x * x - y * y).sqrt();
            encode([x, y, z], channels)
        })
        .collect()
}

#[test]
fn normal_map_output_size_and_padding() {
    for format in NORMAL_FORMATS {
        let channels = format.num_channel_in_memory();
        for ((sw, sh), (tw, th)) in SHAPES {
            let data = random_normals((sw * sh) as usize, channels, sw + sh);
            let out = downsample_normal_map(&Image::new(&data, sw, sh, format), tw, th);
            assert_eq!(
                out.len(),
                (tw * th) as usize * channels,
                "{format:?} {sw}x{sh} -> {tw}x{th}"
            );
        }

        // With a larger stride, padding is documented to be 255.
        let stride = channels + 2;
        let data = random_normals(16 * 16, channels, 29)
            .chunks(channels)
            .flat_map(|p| [p, &[7, 7][..]].concat())
            .collect::<Vec<_>>();
        let out = downsample_normal_map(
            &Image::new_with_pixel_stride(&data, 16, 16, format, stride),
            8,
            8,
        );
        assert_eq!(out.len(), 8 * 8 * stride);
        assert!(
            out.chunks(stride).all(|p| p[channels..] == [255, 255]),
            "{format:?} padding"
        );
    }
}

#[test]
fn normal_map_stride_matches_tightly_packed() {
    // Padding by 1 and 2 bytes: for 2:1 this compares the dedicated 2:1 kernels (packed Rgb8, RGBX, packed Rg8) with
    // the generic box path (the other strides), which must agree exactly.
    for format in NORMAL_FORMATS {
        let channels = format.num_channel_in_memory();
        let packed = random_normals(20 * 20, channels, 31);
        for padding in [1, 2] {
            let stride = channels + padding;
            let padded = packed
                .chunks(channels)
                .flat_map(|p| [p, &[0, 0][..padding]].concat())
                .collect::<Vec<_>>();
            for (tw, th) in [(10, 10), (7, 5)] {
                let expected = downsample_normal_map(&Image::new(&packed, 20, 20, format), tw, th);
                let out = downsample_normal_map(
                    &Image::new_with_pixel_stride(&padded, 20, 20, format, stride),
                    tw,
                    th,
                );
                assert!(
                    out.chunks(stride)
                        .map(|p| &p[..channels])
                        .eq(expected.chunks(channels)),
                    "{format:?} stride {stride} to {tw}x{th}"
                );
            }
        }
    }
}

#[test]
fn normal_map_outputs_unit_normals() {
    for format in NORMAL_FORMATS {
        let channels = format.num_channel_in_memory();
        let data = random_normals(64 * 64, channels, 37);
        for target in [32, 13, 4, 1] {
            let out = downsample_normal_map(&Image::new(&data, 64, 64, format), target, target);
            for p in out.chunks(channels) {
                let [x, y, z] = decode(format, p);
                let len = (x * x + y * y + z * z).sqrt();
                // One quantisation step per channel is ~0.008, so allow a few.
                assert!(
                    (len - 1.0).abs() < 0.025,
                    "{format:?} at {target}: |n| = {len} for {p:?}"
                );
            }
        }
    }
}

#[test]
fn normal_map_constant_tilted_normal_is_preserved() {
    for format in NORMAL_FORMATS {
        let channels = format.num_channel_in_memory();
        let n = [0.3f32, -0.5, (1.0f32 - 0.09 - 0.25).sqrt()];
        let pixel = encode(n, channels);
        let data = pixel.repeat(32 * 32);
        for (tw, th) in [(16, 16), (5, 3), (1, 1)] {
            let out = downsample_normal_map(&Image::new(&data, 32, 32, format), tw, th);
            for p in out.chunks(channels) {
                assert_close(p, &pixel, 1, &format!("{format:?} at {tw}x{th}"));
            }
        }
    }
}

#[test]
fn normal_map_opposite_tilts_average_to_straight_up() {
    for format in NORMAL_FORMATS {
        let channels = format.num_channel_in_memory();
        let left = encode([0.6, 0.0, 0.8], channels);
        let right = encode([-0.6, 0.0, 0.8], channels);
        // Alternating columns, so every 2x2 box sees both.
        let data = (0..16 * 16)
            .flat_map(|i| {
                if i % 2 == 0 {
                    left.clone()
                } else {
                    right.clone()
                }
            })
            .collect::<Vec<_>>();
        let out = downsample_normal_map(&Image::new(&data, 16, 16, format), 8, 8);
        for p in out.chunks(channels) {
            let [x, y, z] = decode(format, p);
            assert!(
                x.abs() < 0.01 && y.abs() < 0.01 && z > 0.99,
                "{format:?}: {:?}",
                [x, y, z]
            );
        }
    }
}

#[test]
fn normal_map_into_matches_and_overwrites_a_reused_buffer() {
    // Every layout and a 2:1 as well as a generic ratio, into a buffer full of garbage from earlier use.
    for (format, stride) in [
        (NormalMapFormat::Rgb8, 3),
        (NormalMapFormat::Rgb8, 4),
        (NormalMapFormat::Rgb8, 5),
        (NormalMapFormat::Rg8TangentSpaceReconstructedZ, 2),
        (NormalMapFormat::Rg8TangentSpaceReconstructedZ, 3),
    ] {
        let channels = format.num_channel_in_memory();
        let data = random_normals(40 * 24, channels, 59)
            .chunks(channels)
            .flat_map(|p| [p, &[7, 7, 7][..stride - channels]].concat())
            .collect::<Vec<_>>();
        let src = Image::new_with_pixel_stride(&data, 40, 24, format, stride);
        for (tw, th) in [(20, 12), (13, 7)] {
            let expected = downsample_normal_map(&src, tw, th);
            let mut reused = noise(expected.len() + 17, 61);
            downsample_normal_map_into(&src, tw, th, &mut reused);
            assert_eq!(
                &reused[..expected.len()],
                &expected[..],
                "{format:?} stride {stride} to {tw}x{th}"
            );
            assert!(
                expected
                    .chunks(stride)
                    .all(|p| p[channels..].iter().all(|&v| v == 255)),
                "{format:?} stride {stride}: padding"
            );
        }
    }
}

#[test]
#[should_panic(expected = "output needs")]
fn normal_map_into_too_small_panics() {
    let data = vec![128u8; 8 * 8 * 3];
    let mut output = vec![0u8; 4 * 4 * 3 - 1];
    downsample_normal_map_into(
        &Image::new(&data, 8, 8, NormalMapFormat::Rgb8),
        4,
        4,
        &mut output,
    );
}

#[test]
#[should_panic(expected = "pixel stride")]
fn normal_map_stride_below_pixel_size_panics() {
    let data = vec![0u8; 8 * 8 * 3];
    downsample_normal_map(
        &Image::new_with_pixel_stride(&data, 8, 8, NormalMapFormat::Rgb8, 2),
        4,
        4,
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Mip chains

#[test]
fn mip_chain_keeps_average_brightness() {
    let size = 256;
    let data = noise(size * size * 4, 41);
    let original = (0..4).map(|c| mean(&data, 4, c)).collect::<Vec<_>>();
    let mut pixels = data;
    let mut s = size as u32;
    while s > 8 {
        pixels = downsample(
            &Image::new(&pixels, s, s, AlbedoFormat::Rgba8Unorm),
            s / 2,
            s / 2,
        );
        s /= 2;
        for (c, before) in original.iter().enumerate() {
            let after = mean(&pixels, 4, c);
            assert!(
                (before - after).abs() < 2.0,
                "{s}x{s} channel {c}: {before:.2} -> {after:.2}"
            );
        }
    }
}

#[test]
fn mip_chain_non_square_non_power_of_two() {
    // Halving with rounding down, as a cook does for odd sizes, all the way to 1x1.
    let (mut w, mut h) = (300u32, 77u32);
    let mut pixels = vec![99u8; (w * h * 3) as usize];
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        pixels = downsample(&Image::new(&pixels, w, h, AlbedoFormat::Rgb8Unorm), nw, nh);
        (w, h) = (nw, nh);
        assert_eq!(pixels.len(), (w * h * 3) as usize);
        assert!(pixels.iter().all(|&v| v == 99), "{w}x{h}");
    }
}

#[test]
fn normal_mip_chain_stays_unit_length() {
    for format in NORMAL_FORMATS {
        let channels = format.num_channel_in_memory();
        let mut pixels = random_normals(128 * 128, channels, 43);
        let mut s = 128;
        while s > 1 {
            pixels = downsample_normal_map(&Image::new(&pixels, s, s, format), s / 2, s / 2);
            s /= 2;
            for p in pixels.chunks(channels) {
                let [x, y, z] = decode(format, p);
                assert!(
                    ((x * x + y * y + z * z).sqrt() - 1.0).abs() < 0.025,
                    "{format:?} at {s}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// sRGB (https://github.com/Traverse-Research/ispc-downsampler/issues/25)

fn linear_to_srgb(linear: f64) -> u8 {
    let s = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

#[test]
fn srgb_filters_in_linear_space() {
    // Alternating black and white rows: the correct average is 50% linear light, not 50% of the encoded value.
    let (w, h) = (16u32, 32u32);
    for (format, channels) in [(AlbedoFormat::Srgb8, 3), (AlbedoFormat::Srgba8, 4)] {
        let data = (0..h)
            .flat_map(|y| vec![if y % 2 == 0 { 0u8 } else { 255 }; (w as usize) * channels])
            .collect::<Vec<_>>();
        let out = downsample(&Image::new(&data, w, h, format), w, h / 2);
        let expected = linear_to_srgb(0.5);
        assert_eq!(expected, 188);
        // Rows near the border see a truncated, renormalized filter, so only check the interior.
        let row = (w as usize) * channels;
        for p in out[row * 3..out.len() - row * 3].chunks(channels) {
            assert_close(&p[..3], &[expected; 3], 1, &format!("{format:?}"));
        }

        let unorm = if channels == 3 {
            AlbedoFormat::Rgb8Unorm
        } else {
            AlbedoFormat::Rgba8Unorm
        };
        let out = downsample(&Image::new(&data, w, h, unorm), w, h / 2);
        assert_close(
            &out[row * 3..row * 3 + 3],
            &[128; 3],
            1,
            "unorm averages encoded values",
        );
    }
}

#[test]
fn srgb_round_trips_every_value() {
    for format in [AlbedoFormat::Srgb8, AlbedoFormat::Srgba8] {
        let channels = format.num_channel_in_memory();
        for value in 0..=255u8 {
            let data = vec![value; 16 * 16 * channels];
            let out = downsample(&Image::new(&data, 16, 16, format), 8, 8);
            assert!(out.iter().all(|&v| v == value), "{format:?} {value}");
        }
    }
}

#[test]
fn srgba_alpha_stays_linear() {
    let data = noise(32 * 32 * 4, 47);
    let srgb = downsample(&Image::new(&data, 32, 32, AlbedoFormat::Srgba8), 11, 11);
    let unorm = downsample(&Image::new(&data, 32, 32, AlbedoFormat::Rgba8Unorm), 11, 11);
    // sRGB reads alpha widened to u16 and unorm reads it as u8, so float rounding may differ by 1.
    let alpha = |v: &[u8]| v.chunks(4).map(|p| p[3]).collect::<Vec<_>>();
    assert_close(&alpha(&srgb), &alpha(&unorm), 1, "alpha");
    assert_ne!(
        srgb, unorm,
        "rgb should differ once filtered in linear space"
    );
}

#[test]
fn srgb_alpha_weighting_filters_in_linear_space() {
    // Alpha weighting and sRGB compose: opaque stripes of black and white, transparent garbage in between.
    let (w, h) = (16u32, 48u32);
    let data = (0..h)
        .flat_map(|y| {
            let p = match y % 3 {
                0 => [0, 0, 0, 255],
                1 => [255, 255, 255, 255],
                _ => [255, 0, 0, 0],
            };
            p.repeat(w as usize)
        })
        .collect::<Vec<_>>();
    let out = downsample_with_alpha_weighting(
        &Image::new(&data, w, h, AlbedoFormat::Srgba8),
        w,
        h / 3,
        1.0,
    );
    let row = w as usize * 4;
    for p in out[row * 3..out.len() - row * 3].chunks(4) {
        assert!(
            p[1] == p[0] && p[2] == p[0],
            "transparent red leaked: {p:?}"
        );
        assert!(p[0] > 150, "averaged in encoded space: {p:?}");
    }
}

#[test]
fn alpha_coverage_accepts_srgba() {
    // Alpha is linear in sRGB formats too, so coverage must work and match the unorm result.
    let data = cutout(64, 64);
    let down = downsample(&Image::new(&data, 64, 64, AlbedoFormat::Rgba8Unorm), 32, 32);
    let scale = |format| {
        scale_alpha_to_original_coverage(
            &Image::new(&data, 64, 64, format),
            &Image::new(&down, 32, 32, format),
            Some(0.5),
        )
    };
    assert_eq!(scale(AlbedoFormat::Srgba8), scale(AlbedoFormat::Rgba8Unorm));
}

#[test]
fn pass_order_does_not_matter() {
    // The horizontal and vertical passes commute only if nothing is clamped or quantized in between. Lanczos
    // has negative lobes, so high-contrast input overshoots after the first pass; clamping it there would make
    // filtering the transposed image give a different result.
    let size = 48usize;
    let data = noise(size * size * 4, 53)
        .into_iter()
        .map(|v| if v > 127 { 255 } else { 0 })
        .collect::<Vec<_>>();
    let transpose = |d: &[u8], n: usize| {
        let mut t = vec![0u8; d.len()];
        for y in 0..n {
            for x in 0..n {
                t[(x * n + y) * 4..][..4].copy_from_slice(&d[(y * n + x) * 4..][..4]);
            }
        }
        t
    };
    for format in [AlbedoFormat::Rgba8Unorm, AlbedoFormat::Srgba8] {
        for target in [24u32, 17] {
            let t = target as usize;
            let out = downsample(&Image::new(&data, 48, 48, format), target, target);
            let transposed = transpose(&data, size);
            let out_t = downsample(&Image::new(&transposed, 48, 48, format), target, target);
            assert_close(
                &out,
                &transpose(&out_t, t),
                1,
                &format!("{format:?} to {target}"),
            );
        }
    }
}
