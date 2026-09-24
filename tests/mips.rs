//! `generate_mips()` and `generate_normal_mips()` must produce exactly what chaining the single-level functions
//! does, in place in the D3D12 upload layout of `mip_layout()`.

use ispc_downsampler::*;

fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

/// Level `index` of `output` as tightly packed rows.
fn read_level(output: &[u8], level: &MipLevel, texel: usize) -> Vec<u8> {
    let row = level.width as usize * texel;
    (0..level.height as usize)
        .flat_map(|y| output[level.offset + y * level.row_pitch..][..row].to_vec())
        .collect()
}

/// Copies tightly packed rows into `level`.
fn write_level(output: &mut [u8], level: &MipLevel, texel: usize, data: &[u8]) {
    let row = level.width as usize * texel;
    for (y, src) in data.chunks_exact(row).enumerate() {
        output[level.offset + y * level.row_pitch..][..row].copy_from_slice(src);
    }
}

#[test]
fn layout_matches_d3d12_footprints() {
    let (levels, size) = mip_layout(5, 3, 4);
    let expect = |offset, row_pitch, width, height| MipLevel {
        offset,
        row_pitch,
        width,
        height,
    };
    assert_eq!(
        levels,
        [
            expect(0, 256, 5, 3),
            expect(1024, 256, 2, 1),
            expect(1536, 256, 1, 1)
        ]
    );
    assert_eq!(size, 1540);

    let (levels, size) = mip_layout(2048, 1024, 4);
    assert_eq!(levels.len(), 12);
    assert_eq!(levels[0], expect(0, 8192, 2048, 1024));
    assert_eq!(levels[1], expect(8192 * 1024, 4096, 1024, 512));
    assert_eq!(
        levels[11],
        MipLevel {
            width: 1,
            height: 1,
            ..levels[11]
        }
    );
    for pair in levels.windows(2) {
        assert_eq!(pair[1].offset % 512, 0);
        assert_eq!(pair[1].row_pitch % 256, 0);
        assert!(pair[1].offset >= pair[0].offset + pair[0].row_pitch * pair[0].height as usize);
    }
    assert_eq!(size, levels[11].offset + 4);
}

/// Every level from the previous one with the single-level functions, as RGBA.
fn reference_chain(
    rgba: &[u8],
    levels: &[MipLevel],
    format: AlbedoFormat,
    options: &MipOptions,
) -> Vec<Vec<u8>> {
    let mut chain = vec![rgba.to_vec()];
    for pair in levels.windows(2) {
        let (previous, level) = (&pair[0], &pair[1]);
        let image = Image::new_with_pixel_stride(
            chain.last().unwrap(),
            previous.width,
            previous.height,
            format,
            4,
        );
        let mut next = if options.alpha_weighting {
            downsample_with_alpha_weighting(&image, level.width, level.height, options.filter_scale)
        } else {
            downsample_with_custom_scale(&image, level.width, level.height, options.filter_scale)
        };
        if format.num_channel_in_memory() == 3 {
            next.as_chunks_mut::<4>()
                .0
                .iter_mut()
                .for_each(|p| p[3] = 255);
        }
        let cutoff = match options.alpha_coverage {
            AlphaCoverage::Unchanged => None,
            AlphaCoverage::Mean => Some(None),
            AlphaCoverage::Cutoff(c) => Some(Some(c)),
        };
        if let Some(cutoff) = cutoff {
            let first = Image::new(&chain[0], levels[0].width, levels[0].height, format);
            next = scale_alpha_to_original_coverage(
                &first,
                &Image::new(&next, level.width, level.height, format),
                cutoff,
            );
        }
        chain.push(next);
    }
    chain
}

#[test]
fn mips_match_chained_downsampling() {
    let formats = [
        AlbedoFormat::Rgb8Unorm,
        AlbedoFormat::Rgb8Snorm,
        AlbedoFormat::Srgb8,
        AlbedoFormat::Rgba8Unorm,
        AlbedoFormat::Rgba8Snorm,
        AlbedoFormat::Srgba8,
    ];
    let alpha_options = [
        MipOptions::default(),
        MipOptions {
            filter_scale: 1.0,
            ..Default::default()
        },
        MipOptions {
            alpha_weighting: true,
            ..Default::default()
        },
        MipOptions {
            alpha_weighting: true,
            alpha_coverage: AlphaCoverage::Cutoff(0.5),
            ..Default::default()
        },
        MipOptions {
            alpha_coverage: AlphaCoverage::Mean,
            ..Default::default()
        },
    ];
    // Tightly packed large levels, padded small ones, and odd sizes where every level is padded.
    for (width, height) in [(256, 128), (100, 60), (1, 9)] {
        for format in formats {
            let channels = format.num_channel_in_memory();
            let pixels = noise((width * height) as usize * channels, width + height);
            let rgba: Vec<u8> = pixels
                .chunks_exact(channels)
                .flat_map(|p| [p[0], p[1], p[2], if channels == 4 { p[3] } else { 255 }])
                .collect();
            let (levels, size) = mip_layout(width, height, 4);
            for options in &alpha_options[..if channels == 4 { 5 } else { 2 }] {
                // Garbage everywhere but level 0, to catch anything that is read before it is written.
                let mut output = noise(size, 99);
                write_level(&mut output, &levels[0], 4, &rgba);
                generate_mips(&mut output, &levels, format, options);
                let reference = reference_chain(&rgba, &levels, format, options);
                // sRGB levels after the first are filtered from the previous level's linear value rather than its
                // rounded sRGB bytes, so they may differ from the chained calls by one step (see
                // `srgb_mips_do_not_drift` for which is right).
                let srgb = matches!(format, AlbedoFormat::Srgb8 | AlbedoFormat::Srgba8);
                for (i, (level, expected)) in levels.iter().zip(&reference).enumerate() {
                    let got = read_level(&output, level, 4);
                    let tolerance = if srgb && i > 1 { 1 } else { 0 };
                    assert!(
                        got.iter()
                            .zip(expected)
                            .all(|(a, b)| a.abs_diff(*b) <= tolerance),
                        "{format:?} {width}x{height} {options:?}: level {i} differs"
                    );
                }
            }
        }
    }
}

#[test]
fn normal_mips_match_chained_downsampling() {
    for format in [
        NormalMapFormat::Rgb8,
        NormalMapFormat::Rg8TangentSpaceReconstructedZ,
    ] {
        let channels = format.num_channel_in_memory();
        let texel = if channels == 3 { 4 } else { 2 };
        for (width, height) in [(256, 128), (100, 60), (1, 9)] {
            let pixels = noise((width * height) as usize * channels, width);
            let (levels, size) = mip_layout(width, height, texel);
            let mut previous: Vec<u8> = pixels
                .chunks_exact(channels)
                .flat_map(|p| {
                    if texel == 4 {
                        vec![p[0], p[1], p[2], 255]
                    } else {
                        p.to_vec()
                    }
                })
                .collect();
            let mut output = noise(size, 5);
            write_level(&mut output, &levels[0], texel, &previous);
            generate_normal_mips(&mut output, &levels, format);
            assert!(read_level(&output, &levels[0], texel) == previous);
            for pair in levels.windows(2) {
                let (p, level) = (&pair[0], &pair[1]);
                let image =
                    Image::new_with_pixel_stride(&previous, p.width, p.height, format, texel);
                previous = downsample_normal_map(&image, level.width, level.height);
                assert!(
                    read_level(&output, level, texel) == previous,
                    "{format:?} {width}x{height} {level:?}"
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "The buffer needs")]
fn mips_too_small_panics() {
    let (levels, size) = mip_layout(16, 16, 4);
    let mut output = vec![0u8; size - 1];
    generate_mips(
        &mut output,
        &levels,
        AlbedoFormat::Rgba8Unorm,
        &MipOptions::default(),
    );
}

fn srgb_to_linear(v: u8) -> f64 {
    let s = v as f64 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(l: f64) -> f64 {
    let l = l.clamp(0.0, 1.0);
    255.0
        * if l <= 0.0031308 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        }
}

/// The library's Lanczos weights for `src` to `target` texels (filter scale 3), in f64: (first texel, weights).
fn lanczos_weights(src: usize, target: usize) -> Vec<(usize, Vec<f64>)> {
    let sinc = |x: f64| {
        if x.abs() < 1e-9 {
            1.0
        } else {
            (x * std::f64::consts::PI).sin() / (x * std::f64::consts::PI)
        }
    };
    let ratio = src as f64 / target as f64;
    let radius = (ratio * 3.0).ceil();
    (0..target)
        .map(|p| {
            let center = (p as f64 + 0.5) * ratio - 0.5;
            let start = ((center - radius).ceil().max(0.0) as usize).min(src - 1);
            let end = ((center + radius).floor().max(0.0) as usize)
                .min(src - 1)
                .max(start);
            let w: Vec<f64> = (start..=end)
                .map(|i| {
                    let t = ((i as f64 - center) / ratio).abs();
                    if t < 3.0 {
                        sinc(t) * sinc(t / 3.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum: f64 = w.iter().sum();
            (start, w.iter().map(|w| w / sum).collect())
        })
        .collect()
}

/// One 2:1 Lanczos step of linear rgb planes in f64, clamped like the library clamps every level.
fn float_step(src: &[[f64; 3]], size: usize) -> Vec<[f64; 3]> {
    let weights = lanczos_weights(size, size / 2);
    let half = size / 2;
    let mut vertical = vec![[0.0; 3]; half * size];
    for (y, (start, w)) in weights.iter().enumerate() {
        for x in 0..size {
            for (i, wi) in w.iter().enumerate() {
                for c in 0..3 {
                    vertical[y * size + x][c] += wi * src[(start + i) * size + x][c];
                }
            }
        }
    }
    let mut out = vec![[0.0; 3]; half * half];
    for y in 0..half {
        for (x, (start, w)) in weights.iter().enumerate() {
            for (i, wi) in w.iter().enumerate() {
                for c in 0..3 {
                    out[y * half + x][c] += wi * vertical[y * size + start + i][c];
                }
            }
            out[y * half + x]
                .iter_mut()
                .for_each(|v| *v = v.clamp(0.0, 1.0));
        }
    }
    out
}

/// Every sRGB level must be the correctly rounded value of the same chain done in f64 without rounding in between,
/// up to float noise at rounding boundaries. Chaining the single-level calls rounds to 8 bits at every level and
/// misses it on a large share of texels.
#[test]
fn srgb_mips_match_an_unrounded_chain() {
    let size = 256;
    let rgba: Vec<u8> = noise(size * size * 4, 3)
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1] / 2 + 64, p[2] / 4 + 16, 255])
        .collect();
    let (levels, total) = mip_layout(size as u32, size as u32, 4);
    let mut output = vec![0u8; total];
    write_level(&mut output, &levels[0], 4, &rgba);
    let options = MipOptions::default();
    generate_mips(&mut output, &levels, AlbedoFormat::Srgba8, &options);
    let chained = reference_chain(&rgba, &levels, AlbedoFormat::Srgba8, &options);

    let mut exact: Vec<[f64; 3]> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| [0, 1, 2].map(|c| srgb_to_linear(p[c])))
        .collect();
    for (k, level) in levels.iter().enumerate().skip(1).take(4) {
        exact = float_step(&exact, level.width as usize * 2);
        let rounded: Vec<u8> = exact
            .iter()
            .flat_map(|p| p.map(|l| linear_to_srgb(l).round() as u8))
            .collect();
        let off = |rgba: &[u8]| {
            rgba.as_chunks::<4>()
                .0
                .iter()
                .zip(rounded.as_chunks::<3>().0)
                .filter(|(a, b)| a[..3] != **b)
                .count() as f64
                / rounded.len() as f64
                * 3.0
        };
        let mips = off(&read_level(&output, level, 4));
        let chained = off(&chained[k]);
        assert!(
            mips < 0.02,
            "mip {k}: {:.1}% of texels are not the rounded exact value",
            mips * 100.0
        );
        if k > 1 {
            assert!(
                mips * 5.0 < chained,
                "mip {k}: generate_mips {mips}, chained calls {chained}"
            );
        }
    }
}

#[test]
fn flat_srgb_mips_stay_flat() {
    let (levels, size) = mip_layout(64, 64, 4);
    for value in 0..=255u8 {
        let mut output = vec![0u8; size];
        write_level(
            &mut output,
            &levels[0],
            4,
            &[value, value, value, 255].repeat(64 * 64),
        );
        generate_mips(
            &mut output,
            &levels,
            AlbedoFormat::Srgba8,
            &MipOptions::default(),
        );
        for level in &levels {
            assert!(
                read_level(&output, level, 4)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|p| *p == [value, value, value, 255]),
                "{value} changed at {}x{}",
                level.width,
                level.height
            );
        }
    }
}
