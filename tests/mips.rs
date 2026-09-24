//! `generate_mips_into()` and `generate_normal_mips_into()` must produce exactly what chaining the single-level
//! functions does, in the D3D12 upload layout of `mip_layout()`.

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
            next.chunks_exact_mut(4).for_each(|p| p[3] = 255);
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
                // Also from a source with a pixel stride.
                let padded: Vec<u8> = pixels
                    .chunks_exact(channels)
                    .flat_map(|p| [p, &[7u8; 2][..]].concat())
                    .collect();
                for src in [
                    Image::new(&pixels, width, height, format),
                    Image::new_with_pixel_stride(&padded, width, height, format, channels + 2),
                ] {
                    let mut output = noise(size, 99);
                    generate_mips_into(&src, &levels, &mut output, options);
                    let reference = reference_chain(&rgba, &levels, format, options);
                    for (i, (level, expected)) in levels.iter().zip(&reference).enumerate() {
                        assert!(
                            read_level(&output, level, 4) == *expected,
                            "{format:?} {width}x{height} {options:?}: level {i} differs"
                        );
                    }
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
            let mut output = noise(size, 5);
            generate_normal_mips_into(
                &Image::new(&pixels, width, height, format),
                &levels,
                &mut output,
            );

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
#[should_panic(expected = "The output needs")]
fn mips_into_too_small_panics() {
    let pixels = vec![0u8; 16 * 16 * 4];
    let (levels, size) = mip_layout(16, 16, 4);
    let mut output = vec![0u8; size - 1];
    generate_mips_into(
        &Image::new(&pixels, 16, 16, AlbedoFormat::Rgba8Unorm),
        &levels,
        &mut output,
        &MipOptions::default(),
    );
}
