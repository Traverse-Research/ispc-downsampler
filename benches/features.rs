//! Benchmarks covering the public API: formats, ratios, filter scales, strides, normal maps,
//! alpha weighting, alpha coverage and full mip chains.
//! Throughput is reported in source pixels, so different sizes and ratios are comparable.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ispc_downsampler::{
    downsample, downsample_normal_map, downsample_normal_map_into, downsample_with_alpha_weighting,
    downsample_with_custom_scale, scale_alpha_to_original_coverage, AlbedoFormat, Image,
    NormalMapFormat,
};
use stb_image::image::{load, LoadResult};
use std::path::Path;

/// 2048x2048 RGB photo.
fn load_rgb() -> (Vec<u8>, u32) {
    match load(Path::new("test_assets/square_test.png")) {
        LoadResult::ImageU8(img) => {
            assert_eq!((img.width, img.height, img.depth), (2048, 2048, 3));
            (img.data, img.width as u32)
        }
        _ => panic!("failed to load test_assets/square_test.png"),
    }
}

/// RGBA with a cut-out alpha (from luminance) so alpha weighting and coverage have edges to work on.
fn to_cutout_rgba(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks(3)
        .flat_map(|p| {
            let luma = (p[0] as u32 * 3 + p[1] as u32 * 6 + p[2] as u32) / 10;
            [p[0], p[1], p[2], if luma > 100 { 255 } else { 0 }]
        })
        .collect()
}

/// Nearest-neighbour crop/resample to an arbitrary size, to get non-square and non-power-of-two sources.
fn resize_nearest(data: &[u8], size: u32, channels: usize, w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h) as usize * channels);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (x * size / w, y * size / h);
            let i = (sy * size + sx) as usize * channels;
            out.extend_from_slice(&data[i..i + channels]);
        }
    }
    out
}

fn pixels(w: u32, h: u32) -> Throughput {
    Throughput::Elements(w as u64 * h as u64)
}

fn ratios(c: &mut Criterion) {
    let (rgb, size) = load_rgb();
    let src = Image::new(&rgb, size, size, AlbedoFormat::Rgb8Unorm);
    let mut group = c.benchmark_group("albedo_ratio_2048");
    group.throughput(pixels(size, size));
    for target in [1024, 512, 256, 64, 8, 1] {
        group.bench_with_input(BenchmarkId::from_parameter(target), &target, |b, &t| {
            b.iter(|| downsample(&src, t, t))
        });
    }
    group.finish();
}

fn filter_scales(c: &mut Criterion) {
    let (rgb, size) = load_rgb();
    let src = Image::new(&rgb, size, size, AlbedoFormat::Rgb8Unorm);
    let mut group = c.benchmark_group("filter_scale_2048_to_512");
    group.throughput(pixels(size, size));
    for scale in [0.5f32, 1.0, 2.0, 3.0, 4.0] {
        group.bench_with_input(BenchmarkId::from_parameter(scale), &scale, |b, &s| {
            b.iter(|| downsample_with_custom_scale(&src, 512, 512, s))
        });
    }
    group.finish();
}

fn formats(c: &mut Criterion) {
    let (rgb, size) = load_rgb();
    let rgba = to_cutout_rgba(&rgb);

    let mut group = c.benchmark_group("format_2048_to_1024");
    group.throughput(pixels(size, size));
    for (name, data, format) in [
        ("rgb8_unorm", &rgb, AlbedoFormat::Rgb8Unorm),
        ("rgb8_snorm", &rgb, AlbedoFormat::Rgb8Snorm),
        ("srgb8", &rgb, AlbedoFormat::Srgb8),
        ("rgba8_unorm", &rgba, AlbedoFormat::Rgba8Unorm),
        ("rgba8_snorm", &rgba, AlbedoFormat::Rgba8Snorm),
        ("srgba8", &rgba, AlbedoFormat::Srgba8),
    ] {
        let src = Image::new(data, size, size, format);
        group.bench_function(name, |b| b.iter(|| downsample(&src, 1024, 1024)));
    }
    // RGB pixels padded to a 4 byte stride, e.g. RGBX data.
    let src = Image::new_with_pixel_stride(&rgba, size, size, AlbedoFormat::Rgb8Unorm, 4);
    group.bench_function("rgb8_unorm_stride4", |b| {
        b.iter(|| downsample(&src, 1024, 1024))
    });
    let src = Image::new(&rgba, size, size, AlbedoFormat::Rgba8Unorm);
    group.bench_function("rgba8_unorm_alpha_weighted", |b| {
        b.iter(|| downsample_with_alpha_weighting(&src, 1024, 1024, 3.0))
    });
    group.finish();
}

/// sRGB decodes on every filter tap and encodes on every write, so compare it against the same data as unorm.
fn srgb(c: &mut Criterion) {
    let (rgb, size) = load_rgb();
    let rgba = to_cutout_rgba(&rgb);
    let mut group = c.benchmark_group("srgb_vs_unorm_2048");
    group.throughput(pixels(size, size));
    for target in [1024, 256] {
        for (name, data, format) in [
            ("rgb8_unorm", &rgb, AlbedoFormat::Rgb8Unorm),
            ("srgb8", &rgb, AlbedoFormat::Srgb8),
            ("rgba8_unorm", &rgba, AlbedoFormat::Rgba8Unorm),
            ("srgba8", &rgba, AlbedoFormat::Srgba8),
        ] {
            let src = Image::new(data, size, size, format);
            group.bench_with_input(BenchmarkId::new(name, target), &target, |b, &t| {
                b.iter(|| downsample(&src, t, t))
            });
        }
        let src = Image::new(&rgba, size, size, AlbedoFormat::Srgba8);
        group.bench_with_input(
            BenchmarkId::new("srgba8_alpha_weighted", target),
            &target,
            |b, &t| b.iter(|| downsample_with_alpha_weighting(&src, t, t, 3.0)),
        );
    }
    group.finish();
}

fn shapes(c: &mut Criterion) {
    let (rgb, size) = load_rgb();
    let mut group = c.benchmark_group("shape");
    for (name, (sw, sh), (tw, th)) in [
        ("1920x1080_to_1280x720", (1920, 1080), (1280, 720)),
        ("1920x1080_to_960x540", (1920, 1080), (960, 540)),
        ("2048x512_to_512x128", (2048, 512), (512, 128)),
        ("512x2048_to_128x512", (512, 2048), (128, 512)),
        ("2048x2048_to_1000x700", (2048, 2048), (1000, 700)),
        ("2048x2048_to_2047x2047", (2048, 2048), (2047, 2047)),
        ("1000x1000_to_333x333", (1000, 1000), (333, 333)),
    ] {
        let data = resize_nearest(&rgb, size, 3, sw, sh);
        let src = Image::new(&data, sw, sh, AlbedoFormat::Rgb8Unorm);
        group.throughput(pixels(sw, sh));
        group.bench_function(name, |b| b.iter(|| downsample(&src, tw, th)));
    }
    group.finish();
}

/// Small images, where precomputing weights and setup dominate.
fn small_images(c: &mut Criterion) {
    let (rgb, size) = load_rgb();
    let mut group = c.benchmark_group("small_rgba_half");
    for s in [4u32, 16, 64, 256] {
        let data = to_cutout_rgba(&resize_nearest(&rgb, size, 3, s, s));
        let src = Image::new(&data, s, s, AlbedoFormat::Rgba8Unorm);
        group.throughput(pixels(s, s));
        group.bench_with_input(BenchmarkId::from_parameter(s), &s, |b, &s| {
            b.iter(|| downsample(&src, s / 2, s / 2))
        });
    }
    group.finish();
}

fn normal_maps(c: &mut Criterion) {
    // Pixel content does not affect the box filter's cost, so the photo doubles as normal data.
    let (rgb, size) = load_rgb();
    let rg = rgb.chunks(3).flat_map(|p| [p[0], p[1]]).collect::<Vec<_>>();
    let rgbx = to_cutout_rgba(&rgb);

    let mut group = c.benchmark_group("normal_map_2048");
    group.throughput(pixels(size, size));
    for target in [1024, 256, 16] {
        let src = Image::new(&rgb, size, size, NormalMapFormat::Rgb8);
        group.bench_with_input(BenchmarkId::new("rgb8", target), &target, |b, &t| {
            b.iter(|| downsample_normal_map(&src, t, t))
        });
        let src = Image::new(
            &rg,
            size,
            size,
            NormalMapFormat::Rg8TangentSpaceReconstructedZ,
        );
        group.bench_with_input(
            BenchmarkId::new("rg8_reconstruct_z", target),
            &target,
            |b, &t| b.iter(|| downsample_normal_map(&src, t, t)),
        );
    }
    let src = Image::new_with_pixel_stride(&rgbx, size, size, NormalMapFormat::Rgb8, 4);
    group.bench_function(BenchmarkId::new("rgb8_stride4", 1024), |b| {
        b.iter(|| downsample_normal_map(&src, 1024, 1024))
    });

    // Writing into a reused buffer, as a pipeline processing many textures would.
    let mut output = vec![0u8; 1024 * 1024 * 4];
    for (name, data, format, stride) in [
        ("rgb8_into", &rgb, NormalMapFormat::Rgb8, 3),
        (
            "rg8_reconstruct_z_into",
            &rg,
            NormalMapFormat::Rg8TangentSpaceReconstructedZ,
            2,
        ),
        ("rgb8_stride4_into", &rgbx, NormalMapFormat::Rgb8, 4),
    ] {
        let src = Image::new_with_pixel_stride(data, size, size, format, stride);
        group.bench_function(BenchmarkId::new(name, 1024), |b| {
            b.iter(|| downsample_normal_map_into(&src, 1024, 1024, &mut output))
        });
    }
    group.finish();
}

fn alpha_coverage(c: &mut Criterion) {
    let (rgb, size) = load_rgb();
    let mut group = c.benchmark_group("alpha_coverage");
    group.sample_size(10);
    for s in [256u32, 1024] {
        let src_data = to_cutout_rgba(&resize_nearest(&rgb, size, 3, s, s));
        let src = Image::new(&src_data, s, s, AlbedoFormat::Rgba8Unorm);
        let down_data = downsample(&src, s / 2, s / 2);
        let down = Image::new(&down_data, s / 2, s / 2, AlbedoFormat::Rgba8Unorm);
        group.throughput(pixels(s, s));
        group.bench_with_input(BenchmarkId::new("cutoff_0.5", s), &s, |b, _| {
            b.iter(|| scale_alpha_to_original_coverage(&src, &down, Some(0.5)))
        });
        group.bench_with_input(BenchmarkId::new("no_cutoff", s), &s, |b, _| {
            b.iter(|| scale_alpha_to_original_coverage(&src, &down, None))
        });
    }
    group.finish();
}

/// What a texture cook does: every mip built from the previous one, down to 1x1.
fn mip_chains(c: &mut Criterion) {
    let (rgb, size) = load_rgb();
    let rgba = to_cutout_rgba(&rgb);

    let chain = |data: &[u8],
                 format: AlbedoFormat,
                 step: &dyn Fn(&Image<'_, AlbedoFormat>, u32) -> Vec<u8>| {
        let mut data = data.to_vec();
        let mut s = size;
        while s > 1 {
            data = step(&Image::new(&data, s, s, format), s / 2);
            s /= 2;
        }
        data
    };

    let mut group = c.benchmark_group("mip_chain_2048");
    group.sample_size(10);
    group.throughput(pixels(size, size));
    group.bench_function("rgb8", |b| {
        b.iter(|| {
            chain(&rgb, AlbedoFormat::Rgb8Unorm, &|src, t| {
                downsample(src, t, t)
            })
        })
    });
    group.bench_function("rgba8", |b| {
        b.iter(|| {
            chain(&rgba, AlbedoFormat::Rgba8Unorm, &|src, t| {
                downsample(src, t, t)
            })
        })
    });
    group.bench_function("srgba8", |b| {
        b.iter(|| chain(&rgba, AlbedoFormat::Srgba8, &|src, t| downsample(src, t, t)))
    });
    group.bench_function("rgba8_alpha_weighted", |b| {
        b.iter(|| {
            chain(&rgba, AlbedoFormat::Rgba8Unorm, &|src, t| {
                downsample_with_alpha_weighting(src, t, t, 3.0)
            })
        })
    });
    group.bench_function("rgba8_alpha_weighted_coverage", |b| {
        b.iter(|| {
            chain(&rgba, AlbedoFormat::Rgba8Unorm, &|src, t| {
                let down = downsample_with_alpha_weighting(src, t, t, 3.0);
                scale_alpha_to_original_coverage(
                    src,
                    &Image::new(&down, t, t, AlbedoFormat::Rgba8Unorm),
                    Some(0.5),
                )
            })
        })
    });
    group.bench_function("normal_rgb8", |b| {
        b.iter(|| {
            let mut data = rgb.clone();
            let mut s = size;
            while s > 1 {
                data = downsample_normal_map(
                    &Image::new(&data, s, s, NormalMapFormat::Rgb8),
                    s / 2,
                    s / 2,
                );
                s /= 2;
            }
            data
        })
    });
    // A cook reusing two buffers for the whole chain instead of allocating every level.
    let mut buffers = [
        vec![0u8; (size / 2 * size / 2 * 3) as usize],
        vec![0u8; (size / 4 * size / 4 * 3) as usize],
    ];
    group.bench_function("normal_rgb8_into", |b| {
        b.iter(|| {
            let [even, odd] = &mut buffers;
            downsample_normal_map_into(
                &Image::new(&rgb, size, size, NormalMapFormat::Rgb8),
                size / 2,
                size / 2,
                even,
            );
            let mut s = size / 2;
            let mut from_even = true;
            while s > 1 {
                let (src, dst) = if from_even {
                    (&*even, &mut *odd)
                } else {
                    (&*odd, &mut *even)
                };
                downsample_normal_map_into(
                    &Image::new(&src[..(s * s * 3) as usize], s, s, NormalMapFormat::Rgb8),
                    s / 2,
                    s / 2,
                    dst,
                );
                from_even = !from_even;
                s /= 2;
            }
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    ratios,
    filter_scales,
    formats,
    srgb,
    shapes,
    small_images,
    normal_maps,
    alpha_coverage,
    mip_chains
);
criterion_main!(benches);
