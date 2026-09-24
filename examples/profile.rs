//! Profiling driver: runs one or all workloads in a loop for a fixed time, for use under a sampling profiler.
//! usage: profile [workload|all] [seconds per workload]

use ispc_downsampler::*;
use stb_image::image::{load, LoadResult};
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(String::as_str).unwrap_or("all");
    let seconds: f64 = args.get(2).map_or(2.0, |s| s.parse().unwrap());

    let LoadResult::ImageU8(img) = load(Path::new("test_assets/square_test.png")) else {
        panic!("failed to load test image")
    };
    let size = img.width as u32;
    let rgb = img.data;
    let rgba: Vec<u8> = rgb
        .chunks(3)
        .flat_map(|p| {
            let luma = (p[0] as u32 * 3 + p[1] as u32 * 6 + p[2] as u32) / 10;
            [p[0], p[1], p[2], if luma > 100 { 255 } else { 0 }]
        })
        .collect();
    let rg: Vec<u8> = rgb.chunks(3).flat_map(|p| [p[0], p[1]]).collect();
    let down = downsample(
        &Image::new(&rgba, size, size, AlbedoFormat::Rgba8Unorm),
        size / 2,
        size / 2,
    );

    type Workload<'a> = (&'static str, Box<dyn Fn() + 'a>);
    let workloads: Vec<Workload> = vec![
        (
            "rgb8",
            Box::new(|| {
                black_box(downsample(
                    &Image::new(&rgb, size, size, AlbedoFormat::Rgb8Unorm),
                    size / 2,
                    size / 2,
                ));
            }),
        ),
        (
            "rgba8",
            Box::new(|| {
                black_box(downsample(
                    &Image::new(&rgba, size, size, AlbedoFormat::Rgba8Unorm),
                    size / 2,
                    size / 2,
                ));
            }),
        ),
        (
            "srgba8",
            Box::new(|| {
                black_box(downsample(
                    &Image::new(&rgba, size, size, AlbedoFormat::Srgba8),
                    size / 2,
                    size / 2,
                ));
            }),
        ),
        (
            "rgba8_alpha_weighted",
            Box::new(|| {
                black_box(downsample_with_alpha_weighting(
                    &Image::new(&rgba, size, size, AlbedoFormat::Rgba8Unorm),
                    size / 2,
                    size / 2,
                    3.0,
                ));
            }),
        ),
        (
            "srgba8_alpha_weighted",
            Box::new(|| {
                black_box(downsample_with_alpha_weighting(
                    &Image::new(&rgba, size, size, AlbedoFormat::Srgba8),
                    size / 2,
                    size / 2,
                    3.0,
                ));
            }),
        ),
        (
            "rgb8_to_256",
            Box::new(|| {
                black_box(downsample(
                    &Image::new(&rgb, size, size, AlbedoFormat::Rgb8Unorm),
                    256,
                    256,
                ));
            }),
        ),
        (
            "normal_rgb8",
            Box::new(|| {
                black_box(downsample_normal_map(
                    &Image::new(&rgb, size, size, NormalMapFormat::Rgb8),
                    size / 2,
                    size / 2,
                ));
            }),
        ),
        (
            "normal_rg8",
            Box::new(|| {
                black_box(downsample_normal_map(
                    &Image::new(
                        &rg,
                        size,
                        size,
                        NormalMapFormat::Rg8TangentSpaceReconstructedZ,
                    ),
                    size / 2,
                    size / 2,
                ));
            }),
        ),
        (
            "coverage_cutoff",
            Box::new(|| {
                black_box(scale_alpha_to_original_coverage(
                    &Image::new(&rgba, size, size, AlbedoFormat::Rgba8Unorm),
                    &Image::new(&down, size / 2, size / 2, AlbedoFormat::Rgba8Unorm),
                    Some(0.5),
                ));
            }),
        ),
        (
            "coverage_linear",
            Box::new(|| {
                black_box(scale_alpha_to_original_coverage(
                    &Image::new(&rgba, size, size, AlbedoFormat::Rgba8Unorm),
                    &Image::new(&down, size / 2, size / 2, AlbedoFormat::Rgba8Unorm),
                    None,
                ));
            }),
        ),
    ];

    for (name, run) in &workloads {
        if which != "all" && which != *name {
            continue;
        }
        let start = Instant::now();
        let mut iterations = 0u32;
        while start.elapsed() < Duration::from_secs_f64(seconds) {
            run();
            iterations += 1;
        }
        println!(
            "{name}: {:.2} ms/iter",
            start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
        );
    }
}
