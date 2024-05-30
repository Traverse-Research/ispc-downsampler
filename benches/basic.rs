use criterion::{criterion_group, criterion_main, Criterion};
use ispc_downsampler::{downsample, AlbedoFormat, Image};
use resize::{px::RGB, Type::Lanczos3};
use stb_image::image::{load, LoadResult};
use std::path::Path;

const DOWNSCALE: usize = 4;

pub fn ispc_downsampler(c: &mut Criterion) {
    if let LoadResult::ImageU8(img) = load(Path::new("test_assets/square_test.png")) {
        let src_fmt = if img.data.len() / (img.width * img.height) == 4 {
            AlbedoFormat::Rgba8Unorm
        } else {
            AlbedoFormat::Rgb8Unorm
        };

        let src_img = Image::new(&img.data, img.width as u32, img.height as u32, src_fmt);

        let target_width = (img.width / DOWNSCALE) as u32;
        let target_height = (img.height / DOWNSCALE) as u32;

        c.bench_function("Downsample `square_test.png` using ispc_downsampler", |b| {
            b.iter(|| downsample(&src_img, target_width, target_height))
        });
    }
}

pub fn resize_rs(c: &mut Criterion) {
    if let LoadResult::ImageU8(img) = load(Path::new("test_assets/square_test.png")) {
        let target_width = img.width / DOWNSCALE;
        let target_height = img.height / DOWNSCALE;

        let src = img
            .data
            .chunks(3)
            .map(|v| RGB::new(v[0], v[1], v[2]))
            .collect::<Vec<_>>();

        c.bench_function("Downsample `square_test.png` using resize", |b| {
            b.iter(|| {
                let mut dst = vec![RGB::new(0, 0, 0); target_width * target_height];
                let mut resizer = resize::new(
                    img.width,
                    img.height,
                    target_width,
                    target_height,
                    resize::Pixel::RGB8,
                    Lanczos3,
                )
                .unwrap();
                resizer.resize(&src, &mut dst)
            })
        });
    }
}

criterion_group!(benches, ispc_downsampler, resize_rs);
criterion_main!(benches);
