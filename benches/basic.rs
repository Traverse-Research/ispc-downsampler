use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ispc_downsampler::{downsample, Format, Image};
use resize::{px::RGB, Type::Lanczos3};
use stb_image::image::{load, LoadResult};
use std::path::Path;

pub fn ispc_downsampler(c: &mut Criterion) {
    if let LoadResult::ImageU8(img) = load(Path::new("test_assets/square_test.png")) {
        let src_fmt = if img.data.len() / (img.width * img.height) == 4 {
            Format::RGBA8
        } else {
            Format::RGB8
        };

        let src_img = Image::new(&img.data, img.width as u32, img.height as u32, src_fmt);

        let target_width = (img.width / 4) as u32;
        let target_height = (img.height / 4) as u32;

        c.bench_function("Downsample `square_test.png` using ispc_downsampler", |b| {
            b.iter(|| downsample(&src_img, target_width, target_height))
        });
    }
}

pub fn resize_rs(c: &mut Criterion) {
    if let LoadResult::ImageU8(img) = load(Path::new("test_assets/square_test.png")) {
        let src_fmt = if img.data.len() / (img.width * img.height) == 4 {
            Format::RGBA8
        } else {
            Format::RGB8
        };

        let src_img = Image::new(&img.data, img.width as u32, img.height as u32, src_fmt);

        let target_width = img.width / 4;
        let target_height = img.height / 4;

        let src = vec![RGB::new(0, 0, 0); img.width * img.height];
        let mut dst = vec![RGB::new(0, 0, 0); target_width * target_height];

        c.bench_function("Downsample `square_test.png` using resize", |b| {
            b.iter(|| {
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
