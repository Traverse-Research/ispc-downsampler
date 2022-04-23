use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stb_image::image::{load, LoadResult};
use ispc_downsampler::{Image, downsample, Format};
use std::path::Path;

pub fn criterion_benchmark(c: &mut Criterion) {
    if let LoadResult::ImageU8(img) = load(Path::new("test_assets/square_test.png")) {
        let src_fmt = if img.data.len() / (img.width * img.height) == 4 {
            Format::RGBA8
        } else {
            Format::RGB8
        };

        println!("Loaded image!");

        let src_img = Image::new(&img.data, img.width as u32, img.height as u32, src_fmt);

        let target_width = (img.width / 4) as u32;
        let target_height = (img.height / 4) as u32;

        c.bench_function("Downsample `square_test.png`", |b| b.iter(||
            downsample(&src_img, target_width, target_height)
        ));
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
