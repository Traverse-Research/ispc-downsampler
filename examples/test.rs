use image::{RgbImage, RgbaImage};
use ispc_downsampler::{downsample, Format, Image};
use stb_image::image::{load, LoadResult};
use std::path::Path;
use std::time::Instant;

fn main() {
    let res = load(Path::new("test_assets/square_test.png"));
    match res {
        LoadResult::Error(str) => panic!("Image loading error: {}", str),
        LoadResult::ImageU8(img) => {
            assert!(!img.data.is_empty());

            let src_fmt = if img.data.len() / (img.width * img.height) == 4 {
                Format::RGBA8
            } else {
                Format::RGB8
            };

            println!("Loaded image!");

            let src_img = Image::new(&img.data, img.width as u32, img.height as u32, src_fmt);

            let target_width = (img.width / 4) as u32;
            let target_height = (img.height / 4) as u32;

            let now = Instant::now();
            println!("Downsampling started!");
            let downsampled_pixels = downsample(&src_img, target_width, target_height);
            println!(
                "Downsampling done! {} seconds elapsed",
                now.elapsed().as_millis() as f32 / 1000.0
            );

            if !std::path::Path::exists(Path::new("example_outputs")) {
                std::fs::create_dir("example_outputs").unwrap();
            }

            match src_fmt {
                Format::RGBA8 => {
                    let mut save_image = RgbaImage::new(target_width, target_height);
                    save_image.copy_from_slice(&downsampled_pixels);
                    save_image
                        .save("example_outputs/square_test_result.png")
                        .unwrap()
                }
                Format::RGB8 => {
                    let mut save_image = RgbImage::new(target_width, target_height);
                    save_image.copy_from_slice(&downsampled_pixels);
                    save_image
                        .save("example_outputs/square_test_result.png")
                        .unwrap()
                }
            }
        }
        _ => panic!("This test only works with 8-bit per channel textures"),
    }
}
