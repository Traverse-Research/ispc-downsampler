use image::{RgbImage, RgbaImage};
use ispc_downsampler::{downsample_with_custom_scale, Format, Image};
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
                Format::Rgba8
            } else {
                Format::Rgb8
            };

            println!("Loaded image!");

            let src_img = Image::new(&img.data, img.width as u32, img.height as u32, src_fmt);

            let target_width = (img.width / 4) as u32;
            let target_height = (img.height / 4) as u32;

            let now = Instant::now();
            println!("Downsampling started!");
            let downsampled_pixels =
                downsample_with_custom_scale(&src_img, target_width, target_height, 1.0);
            println!("Finished downsampling in {:.2?}!", now.elapsed());

            std::fs::create_dir_all("example_outputs").unwrap();
            match src_fmt {
                Format::Rgba8 | Format::Srgba8 => {
                    let save_image =
                        RgbaImage::from_vec(target_width, target_height, downsampled_pixels)
                            .unwrap();
                    save_image
                        .save("example_outputs/square_test_result.png")
                        .unwrap()
                }
                Format::Rgb8 | Format::Srgb8 => {
                    let save_image =
                        RgbImage::from_vec(target_width, target_height, downsampled_pixels)
                            .unwrap();
                    save_image
                        .save("example_outputs/square_test_result.png")
                        .unwrap()
                }
            }
        }
        _ => panic!("This test only works with 8-bit per channel textures"),
    }
}
