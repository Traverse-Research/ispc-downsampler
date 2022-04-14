

use image::{RgbaImage};
use stb_image::image::{load, LoadResult};
use std::path::Path;
use std::time::Instant;
use ispc_downsampler::{downsample, Image};


fn main() {
    let res = load(Path::new("test_assets/square_test.png"));
    match res {
        LoadResult::Error(str) => panic!("Image loading error: {}", str),
        LoadResult::ImageU8(img) => {
            assert!(!img.data.is_empty());

            let mut corrected = Vec::new();
            corrected.reserve(img.width * img.height * 4);
            for i in 0..(img.data.len() / 3) {
                corrected.push(img.data[i * 3]);
                corrected.push(img.data[i * 3 + 1]);
                corrected.push(img.data[i * 3 + 2]);
                corrected.push(255);
            }

            println!("Loaded image!");
    
            let mut src_img = Image::new(&mut corrected, img.width as u32, img.height as u32);
    
            let target_width =  (img.width / 4) as u32;
            let target_height =  (img.height / 4) as u32;
        
            let now = Instant::now();
            println!("Downsampling started!");
            let downsampled_pixels = downsample(&mut src_img, target_width, target_height);
            println!("Downsampling done! {} seconds elapsed", now.elapsed().as_millis() as f32 / 1000.0);
    
            if !std::path::Path::exists(Path::new("example_outputs")) {
                std::fs::create_dir("example_outputs").unwrap();
            }
            let mut save_image = RgbaImage::new(target_width, target_height);
            save_image.copy_from_slice(&downsampled_pixels);
            save_image.save("example_outputs/square_test_result.png").unwrap();
        }
        _ => {}
    }
}
