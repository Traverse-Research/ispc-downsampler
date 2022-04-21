# ISPC image downsampler using a Lanczos filter
[![CI](https://github.com/Traverse-Research/ispc-downsampler/actions/workflows/build.yaml/badge.svg)](https://github.com/Traverse-Research/ispc-downsampler/actions/workflows/build.yaml)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![APACHE2](https://img.shields.io/badge/license-APACHE2-blue.svg)
[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v1.4%20adopted-ff69b4.svg)](../main/CODE_OF_CONDUCT.md)

# About
This crate allows for quick downsampling of images by using ISPC. To preserve image sharpness, a Lanczos filter is used. 

The crate downsamples a 2048x2048 image down to 512x512 in ~1.2 seconds, whereas other crates tested took over 4 seconds.

The crate comes with the bindings and precompiled libraries for Windows, Linux and macOS for the ISPC functions, so the ISPC compiler and libclang are not needed unless you are want to rebuild them with different settings. For that, use `cargo build --features=ispc`. This will expect you to have the ISPC compiler in your global PATH variable.

## Usage
Create a new `ispc_downsampler::Image` from a slice of the texture's pixels, the dimensions of the source image, and the format it is in. Currently only works with RGB8 and RGBA8 textures.
Call `ispc_downsampler::downsample` with the source image, and the target dimension for downsampled image. The function will return a `Vec<u8>` with the pixels of the downsampled image in the same format as the source image.
#### Example
```rust
use image::{io::Reader, RgbaImage};
use ispc_downsampler::{downsample, Format, Image};
use std::path::Path;
use std::time::Instant;

fn main() {
    // Load the image using the `image` crate
    let loaded_img = Reader::open(Path::new("test_assets/square_test.png"))
        .expect("Failed to open image file")
        .decode()
        .expect("Failed to decode loaded image");

    // Get the image data as an RGBA8 image for testing purposes.
    let img_rgba8 = loaded_img.into_rgba8();
    let img_size = img_rgba8.dimensions();

    // Create the source image for the downsampler
    let src_img = Image::new(
        &*img_rgba8,       // The image's pixels as a slice
        img_size.0 as u32, // Source image width
        img_size.1 as u32, // Source image height
        Format::RGBA8,     // Source image format
    );

    let target_width = (img_size.0 / 4) as u32;
    let target_height = (img_size.1 / 4) as u32;

    let now = Instant::now();
    println!("Downsampling started!");
    let downsampled_pixels = downsample(&src_img, target_width, target_height);
    println!("Finished downsampling in {:.2?}!", now.elapsed());

    // Save the downsampled image to an image for comparison
    std::fs::create_dir_all("example_outputs").unwrap();
    let save_image = RgbaImage::from_vec(target_width, target_height, downsampled_pixels).unwrap();
    save_image
        .save("example_outputs/square_test_result.png")
        .unwrap();
}
```
