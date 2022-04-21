# Image downsampler using a Lanczos filter implemented in ISPC
[![CI](https://github.com/Traverse-Research/ispc-downsampler/actions/workflows/build.yaml/badge.svg)](https://github.com/Traverse-Research/ispc-downsampler/actions/workflows/build.yaml)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![APACHE2](https://img.shields.io/badge/license-APACHE2-blue.svg)
[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v1.4%20adopted-ff69b4.svg)](../main/CODE_OF_CONDUCT.md)

# About
This crate aims to allow for quick down sampling of images by using ISPC. To preserve image sharpness, a Lanczos filter is used. 

The crate downsamples a 2048x2048 image down to 512x512 in ~1.2 seconds, where as other crates tested took over 4 seconds.

The crate comes with the necessary bindings for the ISPC functions, so the ISPC compiler and libclang are not needed unless you are aiming to rebuild them. For that, use `cargo build --features=ispc`. This will expect you to have the ISPC compiler in your global PATH variable.

## Usage
Create a new `ispc_downsampler::Image` from a slice of the texture's pixels, the dimensions of the source image, and the format it is in. Currently only works with RGB8 and RGBA8 textures.
Call `ispc_downsampler::downsample` with the source image, and the target dimension for downsampled image. The function will return a `Vec<u8>` with the pixels of the downsampled image in the same format as the source image.
#### Example
```rust
// Create a the source image from a previously loaded file.
let src_img = Image::new(&img.data, // Pixels as a slice
img.width, // Source width
img.height, // Source height
src_fmt); // Format of the image

// Sample down to half-resolution
let target_width: u32 = img.width / 2;
let target_height: u32 = img.height / 2;

// Run the ISPC kernel, with the result being returned as a Vec<u8>
let downsampled_pixels = downsample(&src_img, target_width, target_height);
```
