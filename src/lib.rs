pub mod ispc;

pub struct Image<'a> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
}

impl<'a> Image<'a> {
    pub fn new(pixels: &'a [u8], width: u32, height: u32) -> Self {
        Self {
            pixels,
            width,
            height,
        }
    }
}

pub fn downsample(src: & Image, target_width: u32, target_height: u32) -> Vec<u8> {
    assert!(src.width >= target_width, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");
    assert!(src.height >= target_height, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");
    assert!(
        (src.width * src.height * 4) as usize == src.pixels.len(),
        "TEMPORARY: The source image is not RGBA8: The expected size was {}, but {} was found",
        src.width * src.height * 4,
        src.pixels.len()
    );

    let mut output = Vec::new();
    // TODO: This and the kernel both assume RGBA8 textures. This will crash and burn with RGB8
    output.resize((target_width * target_height * 4) as usize, 0);

    unsafe {
        ispc::downsample_ispc::resample(
            src.width,
            src.height,
            src.width,
            target_width,
            target_height,
            src.pixels.as_ptr(),
            output.as_mut_ptr(),
        )
    }

    output
}
