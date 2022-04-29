pub mod ispc;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Format {
    RGB8,
    RGBA8,
}
impl Format {
    fn num_channels(&self) -> u8 {
        match self {
            Self::RGB8 => 3,
            Self::RGBA8 => 4,
        }
    }
}

pub struct Image<'a> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
    format: Format,
}

impl<'a> Image<'a> {
    pub fn new(pixels: &'a [u8], width: u32, height: u32, format: Format) -> Self {
        Self {
            pixels,
            width,
            height,
            format,
        }
    }
}

pub fn downsample(src: &Image, target_width: u32, target_height: u32) -> Vec<u8> {
    assert!(src.width >= target_width, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");
    assert!(src.height >= target_height, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");

    let output_size = (target_width * target_height * src.format.num_channels() as u32) as usize;
    let mut output : Vec<u8> = Vec::with_capacity(output_size);
    unsafe { output.set_len(output_size); }

    // intermediate upscale to f32's so we only have to do the conversion once
    let intermediate_size = (src.width * src.height * src.format.num_channels() as u32) as usize;
    let mut intermediate: Vec<f32> = Vec::with_capacity(intermediate_size);
    unsafe { intermediate.set_len(intermediate_size); }


    unsafe {
        ispc::downsample_ispc::convert_f32(
            src.width,
            src.height,
            src.format.num_channels(),
            src.pixels.as_ptr(),
            intermediate.as_mut_ptr(),
        )
    }

    unsafe {
        ispc::downsample_ispc::resample(
            src.width,
            src.height,
            src.width,
            src.format.num_channels(),
            target_width,
            target_height,
            intermediate.as_ptr(),
            output.as_mut_ptr(),
        )
    }

    output
}
