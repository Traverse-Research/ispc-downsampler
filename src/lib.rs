use std::f32::consts::PI;
use ispc::downsample_ispc::CoefficientVariables;

mod ispc;

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

/// Describes a source image which can be used for with `downsample`
/// The pixel data is stored as a slice to avoid unnecessarily cloning it.
pub struct Image<'a> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
    format: Format,
}

impl<'a> Image<'a> {
    /// Creates a new source image from the given pixel data slice, dimensions and format.
    pub fn new(pixels: &'a [u8], width: u32, height: u32, format: Format) -> Self {
        Self {
            pixels,
            width,
            height,
            format,
        }
    }
}

/// Runs the ISPC kernel on the source image, sampling it down to the `target_width` and `target_height`. Returns the downsampled pixel data as a `Vec<u8>`.
///
/// Will panic if the target width or height are higher than that of the source image.
pub fn downsample(src: &Image, target_width: u32, target_height: u32) -> Vec<u8> {
    assert!(src.width >= target_width, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");
    assert!(src.height >= target_height, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");

    let mut output = Vec::new();
    output.resize(
        (target_width * target_height * src.format.num_channels() as u32) as usize,
        0,
    );

    unsafe {
        ispc::downsample_ispc::resample(
            src.width,
            src.height,
            src.width,
            src.format.num_channels(),
            target_width,
            target_height,
            src.pixels.as_ptr(),
            output.as_mut_ptr(),
        )
    }

    output
}

struct CachedCoefficients {
    start: u32,
    coefficients: Vec<f32>,
}

fn calculate_coefficients(src: u32, target: u32) -> Vec<CachedCoefficients> {

    assert!(src > target, "Trying to use downsampler to upsample or perform an operation which will cause no changes");

    let mut variables = vec![CoefficientVariables::default(); target as usize];

    unsafe {
        ispc::downsample_ispc::calculate_coefficient_variables(src, target, variables.as_mut_ptr());
    };

    let image_scale = src as f32 / target as f32;

    let mut res = Vec::with_capacity(target as usize);

    for v in variables.iter() {

        let mut coefficients = vec![0.0; (v.src_end - v.src_start + 1) as usize];

        unsafe {
            ispc::downsample_ispc::calculate_coefficients(image_scale, v as *const _, coefficients.as_mut_ptr());
        }

        let cached = CachedCoefficients {
            start: v.src_start,
            coefficients,
        };

        res.push(cached);
    }

    res
}
