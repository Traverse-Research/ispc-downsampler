use std::{f32::EPSILON, collections::HashMap, rc::Rc};

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

#[derive(Debug)]
pub struct CachedCoefficients {
    pub start: u32,
    pub coefficients: Rc<Vec<f32>>,
}

pub fn calculate_coefficients(src: u32, target: u32) -> Vec<CachedCoefficients> {

    assert!(src > target, "Trying to use downsampler to upsample or perform an operation which will cause no changes");

    let mut variables = vec![CoefficientVariables::default(); target as usize];

    unsafe {
        ispc::downsample_ispc::calculate_coefficient_variables(src, target, variables.as_mut_ptr());
    };

    let image_scale = src as f32 / target as f32;

    let mut res = Vec::with_capacity(target as usize);

    let mut reuse_heap = HashMap::<_, Rc<Vec<f32>>>::with_capacity(target as usize / 2);

    for v in variables.iter() {

        let coefficient_count = (v.src_end - v.src_start + 1.0) as u32;
        let reuse_key = (coefficient_count, (v.src_center - v.src_start).to_ne_bytes());

        let reused = reuse_heap.get(&reuse_key);

        let coefficients = if let Some(coefficients) = reused {
            coefficients.clone()
        } else {
            let mut coefficients = vec![0.0; coefficient_count as usize];
            unsafe {
                ispc::downsample_ispc::calculate_coefficients(image_scale, v as *const _, coefficients.as_mut_ptr());
            };
            let coefficients = Rc::new(coefficients);
            reuse_heap.insert(reuse_key, coefficients.clone());
            coefficients
        };

        let cached = CachedCoefficients {
            start: v.src_start as u32,
            coefficients,
        };

        res.push(cached);
    }

    res
}

#[test]
fn verify_cached_coefficients() {
    use resize::{Scale, lanczos};
    use std::num::NonZeroUsize;
    use fallible_collections::TryHashMap;

    let ispc_coefficients = calculate_coefficients(2048, 512);

    let mut recycled_coeffs = TryHashMap::with_capacity(512).unwrap();
    let resize_rs_coefficients = Scale::calc_coeffs(NonZeroUsize::new(2048).unwrap(), 512, (&|x| lanczos(3.0, x), 3.0), &mut recycled_coeffs).unwrap();

    for (index, (ispc, rs)) in ispc_coefficients.iter().zip(resize_rs_coefficients.iter()).enumerate() {
        let mut same = true;
        same &= ispc.start == rs.start as u32;
        same &= ispc.coefficients.len() == rs.coeffs.len();

        for (ispc, rs) in ispc.coefficients.iter().zip(rs.coeffs.iter()) {
            if ispc - rs > EPSILON {
                println!("Reeeeeeee");
                same &= false;
            }
        }

        if !same {
            panic!("Difference found between ispc and resize rs coefficients at index {}.\n ispc: {:?}\n rs: {:?}", index, ispc, rs)
        }
    }
}
