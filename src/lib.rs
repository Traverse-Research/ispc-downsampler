use std::{collections::HashMap, rc::Rc};

use ispc::downsample_ispc::{Cache, WeightVariables};
use ispc::WeightCollection;

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
///
/// Will result in noticably worse images than [`downsample`]. Use this only if you need to quickly sample down and don't care about the quality.
pub fn downsample_fast(src: &Image, target_width: u32, target_height: u32) -> Vec<u8> {
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

#[derive(Debug, Clone)]
struct CachedWeight {
    pub start: u32,
    pub coefficients: Rc<Vec<f32>>,
}

pub(crate) fn calculate_weights(src: u32, target: u32) -> Vec<CachedWeight> {
    assert!(
        src > target,
        "Trying to use downsampler to upsample or perform an operation which will cause no changes"
    );

    let mut variables = vec![WeightVariables::default(); target as usize];

    unsafe {
        ispc::downsample_ispc::calculate_weight_variables(src, target, variables.as_mut_ptr());
    };

    let image_scale = src as f32 / target as f32;

    let mut res = Vec::with_capacity(target as usize);

    let mut reuse_heap = HashMap::<_, Rc<Vec<f32>>>::with_capacity(target as usize / 2);

    for v in variables.iter() {
        let coefficient_count = (v.src_end - v.src_start + 1.0) as u32;
        let reuse_key = (
            coefficient_count,
            (v.src_center - v.src_start).to_ne_bytes(),
        );

        let reused = reuse_heap.get(&reuse_key);

        let coefficients = if let Some(coefficients) = reused {
            coefficients.clone()
        } else {
            let mut coefficients = vec![0.0; coefficient_count as usize];
            unsafe {
                ispc::downsample_ispc::calculate_weights(
                    image_scale,
                    v as *const _,
                    coefficients.as_mut_ptr(),
                );
            };
            let coefficients = Rc::new(coefficients);
            reuse_heap.insert(reuse_key, coefficients.clone());
            coefficients
        };

        let cached = CachedWeight {
            start: v.src_start as u32,
            coefficients,
        };

        res.push(cached);
    }

    res
}

/// Samples the provided image down to the specified width and height.
/// `target_width` and `target_height` are expected to be less than or equal to their `src` counter parts.
/// Will panic if the target dimensions are the same as the source image's.
///
/// Preserves the detail of the source image well.
/// If a faster implementation is needed regardless of the final image quality, see [`downsample_fast`].
pub fn downsample(src: &Image, target_width: u32, target_height: u32) -> Vec<u8> {
    assert!(src.width != target_width || src.height != target_height, "Trying to downsample to an image of the same resolution as the source image. This can be avoided.");
    assert!(src.width >= target_width, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");
    assert!(src.height >= target_height, "The height of the source image is less than the target's height. You are trying to upsample rather than downsample");

    let width_weights = WeightCollection::new(calculate_weights(src.width, target_width));
    let height_weights = if src.width == src.height && target_width == target_height {
        width_weights.clone()
    } else {
        WeightCollection::new(calculate_weights(src.height, target_height))
    };

    let mut scratch_space = Vec::new();
    scratch_space.resize(
        (src.height * target_width * src.format.num_channels() as u32) as usize,
        0u8,
    );

    let mut output = Vec::new();
    output.resize(
        (target_width * target_height * src.format.num_channels() as u32) as usize,
        0,
    );

    let weight_cache = Cache {
        vertical_weights: width_weights.ispc_representation(),
        horizontal_weights: height_weights.ispc_representation(),
    };
    unsafe {
        ispc::downsample_ispc::resample_with_cache(
            src.width,
            src.height,
            target_width,
            target_height,
            src.format.num_channels(),
            &weight_cache as *const _,
            scratch_space.as_mut_ptr(),
            src.pixels.as_ptr(),
            output.as_mut_ptr(),
        )
    }

    output
}
