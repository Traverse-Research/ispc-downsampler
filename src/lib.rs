use std::{collections::HashMap, rc::Rc};

use ispc::WeightCollection;

mod ispc;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Format {
    Rgb8Unorm,
    Rgb8Snorm,
    Srgb8,
    Rgba8Unorm,
    Rgba8Snorm,
    Srgba8,
}

impl Format {
    pub fn num_channels(&self) -> usize {
        match self {
            Self::Rgb8Unorm | Self::Rgb8Snorm | Self::Srgb8 => 3,
            Self::Rgba8Unorm | Self::Rgba8Snorm | Self::Srgba8 => 4,
        }
    }

    pub fn pixel_size(&self) -> usize {
        self.channel_size_in_bytes() * self.num_channels()
    }

    pub fn channel_size_in_bytes(&self) -> usize {
        match self {
            Format::Rgb8Unorm
            | Format::Rgb8Snorm
            | Format::Srgb8
            | Format::Rgba8Unorm
            | Format::Rgba8Snorm
            | Format::Srgba8 => 1,
        }
    }
}

impl From<Format> for ispc::downsample_ispc::PixelFormat {
    fn from(value: Format) -> Self {
        match value {
            Format::Rgb8Unorm => ispc::PixelFormat_Rgb8Unorm,
            Format::Rgb8Snorm => ispc::PixelFormat_Rgb8Snorm,
            Format::Srgb8 => ispc::PixelFormat_Rgb8Unorm,
            Format::Rgba8Unorm => ispc::PixelFormat_Rgba8Unorm,
            Format::Rgba8Snorm => ispc::PixelFormat_Rgba8Snorm,
            Format::Srgba8 => ispc::PixelFormat_Rgba8Unorm,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum NormalMapFormat {
    R8g8b8,
    R8g8TangentSpaceReconstructedZ,
}

impl NormalMapFormat {
    pub fn pixel_size(self) -> usize {
        match self {
            NormalMapFormat::R8g8b8 => 3,
            NormalMapFormat::R8g8TangentSpaceReconstructedZ => 2,
        }
    }

    pub fn channel_size_in_bytes(self) -> usize {
        match self {
            Self::R8g8b8 | Self::R8g8TangentSpaceReconstructedZ => 1,
        }
    }
}

impl From<NormalMapFormat> for ispc::NormalMapFormat {
    fn from(value: NormalMapFormat) -> ispc::NormalMapFormat {
        match value {
            NormalMapFormat::R8g8b8 => ispc::NormalMapFormat_R8g8b8,
            NormalMapFormat::R8g8TangentSpaceReconstructedZ => {
                ispc::NormalMapFormat_R8g8TangentSpaceReconstructedZ
            }
        }
    }
}

/// Describes a source image which can be used for [`downsample()`]
/// The pixel data is stored as a slice to avoid unnecessarily cloning it.
pub struct Image<'a> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
}

impl<'a> Image<'a> {
    /// Creates a new source image from the given pixel data slice, dimensions and format.
    pub fn new(pixels: &'a [u8], width: u32, height: u32) -> Self {
        Self {
            pixels,
            width,
            height,
        }
    }
}

/// Scales the alpha to the downscaled texture to preserve the overall alpha coverage.
///
/// If alpha cutoff is specified, any alpha value above it is considered visible of
/// which the percentage of visible texels will be. Otherwise, visibility is considered
/// a linear sum of the alpha values instead and the source and target alpha coverage
/// are calculated the same way.
pub fn scale_alpha_to_original_coverage(
    src: &Image<'_>,
    downsampled: &Image<'_>,
    alpha_cutoff: Option<f32>,
    format: Format,
) -> Vec<u8> {
    assert!(
        matches!(format, Format::Rgba8Unorm | Format::Rgba8Snorm),
        "Cannot retain alpha coverage on image with no alpha channel"
    );
    let mut alpha_scaled_data = downsampled.pixels.to_vec();
    unsafe {
        ispc::downsample_ispc::scale_to_alpha_coverage(
            src.width,
            src.height,
            src.pixels.as_ptr(),
            downsampled.width,
            downsampled.height,
            alpha_scaled_data.as_mut_ptr(),
            alpha_cutoff
                .as_ref()
                .map_or(std::ptr::null(), |alpha_cutoff| alpha_cutoff),
        );
    }
    alpha_scaled_data
}
// Defines a line of weights. `coefficients` contains a weight for each pixel after `start`
#[derive(Debug, Clone)]
struct CachedWeight {
    pub start: u32,
    pub coefficients: Rc<Vec<f32>>,
}

pub(crate) fn calculate_weights(src: u32, target: u32, filter_scale: f32) -> Vec<CachedWeight> {
    assert!(
        src >= target,
        "Trying to use downsampler to upsample or perform an operation which will cause no changes"
    );
    // Every line of weights is based on the start and end of the line, and its "center" which has the biggest weight.
    // These weight lines follow a pattern, so we can skip calculating some of them by caching all different line we get.
    // For that purpose, we first determine the variables which define the line.
    let mut variables = vec![ispc::WeightDimensions::default(); target as usize];

    unsafe {
        ispc::downsample_ispc::calculate_weight_dimensions(
            filter_scale,
            src,
            target,
            variables.as_mut_ptr(),
        );
    }

    let image_scale = src as f32 / target as f32;

    let mut res = Vec::with_capacity(target as usize);

    // We cache the weights in a map so that we can reuse them as we need.
    // Half of the total number of weights seems like a good starting point to avoid unnecessary copies when resizing.
    let mut reuse_heap = HashMap::<_, Rc<Vec<f32>>>::with_capacity(target as usize / 2);

    for v in variables.iter() {
        let coefficient_count = (v.src_end - v.src_start + 1.0) as u32;
        // The unique values that define a collection of cached weights are how many pixels it includes and the distance from its start to its center.
        // We use them to create a key based on which we reuse ones we've calculated previously.
        let reuse_key = (
            coefficient_count,
            (v.src_center - v.src_start).to_ne_bytes(),
        );

        let reused = reuse_heap.get(&reuse_key);

        // If there is already a weight line calculated for that key, we clone it since it's an `Rc`.
        // If there isn't, we calculate the weights and add them to the reuse heap.
        let coefficients = if let Some(coefficients) = reused {
            coefficients.clone()
        } else {
            let mut coefficients = vec![0.0; coefficient_count as usize];
            unsafe {
                ispc::downsample_ispc::calculate_weights_lanczos(
                    image_scale,
                    filter_scale,
                    v as *const _,
                    coefficients.as_mut_ptr(),
                );
            }
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
/// For a more fine-tunable version of this function, see [downsample_with_custom_scale].
pub fn downsample(
    src: &Image<'_>,
    target_width: u32,
    target_height: u32,
    pixel_stride_in_bytes: usize,
    format: Format,
) -> Vec<u8> {
    downsample_with_custom_scale(
        src,
        target_width,
        target_height,
        3.0,
        pixel_stride_in_bytes,
        format,
    )
}

fn precompute_lanczos_weights(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    filter_scale: f32,
) -> ispc::Weights {
    assert!(src_width != dst_width || src_height != dst_height, "Trying to downsample to an image of the same resolution as the source image. This operation can be avoided.");
    assert!(src_width >= dst_width, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");
    assert!(src_height >= dst_height, "The height of the source image is less than the target's height. You are trying to upsample rather than downsample");
    assert!(
        filter_scale > 0.0,
        "filter_scale must be more than 0.0 when downsampling."
    );

    // The weights are calculated per-axis, and are only based on the source and target dimensions of that axis.
    // Because of that, if both axes have the same source and target dimensions, they will have the same weights.
    let width_weights =
        WeightCollection::new(calculate_weights(src_width, dst_width, filter_scale));
    let height_weights = if src_width == src_height && dst_width == dst_height {
        width_weights.clone()
    } else {
        WeightCollection::new(calculate_weights(src_height, dst_height, filter_scale))
    };

    ispc::Weights::new(width_weights, height_weights)
}

/// Version of [downsample] which allows for a custom filter scale, thus trading between speed and final image quality.
///
/// `filter_scale` controls how many samples are made relative to the size ratio between the source and target resolutions.
/// The higher the scale, the more detail is preserved, but the slower the downsampling is. Note that the effect on the detail becomes smaller the higher the scale is.
///
/// As a guideline, a `filter_scale` of 3.0 preserves detail well.
/// A scale of 1.0 preserves is good if speed is necessary, but still preserves a decent amount of detail.
/// Anything below is even faster, although the loss of detail becomes clear.
pub fn downsample_with_custom_scale(
    src: &Image<'_>,
    target_width: u32,
    target_height: u32,
    filter_scale: f32,
    pixel_stride_in_bytes: usize,
    format: Format,
) -> Vec<u8> {
    assert!(format.pixel_size() <= pixel_stride_in_bytes, "The stride between the pixels cannot be lower than the minimum size of the pixel according to the pixel format.");

    let sample_weights = precompute_lanczos_weights(
        src.width,
        src.height,
        target_width,
        target_height,
        filter_scale,
    );

    // The new implementation needs a src_height * target_width intermediate buffer.
    let mut scratch_space =
        vec![0u8; (src.height * target_width * format.num_channels() as u32) as usize];

    let mut output =
        vec![0u8; (target_width * target_height * format.num_channels() as u32) as usize];

    unsafe {
        if format.num_channels() == 3 {
            ispc::downsample_ispc::resample_with_cached_weights_3(
                &ispc::SourceImage {
                    width: src.width,
                    height: src.height,
                    data: src.pixels.as_ptr(),
                    pixel_stride: pixel_stride_in_bytes as u32,
                },
                &mut ispc::DownsampledImage {
                    width: target_width,
                    height: target_height,
                    data: output.as_mut_ptr(),
                    pixel_stride: pixel_stride_in_bytes as u32,
                },
                ispc::PixelFormat::from(format),
                &mut ispc::DownsamplingContext {
                    weights: *sample_weights.ispc_representation(),
                    scratch_space: scratch_space.as_mut_ptr(),
                },
            );
        } else {
            ispc::downsample_ispc::resample_with_cached_weights_4(
                &ispc::SourceImage {
                    width: src.width,
                    height: src.height,
                    data: src.pixels.as_ptr(),
                    pixel_stride: pixel_stride_in_bytes as u32,
                },
                &mut ispc::DownsampledImage {
                    width: target_width,
                    height: target_height,
                    data: output.as_mut_ptr(),
                    pixel_stride: pixel_stride_in_bytes as u32,
                },
                ispc::PixelFormat::from(format),
                &mut ispc::DownsamplingContext {
                    weights: *sample_weights.ispc_representation(),
                    scratch_space: scratch_space.as_mut_ptr(),
                },
            );
        }
    }

    output
}

pub fn downsample_normal_map(
    src: &Image<'_>,
    target_width: u32,
    target_height: u32,
    pixel_stride_in_bytes: usize,
    normal_map_format: NormalMapFormat,
) -> Vec<u8> {
    let mut data =
        vec![0u8; (target_width * target_height) as usize * pixel_stride_in_bytes];

    unsafe {
        ispc::downsample_normal_map(
            &ispc::SourceImage {
                width: src.width,
                height: src.height,
                data: src.pixels.as_ptr(),
                pixel_stride: pixel_stride_in_bytes as u32,
            },
            &mut ispc::DownsampledImage {
                width: target_width,
                height: target_height,
                data: data.as_mut_ptr(),
                pixel_stride: pixel_stride_in_bytes as u32,
            },
            ispc::NormalMapFormat::from(normal_map_format),
        );
    }

    data
}
