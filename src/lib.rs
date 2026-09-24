use std::{collections::HashMap, rc::Rc};

use ispc::WeightCollection;

mod ispc;

pub trait ImagePixelFormat: Copy {
    /// Returns the number of channels that an image of this format would have in memory.
    /// For example, while a normal map of format [`NormalMapFormat::R8g8TangentSpaceReconstructedZ`] would still have 3 channels when sampled,
    /// in memory it will have 2 channels.
    fn num_channel_in_memory(self) -> usize;

    /// Returns the size of a single value channel, in bytes.
    fn channel_size_in_bytes(self) -> usize;

    /// Returns the size in bytes of a single pixel.
    /// Generally this will be equal to [`channel_size_in_bytes()`][Self::channel_size_in_bytes()] * [`num_channel_in_memory()`][Self::num_channel_in_memory()].
    fn pixel_size_in_bytes(self) -> usize {
        self.channel_size_in_bytes() * self.num_channel_in_memory()
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum AlbedoFormat {
    Rgb8Unorm,
    Rgb8Snorm,
    /// sRGB-encoded color, which is decoded to linear space before filtering and encoded back to
    /// sRGB in the output.
    Srgb8,
    Rgba8Unorm,
    Rgba8Snorm,
    /// sRGB-encoded color with linear alpha, which is decoded to linear space before filtering and
    /// encoded back to sRGB in the output.
    Srgba8,
}

impl ImagePixelFormat for AlbedoFormat {
    fn num_channel_in_memory(self) -> usize {
        match self {
            Self::Rgb8Unorm | Self::Rgb8Snorm | Self::Srgb8 => 3,
            Self::Rgba8Unorm | Self::Rgba8Snorm | Self::Srgba8 => 4,
        }
    }

    fn channel_size_in_bytes(self) -> usize {
        match self {
            AlbedoFormat::Rgb8Unorm
            | AlbedoFormat::Rgb8Snorm
            | AlbedoFormat::Srgb8
            | AlbedoFormat::Rgba8Unorm
            | AlbedoFormat::Rgba8Snorm
            | AlbedoFormat::Srgba8 => 1,
        }
    }
}

impl From<AlbedoFormat> for ispc::downsample_ispc::PixelFormat {
    fn from(value: AlbedoFormat) -> Self {
        match value {
            AlbedoFormat::Rgb8Unorm => ispc::PixelFormat_Rgb8Unorm,
            AlbedoFormat::Rgb8Snorm => ispc::PixelFormat_Rgb8Snorm,
            AlbedoFormat::Srgb8 => ispc::PixelFormat_Srgb8,
            AlbedoFormat::Rgba8Unorm => ispc::PixelFormat_Rgba8Unorm,
            AlbedoFormat::Rgba8Snorm => ispc::PixelFormat_Rgba8Snorm,
            AlbedoFormat::Srgba8 => ispc::PixelFormat_Srgba8,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum NormalMapFormat {
    Rgb8,
    Rg8TangentSpaceReconstructedZ,
}

impl ImagePixelFormat for NormalMapFormat {
    fn num_channel_in_memory(self) -> usize {
        match self {
            NormalMapFormat::Rgb8 => 3,
            NormalMapFormat::Rg8TangentSpaceReconstructedZ => 2,
        }
    }

    fn channel_size_in_bytes(self) -> usize {
        match self {
            Self::Rgb8 | Self::Rg8TangentSpaceReconstructedZ => 1,
        }
    }
}

impl From<NormalMapFormat> for ispc::NormalMapFormat {
    fn from(value: NormalMapFormat) -> ispc::NormalMapFormat {
        match value {
            NormalMapFormat::Rgb8 => ispc::NormalMapFormat_R8g8b8,
            NormalMapFormat::Rg8TangentSpaceReconstructedZ => {
                ispc::NormalMapFormat_R8g8TangentSpaceReconstructedZ
            }
        }
    }
}

/// Describes a source image which can be used for [`downsample()`]
/// The pixel data is stored as a slice to avoid unnecessarily cloning it.
pub struct Image<'a, F: ImagePixelFormat> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
    pixel_stride_in_bytes: usize,
    format: F,
}

impl<'a, F: ImagePixelFormat> Image<'a, F> {
    /// Creates a new source image from the given pixel data slice, dimensions and format.
    pub fn new(pixels: &'a [u8], width: u32, height: u32, format: F) -> Self {
        let pixel_size = format.pixel_size_in_bytes();
        Self::new_with_pixel_stride(pixels, width, height, format, pixel_size)
    }

    pub fn new_with_pixel_stride(
        pixels: &'a [u8],
        width: u32,
        height: u32,
        format: F,
        pixel_stride_in_bytes: usize,
    ) -> Self {
        Self {
            pixels,
            width,
            height,
            pixel_stride_in_bytes,
            format,
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
    src: &Image<'_, AlbedoFormat>,
    downsampled: &Image<'_, AlbedoFormat>,
    alpha_cutoff: Option<f32>,
) -> Vec<u8> {
    assert!(
        matches!(
            src.format,
            AlbedoFormat::Rgba8Unorm | AlbedoFormat::Rgba8Snorm | AlbedoFormat::Srgba8
        ),
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
pub fn downsample(src: &Image<'_, AlbedoFormat>, target_width: u32, target_height: u32) -> Vec<u8> {
    downsample_with_custom_scale(src, target_width, target_height, 3.0)
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

/// If `src` has a pixel stride larger than its format's pixel size, the returned `Vec` uses the same stride, with the padding bytes set to 0.
///
/// Version of [downsample] which allows for a custom filter scale, thus trading between speed and final image quality.
///
/// `filter_scale` controls how many samples are made relative to the size ratio between the source and target resolutions.
/// The higher the scale, the more detail is preserved, but the slower the downsampling is. Note that the effect on the detail becomes smaller the higher the scale is.
///
/// As a guideline, a `filter_scale` of 3.0 preserves detail well.
/// A scale of 1.0 preserves is good if speed is necessary, but still preserves a decent amount of detail.
/// Anything below is even faster, although the loss of detail becomes clear.
pub fn downsample_with_custom_scale(
    src: &Image<'_, AlbedoFormat>,
    target_width: u32,
    target_height: u32,
    filter_scale: f32,
) -> Vec<u8> {
    resample(src, target_width, target_height, filter_scale, false)
}

/// Version of [downsample_with_custom_scale] which weights each texel's colour by its alpha.
///
/// Use this for cut-out textures (foliage, fences, ...): without it the colour stored under fully transparent texels,
/// usually black, bleeds into the visible texels at coarser mip levels.
/// Texels with no visible texels under the filter fall back to the unweighted colour.
/// Pairs with [scale_alpha_to_original_coverage], which only rescales alpha and leaves the colour as is.
///
/// Panics if `src` has no alpha channel.
pub fn downsample_with_alpha_weighting(
    src: &Image<'_, AlbedoFormat>,
    target_width: u32,
    target_height: u32,
    filter_scale: f32,
) -> Vec<u8> {
    assert_eq!(
        src.format.num_channel_in_memory(),
        4,
        "Cannot weight by alpha on image with no alpha channel"
    );
    resample(src, target_width, target_height, filter_scale, true)
}

fn resample(
    src: &Image<'_, AlbedoFormat>,
    target_width: u32,
    target_height: u32,
    filter_scale: f32,
    alpha_weighted: bool,
) -> Vec<u8> {
    // The kernel writes pixels `pixel_stride_in_bytes` apart, so the output must be sized by stride.
    let mut output = vec![0u8; (target_width * target_height) as usize * src.pixel_stride_in_bytes];
    let stride = src.pixel_stride_in_bytes;
    resample_into(
        src,
        target_width,
        target_height,
        filter_scale,
        alpha_weighted,
        &mut output,
        stride,
    );
    output
}

/// Writes `target_width * target_height` pixels `output_stride` bytes apart, without gaps between rows. Bytes
/// past the format's channels are set to 0.
fn resample_into(
    src: &Image<'_, AlbedoFormat>,
    target_width: u32,
    target_height: u32,
    filter_scale: f32,
    alpha_weighted: bool,
    output: &mut [u8],
    output_stride: usize,
) {
    assert!(src.format.pixel_size_in_bytes() <= src.pixel_stride_in_bytes, "The stride between the pixels cannot be lower than the minimum size of the pixel according to the pixel format.");

    let sample_weights = precompute_lanczos_weights(
        src.width,
        src.height,
        target_width,
        target_height,
        filter_scale,
    );

    let channels = src.format.num_channel_in_memory();
    let mut src_image = ispc::SourceImage {
        width: src.width,
        height: src.height,
        data: src.pixels.as_ptr(),
        pixel_stride: src.pixel_stride_in_bytes as u32,
    };

    // sRGB is filtered in linear space; the kernels decode it on load and encode it on the final write.
    let srgb = matches!(src.format, AlbedoFormat::Srgb8 | AlbedoFormat::Srgba8);

    // The kernels read rows without gaps, with up to 8 bytes per pixel (anything past the format's channels is
    // padding). Alpha weighting needs exactly RGBA.
    let packed;
    let stride = src.pixel_stride_in_bytes;
    if (alpha_weighted && stride != channels) || stride > 8 {
        packed = pack_pixels(
            src.pixels,
            src.width,
            src.height,
            channels,
            src.pixel_stride_in_bytes,
        );
        src_image.data = packed.as_ptr();
        src_image.pixel_stride = channels as u32;
    }

    assert!(output.len() >= (target_width * target_height) as usize * output_stride);
    assert!(output_stride >= channels && output_stride <= 8);

    let kernel = if src.format.num_channel_in_memory() == 3 {
        ispc::downsample_ispc::resample_with_cached_weights_3
    } else if alpha_weighted {
        ispc::downsample_ispc::resample_with_cached_weights_4_alpha_weighted
    } else {
        ispc::downsample_ispc::resample_with_cached_weights_4
    };

    unsafe {
        kernel(
            &src_image,
            &mut ispc::DownsampledImage {
                width: target_width,
                height: target_height,
                data: output.as_mut_ptr(),
                pixel_stride: output_stride as u32,
            },
            ispc::PixelFormat::from(src.format),
            &mut ispc::DownsamplingContext {
                weights: *sample_weights.ispc_representation(),
                srgb_encode: if srgb {
                    srgb_encode_table().as_ptr()
                } else {
                    std::ptr::null()
                },
            },
        );
    }
}

/// Linear values quantized to 16 bits to 8-bit sRGB (IEC 61966-2-1), for the kernels' final write.
fn srgb_encode_table() -> &'static [u8; 65536] {
    static TABLE: std::sync::OnceLock<Box<[u8; 65536]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = Box::new([0u8; 65536]);
        for (i, v) in table.iter_mut().enumerate() {
            let linear = i as f64 / 65535.0;
            let srgb = if linear <= 0.0031308 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            *v = (srgb * 255.0).round() as u8;
        }
        table
    })
}

/// Copies `pixels` with `pixel_stride` bytes per pixel into a tightly packed buffer of `channels` bytes per pixel.
fn pack_pixels(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: usize,
    pixel_stride: usize,
) -> Vec<u8> {
    let count = (width * height) as usize;
    let mut packed = vec![0u8; count * channels];
    for (out, pixel) in packed
        .chunks_exact_mut(channels)
        .zip(pixels.chunks(pixel_stride))
    {
        out.copy_from_slice(&pixel[..channels]);
    }
    packed
}

/// Normalized box filter weights: every target texel averages the source texels it covers,
/// `[x * ratio, (x + 1) * ratio)`, and at least one.
fn box_weights(src: u32, target: u32) -> Vec<CachedWeight> {
    let ratio = src as f32 / target as f32;
    (0..target)
        .map(|x| {
            let start = ((x as f32 * ratio).floor() as u32).min(src - 1);
            let end = (((x + 1) as f32 * ratio).ceil() as u32)
                .max(start + 1)
                .min(src);
            let count = end - start;
            CachedWeight {
                start,
                coefficients: Rc::new(vec![1.0 / count as f32; count as usize]),
            }
        })
        .collect()
}

/// Downsamples an image that is meant to be used as a normal map.
/// Uses a box filter instead of a lanczos filter, and normalizes each pixel to preserve unit length for the normals after downsampling.
///
/// Returns a `Vec` with the downsampled data. If `normal_map_format.pixel_size() < pixel_stride_in_bytes`, the `Vec` will contain more values than channels than the format has specified, with all pixels in them initialized to 255.
///
/// See [downsample_normal_map_into] to write into an existing buffer instead.
pub fn downsample_normal_map(
    src: &Image<'_, NormalMapFormat>,
    target_width: u32,
    target_height: u32,
) -> Vec<u8> {
    let mut data = vec![0u8; (target_width * target_height) as usize * src.pixel_stride_in_bytes];
    downsample_normal_map_into(src, target_width, target_height, &mut data);
    data
}

/// Version of [downsample_normal_map] that writes into `output` rather than allocating: `target_width *
/// target_height` pixels of `pixel_stride_in_bytes` each, the same layout as the source. Padding bytes are set
/// to 255.
///
/// Reusing one buffer for many calls (like every level of a mip chain) avoids allocating fresh memory every time,
/// which for large outputs costs about as much as downsampling itself.
pub fn downsample_normal_map_into(
    src: &Image<'_, NormalMapFormat>,
    target_width: u32,
    target_height: u32,
    output: &mut [u8],
) {
    assert!(src.format.pixel_size_in_bytes() <= src.pixel_stride_in_bytes, "The pixel stride in bytes must be more or equal than the size of a single pixel as described by the format of the normal map.");
    let output_size = (target_width * target_height) as usize * src.pixel_stride_in_bytes;
    assert!(
        output.len() >= output_size,
        "The output needs {output_size} bytes for {target_width}x{target_height} pixels"
    );
    let output = &mut output[..output_size];

    // Padding bytes the kernels do not write: they store 32 bits per pixel, with 255 in the unused bytes.
    let stride = src.pixel_stride_in_bytes;
    if stride > 4 || (stride == 3 && src.format == NormalMapFormat::Rg8TangentSpaceReconstructedZ) {
        output.fill(255);
    }

    // Rgb8 is read with its pixel stride (up to 8 bytes, the rest is padding); Rg8 needs to be packed.
    let channels = src.format.num_channel_in_memory();
    let stride = src.pixel_stride_in_bytes;
    let packed;
    let (pixels, stride) = if (src.format == NormalMapFormat::Rg8TangentSpaceReconstructedZ
        && stride != channels)
        || stride > 8
    {
        packed = pack_pixels(src.pixels, src.width, src.height, channels, stride);
        (&packed[..], channels)
    } else {
        (src.pixels, stride)
    };

    // Exact 2:1 of these layouts (every mip level of a power of two texture) has a dedicated kernel that does not
    // use the weights, so skip building them.
    let exact_2x2 = src.width == 2 * target_width
        && src.height == 2 * target_height
        && matches!(
            (src.format, stride),
            (NormalMapFormat::Rgb8, 3)
                | (NormalMapFormat::Rgb8, 4)
                | (NormalMapFormat::Rg8TangentSpaceReconstructedZ, 2)
        )
        && stride == src.pixel_stride_in_bytes;
    let weights = (!exact_2x2).then(|| {
        let width_weights = WeightCollection::new(box_weights(src.width, target_width));
        let height_weights = if src.width == src.height && target_width == target_height {
            width_weights.clone()
        } else {
            WeightCollection::new(box_weights(src.height, target_height))
        };
        ispc::Weights::new(width_weights, height_weights)
    });

    unsafe {
        ispc::downsample_normal_map(
            &ispc::SourceImage {
                width: src.width,
                height: src.height,
                data: pixels.as_ptr(),
                pixel_stride: stride as u32,
            },
            &mut ispc::DownsampledImage {
                width: target_width,
                height: target_height,
                data: output.as_mut_ptr(),
                pixel_stride: src.pixel_stride_in_bytes as u32,
            },
            ispc::NormalMapFormat::from(src.format),
            &mut ispc::DownsamplingContext {
                weights: weights.as_ref().map_or(
                    ispc::SampleWeights {
                        vertical_weights: std::ptr::null(),
                        horizontal_weights: std::ptr::null(),
                    },
                    |weights| *weights.ispc_representation(),
                ),
                srgb_encode: std::ptr::null(),
            },
        );
    }
}

/// Row pitch alignment of texture uploads, `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`. Vulkan accepts the same layout.
pub const MIP_ROW_PITCH_ALIGNMENT: usize = 256;
/// Alignment of every mip level's offset, `D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT`.
pub const MIP_PLACEMENT_ALIGNMENT: usize = 512;

/// Where one mip level is stored in a buffer: `height` rows of `width` texels, `row_pitch` bytes apart, starting at
/// `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipLevel {
    pub offset: usize,
    pub row_pitch: usize,
    pub width: u32,
    pub height: u32,
}

/// The full mip chain of a `width` x `height` texture, down to 1x1, laid out for upload like D3D12's
/// `GetCopyableFootprints()`: rows aligned to [MIP_ROW_PITCH_ALIGNMENT], levels to [MIP_PLACEMENT_ALIGNMENT].
/// Returns the levels and the total size of the buffer. Take a prefix of the levels for a shorter chain.
pub fn mip_layout(width: u32, height: u32, bytes_per_texel: usize) -> (Vec<MipLevel>, usize) {
    let align = |value: usize, alignment: usize| value.div_ceil(alignment) * alignment;
    let count = 32 - width.max(height).max(1).leading_zeros();
    let mut size = 0;
    let levels = (0..count)
        .map(|i| {
            let (width, height) = ((width >> i).max(1), (height >> i).max(1));
            let row_pitch = align(width as usize * bytes_per_texel, MIP_ROW_PITCH_ALIGNMENT);
            let offset = align(size, MIP_PLACEMENT_ALIGNMENT);
            size = offset + row_pitch * (height as usize - 1) + width as usize * bytes_per_texel;
            MipLevel {
                offset,
                row_pitch,
                width,
                height,
            }
        })
        .collect();
    (levels, size)
}

/// How [generate_mips_into] keeps alpha coverage (see [scale_alpha_to_original_coverage]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaCoverage {
    /// Alpha is filtered like the colour.
    Unchanged,
    /// Every level keeps the mean alpha of the source.
    Mean,
    /// Every level keeps the fraction of the source that is above this alpha cutoff.
    Cutoff(f32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MipOptions {
    /// See [downsample_with_custom_scale].
    pub filter_scale: f32,
    /// See [downsample_with_alpha_weighting]. Needs an alpha channel.
    pub alpha_weighting: bool,
    /// Needs an alpha channel.
    pub alpha_coverage: AlphaCoverage,
}

impl Default for MipOptions {
    fn default() -> Self {
        Self {
            filter_scale: 3.0,
            alpha_weighting: false,
            alpha_coverage: AlphaCoverage::Unchanged,
        }
    }
}

/// Fills a mip chain: copies `src` into `levels[0]` as `texel`-byte texels (`pad` in the bytes past its channels)
/// and calls `step(previous, level, output)` for every next level, with `previous` the level before as a tightly
/// packed image and `output` the tightly packed pixels of `level`. Tightly packed levels are read and written in
/// place; padded ones go through scratch memory.
fn fill_mip_chain<F: ImagePixelFormat>(
    src: &Image<'_, F>,
    levels: &[MipLevel],
    output: &mut [u8],
    texel: usize,
    pad: u8,
    mut step: impl FnMut(&Image<'_, F>, &MipLevel, &mut [u8]),
) {
    let channels = src.format.num_channel_in_memory();
    assert!(src.format.pixel_size_in_bytes() <= src.pixel_stride_in_bytes);
    assert!(
        levels
            .first()
            .is_some_and(|l| l.width == src.width && l.height == src.height),
        "The first mip level must have the size of the source"
    );
    let size = |l: &MipLevel| (l.width * l.height) as usize * texel;
    let mut end = 0;
    for level in levels {
        assert!(
            level.offset >= end,
            "Mip levels must be in order and not overlap"
        );
        assert!(level.row_pitch >= level.width as usize * texel);
        end = level.offset
            + level.row_pitch * (level.height as usize - 1)
            + level.width as usize * texel;
    }
    assert!(
        output.len() >= end,
        "The output needs {end} bytes for these mip levels"
    );
    let tight = |l: &MipLevel| l.row_pitch == l.width as usize * texel;

    let first = &levels[0];
    let row = src.width as usize * texel;
    let src_row = src.width as usize * src.pixel_stride_in_bytes;
    for y in 0..src.height as usize {
        let src_row = &src.pixels[y * src_row..][..src_row];
        let dst_row = &mut output[first.offset + y * first.row_pitch..][..row];
        if src.pixel_stride_in_bytes == texel && channels == texel {
            dst_row.copy_from_slice(src_row);
        } else if channels == 3 && texel == 4 {
            // Fixed-size chunks, which the compiler vectorises.
            let pixels = src_row.chunks_exact(src.pixel_stride_in_bytes);
            for (dst, pixel) in dst_row.chunks_exact_mut(4).zip(pixels) {
                dst.copy_from_slice(&[pixel[0], pixel[1], pixel[2], pad]);
            }
        } else {
            for (dst, pixel) in dst_row
                .chunks_exact_mut(texel)
                .zip(src_row.chunks(src.pixel_stride_in_bytes))
            {
                dst[..channels].copy_from_slice(&pixel[..channels]);
                dst[channels..].fill(pad);
            }
        }
    }

    let largest = levels.iter().map(size).max().unwrap_or(0);
    let mut scratch = [vec![0u8; largest], vec![0u8; largest]];
    let mut previous_in_scratch = !tight(first);
    if previous_in_scratch {
        for y in 0..first.height as usize {
            scratch[0][y * row..][..row]
                .copy_from_slice(&output[first.offset + y * first.row_pitch..][..row]);
        }
    }

    for pair in levels.windows(2) {
        let (previous, level) = (&pair[0], &pair[1]);
        let (head, tail) = output.split_at_mut(level.offset);
        let [scratch_previous, scratch_current] = &mut scratch;
        let previous_pixels: &[u8] = if previous_in_scratch {
            &scratch_previous[..size(previous)]
        } else {
            &head[previous.offset..][..size(previous)]
        };
        let image = Image::new_with_pixel_stride(
            previous_pixels,
            previous.width,
            previous.height,
            src.format,
            texel,
        );
        if tight(level) {
            step(&image, level, &mut tail[..size(level)]);
            previous_in_scratch = false;
        } else {
            let current = &mut scratch_current[..size(level)];
            step(&image, level, current);
            let row = level.width as usize * texel;
            for (y, src) in current.chunks_exact(row).enumerate() {
                tail[y * level.row_pitch..][..row].copy_from_slice(src);
            }
            scratch.swap(0, 1);
            previous_in_scratch = true;
        }
    }
}

/// Generates every level of a mip chain in one buffer laid out like [mip_layout]: `levels[0]` gets a copy of
/// `src`, and every next level is downsampled from the one before. Texels are 4 bytes, the layout of
/// `DXGI_FORMAT_R8G8B8A8_*` and `VK_FORMAT_R8G8B8A8_*`, with alpha 255 for formats without alpha.
///
/// Faster than downsampling level by level: it writes straight into the (upload) buffer rather than allocating
/// every level, and measures the alpha coverage of the source only once.
pub fn generate_mips_into(
    src: &Image<'_, AlbedoFormat>,
    levels: &[MipLevel],
    output: &mut [u8],
    options: &MipOptions,
) {
    let has_alpha = src.format.num_channel_in_memory() == 4;
    assert!(
        has_alpha
            || (!options.alpha_weighting && options.alpha_coverage == AlphaCoverage::Unchanged),
        "Alpha weighting and coverage need an alpha channel"
    );
    let rgba_format = match src.format {
        AlbedoFormat::Rgb8Unorm => AlbedoFormat::Rgba8Unorm,
        AlbedoFormat::Rgb8Snorm => AlbedoFormat::Rgba8Snorm,
        AlbedoFormat::Srgb8 => AlbedoFormat::Srgba8,
        format => format,
    };
    let cutoff = match options.alpha_coverage {
        AlphaCoverage::Cutoff(cutoff) => Some(cutoff),
        _ => None,
    };
    let cutoff = cutoff
        .as_ref()
        .map_or(std::ptr::null(), |c| c as *const f32);
    // Coverage of the source, which every level keeps; measured on the first step.
    let mut target_coverage = None;

    fill_mip_chain(src, levels, output, 4, 255, |previous, level, output| {
        if options.alpha_coverage != AlphaCoverage::Unchanged && target_coverage.is_none() {
            target_coverage = Some(unsafe {
                ispc::downsample_ispc::alpha_coverage(
                    previous.width,
                    previous.height,
                    previous.pixels.as_ptr(),
                    cutoff,
                )
            });
        }
        // Levels after the first are RGBA with opaque alpha, which the 4-channel kernels (faster than 3 channels
        // with a pixel stride) keep at exactly 255.
        let previous = &Image::new(
            previous.pixels,
            previous.width,
            previous.height,
            rgba_format,
        );
        resample_into(
            previous,
            level.width,
            level.height,
            options.filter_scale,
            options.alpha_weighting,
            output,
            4,
        );
        if let Some(target) = target_coverage {
            unsafe {
                ispc::downsample_ispc::scale_to_target_alpha_coverage(
                    level.width,
                    level.height,
                    output.as_mut_ptr(),
                    cutoff,
                    target,
                );
            }
        }
    });
}

/// Version of [generate_mips_into] for normal maps, see [downsample_normal_map]. Texels are 4 bytes for
/// [NormalMapFormat::Rgb8] (`R8G8B8A8`, with 255 in the 4th byte) and 2 bytes for
/// [NormalMapFormat::Rg8TangentSpaceReconstructedZ] (`R8G8`).
pub fn generate_normal_mips_into(
    src: &Image<'_, NormalMapFormat>,
    levels: &[MipLevel],
    output: &mut [u8],
) {
    let texel = if src.format == NormalMapFormat::Rgb8 {
        4
    } else {
        2
    };
    fill_mip_chain(
        src,
        levels,
        output,
        texel,
        255,
        |previous, level, output| {
            downsample_normal_map_into(previous, level.width, level.height, output)
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mip chains are built by feeding each mip back in, so any per-call bias accumulates.
    #[test]
    fn mip_chain_preserves_constant_color() {
        for value in 0..=255u8 {
            let mut pixels = vec![value; 64 * 64 * 4];
            let mut size = 64;
            while size > 1 {
                let src = Image::new(&pixels, size, size, AlbedoFormat::Rgba8Unorm);
                size /= 2;
                pixels = downsample(&src, size, size);
                let bad = pixels.iter().find(|&&p| p != value);
                assert!(
                    bad.is_none(),
                    "value {value} drifted to {bad:?} at {size}x{size}"
                );
            }
        }
    }

    #[test]
    fn lanczos_weights_are_normalized_and_in_bounds() {
        for (src, target) in [
            (64, 32),
            (64, 1),
            (64, 63),
            (37, 12),
            (3, 1),
            (2, 1),
            (1000, 333),
        ] {
            for filter_scale in [0.25, 1.0, 3.0, 8.0] {
                let weights = calculate_weights(src, target, filter_scale);
                assert_eq!(weights.len(), target as usize);
                for (i, w) in weights.iter().enumerate() {
                    let sum: f32 = w.coefficients.iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-3,
                        "{src}->{target} scale {filter_scale} row {i}: sum {sum}"
                    );
                    assert!(
                        w.start as usize + w.coefficients.len() <= src as usize,
                        "{src}->{target} scale {filter_scale} row {i}: reads past the source"
                    );
                }
            }
        }
    }

    #[test]
    fn downsample_with_pixel_stride() {
        // RGB stored as RGBX: the padding byte must be skipped on read and left alone on write.
        let pixels = [10u8, 20, 30, 99].repeat(8 * 8);
        let src = Image::new_with_pixel_stride(&pixels, 8, 8, AlbedoFormat::Rgb8Unorm, 4);
        assert_eq!(downsample(&src, 4, 4), [10u8, 20, 30, 0].repeat(4 * 4));
    }

    // A leaf on a cut-away background: transparent texels are black and must not darken the leaf.
    #[test]
    fn alpha_weighting_stops_transparent_bleed() {
        const LEAF: [u8; 3] = [40, 200, 40];
        let size = 64;
        let mut pixels = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let leaf = (x / 4 + y / 4) % 2 == 0 || (x * y) % 7 == 0;
                pixels.extend(if leaf { [40, 200, 40, 255] } else { [0; 4] });
            }
        }

        let mut size = size;
        while size > 1 {
            let src = Image::new(&pixels, size, size, AlbedoFormat::Rgba8Unorm);
            size /= 2;
            let downsampled = downsample_with_alpha_weighting(&src, size, size, 3.0);
            let dst = Image::new(&downsampled, size, size, AlbedoFormat::Rgba8Unorm);
            pixels = scale_alpha_to_original_coverage(&src, &dst, Some(0.5));

            for p in pixels.chunks(4).filter(|p| p[3] > 0) {
                assert!(
                    p[..3].iter().zip(LEAF).all(|(&c, l)| c.abs_diff(l) <= 1),
                    "leaf bled to {p:?} at {size}x{size}"
                );
            }
        }
    }

    #[test]
    fn mip_chain_preserves_flat_normal() {
        for (format, flat) in [
            (NormalMapFormat::Rgb8, &[128u8, 128, 255][..]),
            (
                NormalMapFormat::Rg8TangentSpaceReconstructedZ,
                &[128u8, 128][..],
            ),
        ] {
            let mut pixels = flat.repeat(64 * 64);
            let mut size = 64;
            while size > 1 {
                let src = Image::new(&pixels, size, size, format);
                size /= 2;
                pixels = downsample_normal_map(&src, size, size);
                assert!(
                    pixels.chunks(flat.len()).all(|p| p == flat),
                    "{format:?} normal tilted at {size}x{size}: {:?}",
                    &pixels[..flat.len()]
                );
            }
        }
    }

    /// A flat-colored image must come out of the linearize-filter-encode roundtrip unchanged
    #[test]
    fn flat_srgb_image_is_preserved() {
        for format in [AlbedoFormat::Srgb8, AlbedoFormat::Srgba8] {
            let num_channels = format.num_channel_in_memory();
            for value in 0..=255u8 {
                let pixels = vec![value; 64 * 64 * num_channels];
                let src = Image::new(&pixels, 64, 64, format);
                let downsampled = downsample(&src, 16, 16);
                assert!(
                    downsampled.iter().all(|&v| v == value),
                    "{format:?} value {value} changed to {:?}",
                    downsampled.iter().find(|&&v| v != value)
                );
            }
        }
    }

    /// Fine black and white line patterns, like the test image from
    /// <http://www.ericbrasseur.org/gamma.html> linked in #25, carry 50% linear light: that is 188
    /// in sRGB, not the 128 that filtering the encoded values produces.
    #[test]
    fn srgb_line_pattern_averages_in_linear_light() {
        let pixels = (0..64)
            .flat_map(|y| vec![if y % 2 == 0 { 0 } else { 255 }; 64 * 3])
            .collect::<Vec<u8>>();
        for (format, expected) in [(AlbedoFormat::Srgb8, 188), (AlbedoFormat::Rgb8Unorm, 128)] {
            let downsampled = downsample(&Image::new(&pixels, 64, 64, format), 16, 16);
            // The filter is cut off at the borders, where the pattern doesn't average out
            for y in 2..14 {
                let row = &downsampled[y * 16 * 3..][..16 * 3];
                assert!(
                    row.iter().all(|&v| v.abs_diff(expected) <= 1),
                    "{format:?} row {y} should average to {expected}, got {row:?}"
                );
            }
        }
    }

    #[test]
    fn srgb_downsample_with_pixel_stride() {
        // sRGB stored as sRGBX: the padding byte must be skipped on read and left alone on write.
        let pixels = [10u8, 20, 30, 99].repeat(8 * 8);
        let src = Image::new_with_pixel_stride(&pixels, 8, 8, AlbedoFormat::Srgb8, 4);
        assert_eq!(downsample(&src, 4, 4), [10u8, 20, 30, 0].repeat(4 * 4));
    }

    /// Same as `alpha_weighting_stops_transparent_bleed()`, for an sRGB-encoded leaf
    #[test]
    fn srgb_alpha_weighting_stops_transparent_bleed() {
        const LEAF: [u8; 3] = [40, 200, 40];
        let size = 64;
        let mut pixels = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let leaf = (x / 4 + y / 4) % 2 == 0 || (x * y) % 7 == 0;
                pixels.extend(if leaf { [40, 200, 40, 255] } else { [0; 4] });
            }
        }

        let mut size = size;
        while size > 1 {
            let src = Image::new(&pixels, size, size, AlbedoFormat::Srgba8);
            size /= 2;
            let downsampled = downsample_with_alpha_weighting(&src, size, size, 3.0);
            let dst = Image::new(&downsampled, size, size, AlbedoFormat::Srgba8);
            pixels = scale_alpha_to_original_coverage(&src, &dst, Some(0.5));

            for p in pixels.chunks(4).filter(|p| p[3] > 0) {
                assert!(
                    p[..3].iter().zip(LEAF).all(|(&c, l)| c.abs_diff(l) <= 1),
                    "leaf bled to {p:?} at {size}x{size}"
                );
            }
        }
    }
}
