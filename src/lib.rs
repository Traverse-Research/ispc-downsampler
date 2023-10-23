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

/// Describes a source image which can be used for [`downsample()`]
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
) -> Vec<u8> {
    assert!(
        matches!(src.format, Format::RGBA8),
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

/// Runs the ISPC kernel on the source image, sampling it down to the `target_width` and `target_height`. Returns the downsampled pixel data as a `Vec<u8>`.
///
/// Will panic if the target width or height are higher than that of the source image.
pub fn downsample(src: &Image<'_>, target_width: u32, target_height: u32) -> Vec<u8> {
    assert!(src.width >= target_width, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");
    assert!(src.height >= target_height, "The width of the source image is less than the target's width. You are trying to upsample rather than downsample");

    let num_channels = src.format.num_channels();
    let mut output = vec![0; (target_width * target_height * num_channels as u32) as usize];

    unsafe {
        ispc::downsample_ispc::resample(
            src.width,
            src.height,
            src.width,
            num_channels,
            target_width,
            target_height,
            src.pixels.as_ptr(),
            output.as_mut_ptr(),
        )
    }

    output
}
