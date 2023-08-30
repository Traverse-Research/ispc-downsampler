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

/// Describes a source image which can be used for `downsample`
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

    fn get_alpha(&self, x: usize, y: usize) -> f32 {
        let pixel_index = x + y * self.width as usize;
        self.pixels[pixel_index * 4 + 3] as f32 / 255.0
    }

    fn calculate_alpha_coverage(&self, alpha_cutoff: Option<f32>) -> f32 {
        self.calculate_scaled_alpha_coverage(alpha_cutoff, 1.0f32)
    }

    /// Computes the percentage of texels that are visible in an alpha masked texture using
    /// 4x4 subsampling.
    /// Ported version of implementation in https://github.com/castano/nvidia-texture-tools/.
    fn calculate_scaled_alpha_coverage(&self, alpha_cutoff: Option<f32>, scale: f32) -> f32 {
        // Edge-case for if the texture is downsampled to 1x1
        if self.width == 1 && self.height == 1 {
            return 0f32;
        }
        let mut coverage = 0.0f32;
        let subsample_factor = 4;

        for y in 0..(self.height as usize) - 1 {
            for x in 0..(self.width as usize) - 1 {
                let top_left = (self.get_alpha(x, y) * scale).min(1.0);
                let top_right = (self.get_alpha(x + 1, y) * scale).min(1.0);
                let bottom_left = (self.get_alpha(x, y + 1) * scale).min(1.0);
                let bottom_right = (self.get_alpha(x + 1, y + 1) * scale).min(1.0);

                let mut texel_coverage = 0.0;
                for sy in 0..subsample_factor {
                    let fy = (sy as f32 + 0.5) / subsample_factor as f32;
                    for sx in 0..subsample_factor {
                        let fx = (sx as f32 + 0.5) / subsample_factor as f32;
                        let alpha = top_left * (1.0 - fx) * (1.0 - fy)
                            + top_right * fx * (1.0 - fy)
                            + bottom_left * (1.0 - fx) * fy
                            + bottom_right * fx * fy;
                        if let Some(alpha_cutoff) = alpha_cutoff {
                            if alpha > alpha_cutoff {
                                texel_coverage += 1.0;
                            }
                        } else {
                            texel_coverage += alpha
                        }
                    }
                }
                coverage += texel_coverage / (subsample_factor * subsample_factor) as f32;
            }
        }

        coverage / ((self.width - 1) * (self.height - 1)) as f32
    }

    /// Computes the scaling factor needed for the texture's alpha such that the desired
    /// coverage is best approximated
    /// Ported version of implementation in https://github.com/castano/nvidia-texture-tools/.
    fn find_alpha_scale_for_coverage(
        &self,
        desired_coverage: f32,
        alpha_cutoff: Option<f32>,
    ) -> f32 {
        // This range of potential scale values is an estimate. Especially the upper bound
        // may not be sufficient for some use-cases, depending on the downscale
        // compared to mip 0
        // TO-DO: Figure out if this should be exposed
        let mut alpha_scale_range_start = 0.0;
        let mut alpha_scale_range_end = 8.0;

        let mut alpha_scale = 1.0;

        // The search is done heuristically, so a tested alpha scale value
        // does not directly map to the resulting alpha coverage. To ensure,
        // we do not overwrite the best result, store
        // the best result so far and use that instead of the last found scale
        let mut best_abs_diff = f32::INFINITY;
        let mut best_alpha_scale = alpha_scale;

        // 10-step binary search for the alpha multiplier that best matches
        // the desired alpha coverage
        for _ in 0..10 {
            let current_coverage = self.calculate_scaled_alpha_coverage(alpha_cutoff, alpha_scale);
            let coverage_diff = current_coverage - desired_coverage;

            if coverage_diff.abs() < best_abs_diff {
                best_abs_diff = coverage_diff.abs();
                best_alpha_scale = alpha_scale;
            }

            if current_coverage < desired_coverage {
                alpha_scale_range_start = alpha_scale;
            } else if current_coverage > desired_coverage {
                alpha_scale_range_end = alpha_scale;
            } else {
                break;
            }
            alpha_scale = (alpha_scale_range_start + alpha_scale_range_end) / 2.0
        }
        best_alpha_scale
    }
}

fn apply_alpha_scale(data: &mut [u8], alpha_scale: f32) {
    for pixel in data.iter_mut().skip(3).step_by(4) {
        *pixel = (*pixel as f32 * alpha_scale).min(255.0) as u8
    }
}

#[derive(Default)]
pub enum AlphaCoverageSetting {
    #[default]
    None,

    /// Scales the alpha to the downscaled texture to preserve the overall alpha coverage.
    /// If alpha cutoff is specified, any alpha value above it is considered visible
    /// of which the percentage of visible texels will be.
    /// Otherwise, visibility is considered a linear sum of the alpha values instead
    /// and the source and target alpha coverage are calculated the same way
    RetainAlphaCoverage { alpha_cutoff: Option<f32> },
}

/// Runs the ISPC kernel on the source image, sampling it down to the `target_width` and `target_height`. Returns the downsampled pixel data as a `Vec<u8>`.
///
/// Will panic if the target width or height are higher than that of the source image.
pub fn downsample(
    src: &Image<'_>,
    target_width: u32,
    target_height: u32,
    target_desired_alpha_coverage: AlphaCoverageSetting,
) -> Vec<u8> {
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

    if let AlphaCoverageSetting::RetainAlphaCoverage { alpha_cutoff } =
        target_desired_alpha_coverage
    {
        assert!(
            matches!(src.format, Format::RGBA8),
            "Cannot retain alpha coverage on image with no alpha channel"
        );
        let scale = Image::new(&output, target_width, target_height, src.format)
            .find_alpha_scale_for_coverage(
                src.calculate_alpha_coverage(alpha_cutoff),
                alpha_cutoff,
            );
        apply_alpha_scale(&mut output, scale);
    }

    output
}
