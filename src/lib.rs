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
        self.pixels[x * 4 + 3 + y * 4 * self.width as usize] as f32 / 255.0
    }

    /// Computes the percentage of texels that are visible in an alpha masked texture using
    /// 4x4 subsampling.
    /// Ported version of implementation in https://github.com/castano/nvidia-texture-tools/.
    pub fn calculate_alpha_coverage(&self, alpha_cutoff: f32) -> f32 {
        self.calculate_scaled_alpha_coverage(alpha_cutoff, 1.0f32)
    }

    fn calculate_scaled_alpha_coverage(&self, alpha_cutoff: f32, scale: f32) -> f32 {
        let mut coverage = 0.0f32;
        let subsample_factor = 4;

        for y in 0..(self.height as usize) - 1 {
            for x in 0..(self.width as usize) - 1 {
                let top_left = (self.get_alpha(x, y) * scale).min(255.0);
                let top_right = (self.get_alpha(x + 1, y) * scale).min(255.0);
                let bottom_left = (self.get_alpha(x, y + 1) * scale).min(255.0);
                let bottom_right = (self.get_alpha(x + 1, y + 1) * scale).min(255.0);

                let mut texel_coverage = 0.0;
                for sy in 0..subsample_factor {
                    let fy = (sy as f32 + 0.5) / subsample_factor as f32;
                    for sx in 0..subsample_factor {
                        let fx = (sx as f32 + 0.5) / subsample_factor as f32;
                        let alpha = top_left * (1.0 - fx) * (1.0 - fy)
                            + top_right * fx * (1.0 - fy)
                            + bottom_left * (1.0 - fx) * fy
                            + bottom_right * fx * fy;
                        if alpha > alpha_cutoff {
                            texel_coverage += 1.0;
                        }
                    }
                }
                coverage += texel_coverage / (subsample_factor * subsample_factor) as f32;
            }
        }

        coverage / ((self.width - 1) * (self.height - 1)) as f32
    }

    pub fn find_alpha_scale_for_coverage(&self, desired_coverage: f32, alpha_cutoff: f32) -> f32 {
        let mut alpha_scale_range_start = 0.0;

        // This value might need exposure for tweaking
        let mut alpha_scale_range_end = 4.0;
        let mut alpha_scale = 1.0;

        // Due to the subsampling when determining the alpha coverage of an image, we can technically
        // overshoot the used alpha scale. It's therefore necessary to keep track of what was the
        // best result so far and use that instead of the last found scale
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

            if coverage_diff < 0.0 {
                alpha_scale_range_start = alpha_scale;
            } else if coverage_diff > 0.0 {
                alpha_scale_range_end = alpha_scale;
            } else {
                break;
            }
            alpha_scale = (alpha_scale_range_start + alpha_scale_range_end) / 2.0
        }
        best_alpha_scale
    }
}

pub fn apply_alpha_scale(data: &mut [u8], alpha_scale: f32) {
    for pixel in data.iter_mut() {
        *pixel = (*pixel as f32 / alpha_scale).round() as u8
    }
}

pub enum AlphaCoverageSetting {
    None,
    RetainAlphaCoverage {
        target_alpha_coverage: f32,
        alpha_cutoff: f32
    }
}

/// Runs the ISPC kernel on the source image, sampling it down to the `target_width` and `target_height`. Returns the downsampled pixel data as a `Vec<u8>`.
///
/// Will panic if the target width or height are higher than that of the source image.
pub fn downsample(
    src: &Image,
    target_width: u32,
    target_height: u32,
    target_desired_alpha_coverage: AlphaCoverageSetting
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

    if let AlphaCoverageSetting::RetainAlphaCoverage { target_alpha_coverage, alpha_cutoff } = target_desired_alpha_coverage {
        let scale = Image::new(&output, target_width, target_height, src.format).find_alpha_scale_for_coverage(target_alpha_coverage, alpha_cutoff);
        apply_alpha_scale(&mut output, scale);
    }

    output
}
