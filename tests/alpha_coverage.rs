//! Checks `scale_alpha_to_original_coverage` against a reference implementation of the coverage model it
//! optimizes: alpha is sampled at 4x4 bilinear subsamples per texel quad, after scaling and clamping each texel.

use ispc_downsampler::{downsample, scale_alpha_to_original_coverage, AlbedoFormat, Image};

/// Reference coverage of 8-bit alpha scaled by `scale`, as the kernel defines it.
fn coverage(alpha: &[u8], w: usize, h: usize, scale: f64, cutoff: Option<f64>) -> f64 {
    let a = |x: usize, y: usize| (alpha[y * w + x] as f64 / 255.0 * scale).min(1.0);
    let visible = |v: f64| cutoff.map_or(v, |c| if v > c { 1.0 } else { 0.0 });
    if w == 1 || h == 1 {
        let n = (w * h) as f64;
        return (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| visible(a(x, y)))
            .sum::<f64>()
            / n;
    }
    let mut total = 0.0;
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            for sy in 0..4 {
                let fy = (sy as f64 + 0.5) / 4.0;
                for sx in 0..4 {
                    let fx = (sx as f64 + 0.5) / 4.0;
                    let v = a(x, y) * (1.0 - fx) * (1.0 - fy)
                        + a(x + 1, y) * fx * (1.0 - fy)
                        + a(x, y + 1) * (1.0 - fx) * fy
                        + a(x + 1, y + 1) * fx * fy;
                    total += visible(v) / 16.0;
                }
            }
        }
    }
    total / ((w - 1) * (h - 1)) as f64
}

fn alpha_of(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks(4).map(|p| p[3]).collect()
}

/// Soft-edged cut-out shapes with varying density, so the scale has to move a lot or a little.
fn cutout(w: u32, h: u32, density: f32, seed: f32) -> Vec<u8> {
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = ((x as f32 * 0.37 + seed).sin() * (y as f32 * 0.23 - seed).cos()
                + (x as f32 * 0.11 + y as f32 * 0.07 + seed).sin())
                * 0.5
                + density;
            data.extend([90, 140, 30, (v * 2.0 * 255.0).clamp(0.0, 255.0) as u8]);
        }
    }
    data
}

struct Case {
    name: &'static str,
    src: Vec<u8>,
    size: (u32, u32),
    target: (u32, u32),
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (name, density, seed) in [
        ("sparse", -0.35, 0.0),
        ("medium", 0.0, 1.3),
        ("dense", 0.3, 2.1),
    ] {
        for (size, target) in [
            ((64, 64), (32, 32)),
            ((128, 96), (32, 24)),
            ((64, 64), (8, 8)),
            ((32, 1), (8, 1)),
        ] {
            cases.push(Case {
                name,
                src: cutout(size.0, size.1, density, seed),
                size,
                target,
            });
        }
    }
    cases
}

/// Coverage achieved relative to the source (measured with the reference model on the quantized output),
/// and the error allowed for it: one coverage step at the target size, plus u8 rounding of the scaled alpha.
fn coverage_errors(cutoff: Option<f32>) -> Vec<(String, f64, f64)> {
    cases()
        .into_iter()
        .map(|c| {
            let src = Image::new(&c.src, c.size.0, c.size.1, AlbedoFormat::Rgba8Unorm);
            let down = downsample(&src, c.target.0, c.target.1);
            let scaled = scale_alpha_to_original_coverage(
                &src,
                &Image::new(&down, c.target.0, c.target.1, AlbedoFormat::Rgba8Unorm),
                cutoff,
            );
            let c64 = cutoff.map(|c| c as f64);
            let (tw, th) = (c.target.0 as usize, c.target.1 as usize);
            let want = coverage(
                &alpha_of(&c.src),
                c.size.0 as usize,
                c.size.1 as usize,
                1.0,
                c64,
            );
            let got = coverage(&alpha_of(&scaled), tw, th, 1.0, c64);
            let samples = if tw == 1 || th == 1 {
                tw * th
            } else {
                (tw - 1) * (th - 1) * 16
            };
            let tolerance = 1.0 / samples as f64 + 0.002;
            (
                format!("{} {:?}->{:?}", c.name, c.size, c.target),
                (got - want).abs(),
                tolerance,
            )
        })
        .collect()
}

#[test]
fn coverage_matches_source() {
    for cutoff in [Some(0.5), Some(0.2), Some(0.9), None] {
        for (name, err, tolerance) in coverage_errors(cutoff) {
            assert!(
                err <= tolerance,
                "cutoff {cutoff:?} {name}: off by {err:.5}, tolerance {tolerance:.5}"
            );
        }
    }
}

#[test]
fn full_and_empty_alpha_are_left_alone() {
    for alpha in [0u8, 255] {
        let data = [10u8, 20, 30, alpha].repeat(32 * 32);
        let src = Image::new(&data, 32, 32, AlbedoFormat::Rgba8Unorm);
        let down = downsample(&src, 16, 16);
        for cutoff in [Some(0.5), None] {
            let scaled = scale_alpha_to_original_coverage(
                &src,
                &Image::new(&down, 16, 16, AlbedoFormat::Rgba8Unorm),
                cutoff,
            );
            assert_eq!(scaled, down, "alpha {alpha} cutoff {cutoff:?}");
        }
    }
}
