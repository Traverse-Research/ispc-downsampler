#[cfg(feature = "ispc")]
fn compile_bindings() {
    use ispc_compile::{bindgen::builder, Config, MathLib, TargetISA};

    // Compile our ISPC library, this call will exit with EXIT_FAILURE if
    // compilation fails.

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_isas = match target_arch.as_str() {
        "x86" | "x86_64" => vec![
            TargetISA::SSE2i32x4,
            TargetISA::SSE4i32x4,
            TargetISA::AVX1i32x8,
            TargetISA::AVX2i32x8,
            TargetISA::AVX512KNLi32x16,
            TargetISA::AVX512SKXi32x16,
        ],
        "arm" | "aarch64" => vec![
            // TargetISA::Neoni32x4,
            TargetISA::Neoni32x8,
        ],
        x => panic!("Unsupported target architecture {}", x),
    };

    Config::new()
        .file("src/ispc/kernels/filters/lanczos.ispc")
        .file("src/ispc/kernels/rescale_alpha.ispc")
        .file("src/ispc/kernels/downsampling.ispc")
        .file("src/ispc/kernels/weight_dimensions.ispc")
        .opt_level(2)
        .woff()
        .target_isas(target_isas)
        .math_lib(MathLib::Fast)
        .bindgen_builder(
            builder()
                .allowlist_function("resample_with_cached_weights_3")
                .allowlist_function("resample_with_cached_weights_4")
                .allowlist_function("downsample_normal_map")
                .allowlist_function("calculate_weights_lanczos")
                .allowlist_function("calculate_weight_dimensions")
                .allowlist_function("scale_to_alpha_coverage"),
        )
        .out_dir("src/ispc")
        .compile("downsample_ispc");
}

#[cfg(not(feature = "ispc"))]
fn compile_bindings() {
    ispc_rt::PackagedModule::new("downsample_ispc")
        .lib_path("src/ispc")
        .link();
}

fn main() {
    compile_bindings();
}
