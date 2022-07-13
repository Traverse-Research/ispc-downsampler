#[cfg(feature = "ispc")]
fn compile_bindings() {
    use ispc_compile::{BindgenOptions, Config, MathLib, TargetISA};

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
        .file("src/ispc/kernels/lanczos3.ispc")
        .opt_level(2)
        .woff()
        .target_isas(target_isas)
        .math_lib(MathLib::Fast)
        .bindgen_options(BindgenOptions {
            allowlist_functions: vec!["resample".into()],
        })
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
