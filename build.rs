#[cfg(feature = "ispc_cmp")]
use ispc::{TargetISA, TargetOS};

fn main() {
    // Compile our ISPC library, this call will exit with EXIT_FAILURE if
    // compilation fails.
    #[cfg(feature = "ispc_cmp")]
    ispc_compile::Config::new()
        .file("src/ispc/kernels/lanczos3.ispc")
        .opt_level(2)
        .woff()
        .target_isas(vec![
            TargetISA::SSE2i32x4,
            TargetISA::SSE4i32x4,
            TargetISA::AVX1i32x8,
            TargetISA::AVX2i32x8,
            TargetISA::AVX512KNLi32x16,
            TargetISA::AVX512SKXi32x16,
        ])
        .target_os(TargetOS::Windows)
        .out_dir("src/ispc")
        .compile("downsample_ispc");
    if !cfg!(feature = "ispc_cmp") {
        ispc_rt::PackagedModule::new("kernel")
            .lib_path("src/ispc")
            .link();
    }
}
