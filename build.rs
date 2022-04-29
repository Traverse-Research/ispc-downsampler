#[cfg(feature = "ispc")]
fn compile_bindings() {
    use ispc_compile::{MathLib, TargetISA, TargetOS};

    // Compile our ISPC library, this call will exit with EXIT_FAILURE if
    // compilation fails.

    let target_os = if cfg!(target_os = "windows") {
        TargetOS::Windows
    } else if cfg!(target_os = "linux") {
        TargetOS::Linux
    } else if cfg!(target_os = "macos") {
        TargetOS::Macos
    } else if cfg!(target_os = "android") {
        TargetOS::Android
    } else {
        panic!("Unsupported platform")
    };

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
        .target_os(target_os)
        .math_lib(MathLib::Fast)
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
