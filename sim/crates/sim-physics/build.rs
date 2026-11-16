fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .or_else(|_| std::env::var("RUSTFLAGS"))
        .unwrap_or_default();
    assert!(
        flags.contains("codegen-backend=") && flags.contains("rustc_codegen_cuda"),
        "sim-physics requires NVIDIA cuda-oxide; build with cargo oxide build --arch sm_120 --device-codegen-crate sim_physics -- -p sim-smoke --release (plain cargo has no device-code fallback)"
    );
}
