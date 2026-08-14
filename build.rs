//! Embeds build provenance (rustc version, profile, RUSTFLAGS) so run
//! manifests can record exactly how the binary was produced.

use std::env;
use std::process::Command;

fn main() {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=BUILD_RUSTC_VERSION={version}");
    println!(
        "cargo:rustc-env=BUILD_PROFILE={}",
        env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string())
    );
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS")
        .map(|f| f.replace('\u{1f}', " "))
        .unwrap_or_default();
    println!("cargo:rustc-env=BUILD_RUSTFLAGS={rustflags}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}
