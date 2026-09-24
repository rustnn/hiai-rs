/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

fn build_bindings() {
    use std::{env, path::Path};

    // The pure-C adapter is vendored inside this crate (cargo only packages
    // files below the crate root, so it cannot live at the workspace root).
    let shim_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("adapter");
    let shim_dir = shim_dir.to_string_lossy();

    // Generate Rust FFI bindings from the adapter's self-contained public C
    // headers. These include only `adapter_types.h` + `<stdint.h>`, so bindgen
    // runs on any host (no DDK required).
    let headers = [
        "adapter_types.h",
        "context_adapter.h",
        "graph_adapter.h",
        "io_tensor_adapter.h",
        "model_adapter.h",
        "model_manager_adapter.h",
        "operator_adapter.h",
        "op_tensor_adapter.h",
    ];

    let mut builder = bindgen::Builder::default()
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{shim_dir}"))
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .allowlist_type(".*Cann.*")
        .allowlist_function(".*cann_.*");

    for header in &headers {
        println!("cargo:rerun-if-changed={shim_dir}/{header}");
        builder = builder.header(format!("{shim_dir}/{header}"));
    }

    let bindings = builder
        .generate()
        .expect("Unable to generate CANN bindings");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("cann_bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write CANN bindings");

    // Compile the C++ adapter and link the HiAI DDK. Only valid on the
    // OpenHarmony target; on other targets we only generate bindings.
    if env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() != "ohos" {
        return;
    }

    // The Huawei CANN-Kit DDK is a prerequisite for compiling/linking.
    let ddk = env::var("CANN_DDK")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| panic!("CANN_DDK not set. export CANN_DDK=/path/to/CANN-Kit-next/ddk/"));
    println!("cargo:rerun-if-env-changed=CANN_DDK");

    let ddk_include = Path::new(&ddk).join("ai_ddk_lib").join("include");
    let ddk_lib = Path::new(&ddk).join("ai_ddk_lib").join("lib64");

    let sources = [
        "context_adapter.cc",
        "graph_adapter.cc",
        "model_adapter.cc",
        "model_manager_adapter.cc",
        "operator_adapter.cc",
        "io_tensor_adapter.cc",
        "op_tensor_adapter.cc",
    ];

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .include(shim_dir.as_ref())
        .include(&ddk_include)
        .flag("-std=c++17")
        .flag("-fvisibility=hidden");
    for src in &sources {
        let path = format!("{shim_dir}/{src}");
        println!("cargo:rerun-if-changed={path}");
        build.file(&path);
    }
    build.compile("hiai_adapter");

    println!("cargo:rustc-link-lib=dylib=hiai");
    println!("cargo:rustc-link-lib=dylib=hiai_ir");
    println!("cargo:rustc-link-lib=dylib=hiai_ir_build");
    println!("cargo:rustc-link-lib=dylib=hiai_ir_build_aipp");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!(
        "cargo:rustc-link-search=native={}",
        ddk_lib.to_string_lossy()
    );
}

fn main() {
    build_bindings();
}
