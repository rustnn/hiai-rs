# Design

## Overview

hiai-rs binds Huawei's HiAI (CANN) DDK (`libhiai`) so Rust can build and run neural-network graphs on Ascend NPUs from OpenHarmony.

The DDK's native API is C++ (`hiai::op::*` operators, the GE graph engine, and the model manager). Rust cannot call those classes directly, so this crate ships a **pure-C adapter** (`adapter/`) that wraps the DDK into `extern "C"` functions (`cann_*`) over opaque handles (`Cann*`).

## Why bindgen

We use `bindgen` to generate Rust bindings for the **flat pure-C API** exposed by the HiAI adapter.

```
adapter/*.h  ──bindgen──►  OUT_DIR/cann_bindings.rs  ──include!──►  hiai_rs_sys::sys
adapter/*.cc ──cc───────►  compiled adapter          ──link──────►  libhiai (OHOS only)
```

## Bindgen configuration

In `hiai-rs-sys/build.rs`:

- `-x c++ -std=c++17 -Iadapter`
- `default_enum_style(Rust { non_exhaustive: false })`
- `allowlist_type(".*Cann.*")` + `allowlist_function(".*cann_.*")`

The public adapter headers (`adapter_types.h`, `*_adapter.h`) include only `adapter_types.h` + `<stdint.h>`, so bindgen runs on any host with libclang — no DDK required.

## Compilation / linking

The adapter `.cc` files include DDK headers (`compatible/*.h`, `graph/*.h`, `hiai_ir_build.h`, `HiAiModelManagerService.h`) and link `libhiai`, `libhiai_ir`, `libhiai_ir_build`, `libhiai_ir_build_aipp`, and `libc++`. This is only valid on the OpenHarmony target, so `build.rs` gates it behind `CARGO_CFG_TARGET_ENV == "ohos"` and requires `CANN_DDK`.

## Data types

The adapter mirrors `ge::DataType` and `ge::Format` as `CannDataType` / `CannFormat` enums. `hiai-rs`'s `TensorDesc.dtype` carries the raw `CANN_DT_*` value (`i32`) and is mapped to the bindgen enum via an explicit `match` in `dispatch()`; unknown or sentinel codes are rejected with `Error::InvalidDataType`.

## Model lifecycle

`dispatch()` performs the full runtime path:

1. Create + init the model manager (`cann_model_manager_*`).
2. Build a model descriptor from the offline model bytes (`cann_model_desc_*`).
3. Load the model.
4. Create input/output IO tensors from `TensorDesc` shape + dtype.
5. Run inference (`cann_model_manager_process`).
6. Read outputs back, then tear everything down.

## Provenance / license

The adapter is a local pure-C shim (Apache-2.0, SPDX headers) wiring. The DDK itself (`libhiai`) is Huawei's CANN-Kit and is distributed under its own license; it is a link-time dependency, not vendored here.
