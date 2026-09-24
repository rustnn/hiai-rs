# hiai-rs

Rust bindings for Huawei Ascend NPUs using the HiAI (CANN) DDK on OpenHarmony — build and run neural-network graphs.

> **EXPERIMENTAL — NOT FOR PRODUCTION USE.** The API is unstable and will change.

## Workspace layout

| Crate | Purpose |
|---|---|
| [`hiai-rs-sys`](hiai-rs-sys/) | Raw, `unsafe` FFI bindings to the HiAI DDK (bindgen-generated). Internal use only. |
| [`hiai-rs`](hiai-rs/) | Safe Rust wrapper: `TensorDesc` + `dispatch()` for running prebuilt models. |

## What it wraps

The HiAI DDK is Huawei's `libhiai` runtime (part of the CANN stack) for OpenHarmony devices with Ascend NPUs. This crate binds a thin **pure-C adapter** (`hiai-rs-sys/adapter/`) that wraps the DDK's C++ API (`hiai::op::*`, the GE graph engine, and the model manager) so it can be called from Rust.

- **Graph build** (WebNN → CANN IR) lives in [rustnn](https://github.com/rustnn/rustnn); this crate only binds the low-level FFI and model dispatch.
- The adapter is compiled and `libhiai` is linked **only** when cross-compiling for OpenHarmony (`aarch64-unknown-linux-ohos`).

## Building

### Host (bindgen smoke test — no DDK required)

The adapter's public headers are self-contained, so bindgen runs on any host:

```bash
cargo build
```

### OpenHarmony device (full build + link)

Requires the Huawei CANN-Kit DDK and the OpenHarmony SDK, plus the OHOS target:

```bash
export CANN_DDK=/path/to/CANN-Kit-next/ddk/
export OHOS_SDK_NATIVE=/path/to/OpenHarmony/sdk/native
cargo build --target aarch64-unknown-linux-ohos --release
```

## Usage

```rust,no_run
use hiai_rs::{TensorDesc, dispatch};

// Prebuilt offline model bytes (produced by the rustnn CANN converter).
let model_bytes: &[u8] = todo!();

let inputs = vec![TensorDesc {
    data: vec![1.0f32; 4].into_iter().flat_map(|v| v.to_le_bytes()).collect(),
    shape: vec![1, 4],
    dtype: 0, // CANN_DT_FLOAT
}];

let mut outputs = vec![TensorDesc {
    data: vec![0u8; 16],
    shape: vec![1, 4],
    dtype: 0,
}];

dispatch(model_bytes, &inputs, &mut outputs)?;
# Ok::<(), hiai_rs::Error>(())
```

## Environment variables

- `CANN_DDK` — path to the Huawei CANN-Kit DDK (required for OpenHarmony builds).
- `OHOS_SDK_NATIVE` — path to the OpenHarmony SDK native toolchain (for the `aarch64-unknown-linux-ohos` target).

## Attribution

The C++ adapter under `hiai-rs-sys/adapter/` is a local pure-C shim over the HiAI DDK. It is licensed under Apache-2.0 (see each file's SPDX header).

## License

Apache-2.0. See [LICENSE](LICENSE).
