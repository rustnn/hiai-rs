/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

//! Model dispatch over the HiAI DDK.
//!
//! `dispatch` is a thin, safe wrapper over the raw [`hiai_rs_sys`] bindings:
//! it loads a prebuilt offline model, feeds input tensors, runs inference on
//! the NPU, and reads the output tensors back.

use crate::error::{CannStatus, Error, Result};
use hiai_rs_sys::*;

/// Enable per-dispatch `[cann-timing]` logs. Off by default; set `CANN_TIMING=1`.
/// The breadcrumbs are emitted at `debug` via the `log` crate, so a logger
/// backend (e.g. `env_logger`) must also be installed to see them.
fn timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    env_flag("CANN_TIMING", &ENABLED)
}

/// Read an env var as a bool (true only when it equals `"1"`), cached in `cache`.
fn env_flag(name: &'static str, cache: &std::sync::OnceLock<bool>) -> bool {
    *cache.get_or_init(|| std::env::var(name).map(|v| v == "1").unwrap_or(false))
}

/// Elapsed milliseconds between two optional instants, or `0.0` when timing is
/// disabled (both `None`).
fn elapsed_ms(from: Option<std::time::Instant>, to: Option<std::time::Instant>) -> f64 {
    match (from, to) {
        (Some(a), Some(b)) => (b - a).as_secs_f64() * 1e3,
        _ => 0.0,
    }
}

/// Input tensor descriptor: `data` borrows the caller's bytes, which
/// `dispatch` copies into the DDK's ION buffer for the inference.
#[derive(Debug, Clone)]
pub struct InputDesc<'a> {
    /// Raw tensor bytes (little-endian), borrowed from the caller.
    pub data: &'a [u8],
    /// Tensor shape (dimension sizes).
    pub shape: Vec<u32>,
    /// Raw CANN data-type value (`CANN_DT_*`). Converted to the bindgen
    /// `ddk_CannDataType` enum at the FFI boundary.
    pub dtype: i32,
}

/// Output tensor descriptor: `dispatch` copies the NPU result into `data` and
/// records the produced byte length in `actual_len`.
#[derive(Debug)]
pub struct OutputDesc<'a> {
    /// Output buffer, filled by `dispatch`. The caller must provide a buffer at
    /// least as large as the tensor's logical size; `actual_len` is set to the
    /// number of bytes actually produced.
    pub data: &'a mut [u8],
    /// Tensor shape (dimension sizes).
    pub shape: Vec<u32>,
    /// Raw CANN data-type value (`CANN_DT_*`). Converted to the bindgen
    /// `ddk_CannDataType` enum at the FFI boundary.
    pub dtype: i32,
    /// Number of bytes the NPU produced for this output (≤ the length of
    /// `data`). Set by `dispatch` on success; on any error it is left
    /// unchanged, so ignore it unless `dispatch` returned `Ok`.
    pub actual_len: usize,
}

/// Maps a raw `CANN_DT_*` integer to the bindgen `ddk_CannDataType` enum.
///
/// Uses an explicit match (no `transmute`) so an out-of-range or negative value
/// from the caller yields an error instead of an invalid enum value.
fn data_type(value: i32) -> Result<ddk_CannDataType> {
    match value {
        0 => Ok(ddk_CannDataType::CANN_DT_FLOAT),
        1 => Ok(ddk_CannDataType::CANN_DT_FLOAT16),
        2 => Ok(ddk_CannDataType::CANN_DT_INT8),
        3 => Ok(ddk_CannDataType::CANN_DT_INT32),
        4 => Ok(ddk_CannDataType::CANN_DT_UINT8),
        6 => Ok(ddk_CannDataType::CANN_DT_INT16),
        7 => Ok(ddk_CannDataType::CANN_DT_UINT16),
        8 => Ok(ddk_CannDataType::CANN_DT_UINT32),
        9 => Ok(ddk_CannDataType::CANN_DT_INT64),
        10 => Ok(ddk_CannDataType::CANN_DT_UINT64),
        11 => Ok(ddk_CannDataType::CANN_DT_DOUBLE),
        12 => Ok(ddk_CannDataType::CANN_DT_BOOL),
        13 => Ok(ddk_CannDataType::CANN_DT_DUAL),
        14 => Ok(ddk_CannDataType::CANN_DT_DUAL_SUB_INT8),
        15 => Ok(ddk_CannDataType::CANN_DT_DUAL_SUB_UINT8),
        16 => Ok(ddk_CannDataType::CANN_DT_COMPLEX64),
        21 => Ok(ddk_CannDataType::CANN_DT_2BIT),
        22 => Ok(ddk_CannDataType::CANN_DT_INT4),
        23 => Ok(ddk_CannDataType::CANN_DT_QUINT8),
        24 => Ok(ddk_CannDataType::CANN_DT_RESOURCE),
        25 => Ok(ddk_CannDataType::CANN_DT_3BIT),
        26 => Ok(ddk_CannDataType::CANN_DT_UINT2),
        27 => Ok(ddk_CannDataType::CANN_DT_UINT4),
        28 => Ok(ddk_CannDataType::CANN_DT_STRING),
        35 => Ok(ddk_CannDataType::CANN_DT_FLOAT8_E5M2),
        40 => Ok(ddk_CannDataType::CANN_DT_FLOAT4_E2M1),
        _ => Err(Error::InvalidDataType { value }),
    }
}

/// Converts a `usize` count to the DDK's `i32` representation, guarding against
/// truncation (the DDK takes `int32_t` counts).
fn count_i32(value: usize, what: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::TooMany { what })
}

/// Defines a RAII guard for a raw CANN handle: it owns the handle and destroys
/// it on `Drop`, so early returns (and panics) cannot leak the C resource.
macro_rules! raw_handle {
    ($name:ident, $ty:ty, $drop:path) => {
        struct $name($ty);

        impl $name {
            /// Wraps `handle`, or returns `None` if it is null.
            fn new(handle: $ty) -> Option<Self> {
                if handle.is_null() {
                    None
                } else {
                    Some(Self(handle))
                }
            }

            fn as_ptr(&self) -> $ty {
                self.0
            }

            // Only some handles need a mutable reference (e.g. `load` takes
            // `&mut ddk_CannModelDescHandle`), so allow unused in others.
            #[allow(dead_code)]
            fn as_mut_ptr(&mut self) -> *mut $ty {
                &mut self.0
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: `self.0` is a non-null handle owned by this guard.
                unsafe { $drop(self.0) };
            }
        }
    };
}

raw_handle!(
    ModelManager,
    ddk_CannModelManagerHandle,
    ddk_cann_model_manager_destroy
);
raw_handle!(
    ModelDesc,
    ddk_CannModelDescHandle,
    ddk_cann_model_desc_destroy
);
raw_handle!(Context, ddk_CannContextHandle, ddk_cann_context_destroy);
raw_handle!(IoTensor, ddk_CannIOTensorHandle, ddk_cann_io_tensor_destroy);
raw_handle!(
    IoTensorDim,
    ddk_CannIOTensorDimensionHandle,
    ddk_cann_io_tensor_dim_destroy
);

/// A DDK IO tensor (dimension + tensor handle) cached across dispatches so the
/// per-call `AiTensor`/dimension construction is paid once, not every inference.
struct CachedIo {
    // Declared before `dimension` on purpose: fields drop in declaration order,
    // so the tensor is destroyed first and the DDK dimension handle it may
    // reference internally outlives it.
    tensor: IoTensor,
    // Held (never read) to keep the DDK dimension handle alive for as long as
    // the tensor: the tensor may reference it internally until `Process`.
    #[allow(dead_code)]
    dimension: IoTensorDim,
    shape: Vec<u32>,
    dtype: i32,
}

/// Cached input/output IO tensors, reused across dispatches when shapes and
/// dtypes are unchanged (the common case for a fixed-shape model). This is the
/// standard HiAI pattern: `AiTensor` objects are created once from the model's
/// IO dims and passed to `Process` on every call.
#[derive(Default)]
struct IoCache {
    inputs: Vec<CachedIo>,
    outputs: Vec<CachedIo>,
}

/// Selects the input or output half of the [`IoCache`].
#[derive(Clone, Copy)]
enum IoKind {
    Input,
    Output,
}

impl IoCache {
    fn slots_mut(&mut self, kind: IoKind) -> &mut Vec<CachedIo> {
        match kind {
            IoKind::Input => &mut self.inputs,
            IoKind::Output => &mut self.outputs,
        }
    }
}

/// A loaded CANN model, reusable across many inferences.
///
/// `Session::load` performs the expensive model-manager init + model load +
/// inference-context creation once. `Session::dispatch` reuses the loaded
/// model, context, and (once created) the IO tensors, avoiding the model
/// reload, per-call `AiContext` construction, and per-call IO-tensor/dimension
/// construction that dominate per-call latency.
pub struct Session {
    manager: ModelManager,
    context: Context,
    io_cache: IoCache,
}

// SAFETY: the wrapped handles are owned exclusively by this `Session`; moving
// the `Session` to another thread transfers that exclusive ownership. `Sync` is
// deliberately NOT implemented: `dispatch(&mut self)` mutates the DDK model
// manager internally, so sharing a `&Session` across threads would race inside
// C++.
unsafe impl Send for Session {}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

impl Session {
    /// Loads a prebuilt offline model into the NPU model manager.
    pub fn load(model_bytes: &[u8]) -> Result<Self> {
        use std::ffi::CString;

        let manager =
            ModelManager::new(unsafe { ddk_cann_model_manager_create() }).ok_or(Error::Null {
                what: "model manager",
            })?;
        let status = unsafe { ddk_cann_model_manager_init(manager.as_ptr()) };
        if status != 0 {
            return Err(Error::Cann {
                operation: "initialize model manager",
                status: CannStatus::from_code(status),
            });
        }

        let name = CString::new("webnn_model").unwrap();
        let mut descriptor = ModelDesc::new(unsafe {
            ddk_cann_model_desc_create(
                name.as_ptr(),
                3, // AiModelDescription_Frequency_HIGH
                0, // HIAI_FRAMEWORK_NONE
                1, // HIAI_MODELTYPE_OFFLINE
                0, // AiModelDescription_DeviceType_NPU
            )
        })
        .ok_or(Error::Null {
            what: "model descriptor",
        })?;

        // Run inference in FP16 (the Ascend NPU's native precision). This must
        // be set on the model descriptor before loading.
        let status = unsafe { ddk_cann_model_desc_set_precision_mode(descriptor.as_ptr(), 1) };
        if status != 0 {
            return Err(Error::Cann {
                operation: "set precision mode",
                status: CannStatus::from_code(status),
            });
        }

        // The DDK's SetModelBuffer takes a `uint32_t` size; reject a buffer that
        // would not fit instead of truncating via `as u32`.
        let model_len: u32 = model_bytes.len().try_into().map_err(|_| Error::TooLarge {
            what: "model buffer",
        })?;
        let status = unsafe {
            ddk_cann_model_desc_set_model_buffer(
                descriptor.as_ptr(),
                model_bytes.as_ptr() as *const _,
                model_len,
            )
        };
        if status != 0 {
            return Err(Error::Cann {
                operation: "set model buffer",
                status: CannStatus::from_code(status),
            });
        }
        let status =
            unsafe { ddk_cann_model_manager_load(manager.as_ptr(), descriptor.as_mut_ptr(), 1) };
        if status != 0 {
            return Err(Error::Cann {
                operation: "load model",
                status: CannStatus::from_code(status),
            });
        }

        // Create the inference context once and bind it to the loaded model.
        // This was previously done per-`dispatch` (`new hiai::AiContext` +
        // `SetPara` each call); hoisting it here removes that per-call cost.
        let context = Context::new(unsafe { ddk_cann_context_create() }).ok_or(Error::Null {
            what: "inference context",
        })?;
        let model_name = CString::new("model_name").unwrap();
        let model_val = CString::new("webnn_model").unwrap();
        let status = unsafe {
            ddk_cann_context_set_para(context.as_ptr(), model_name.as_ptr(), model_val.as_ptr())
        };
        if status != 0 {
            return Err(Error::Cann {
                operation: "set context parameter",
                status: CannStatus::from_code(status),
            });
        }

        Ok(Self {
            manager,
            context,
            io_cache: IoCache::default(),
        })
    }

    /// Reconciles the cached IO tensors for `kind` with `slots`, recreating only
    /// the entries whose shape/dtype changed. On the common fixed-shape path
    /// this is a no-op.
    fn ensure_cached(&mut self, kind: IoKind, slots: &[(Vec<u32>, i32)]) -> Result<()> {
        let cache = self.io_cache.slots_mut(kind);
        cache.truncate(slots.len());
        for (index, (shape, dtype)) in slots.iter().enumerate() {
            let unchanged = cache
                .get(index)
                .is_some_and(|cached| &cached.shape == shape && cached.dtype == *dtype);
            if unchanged {
                continue;
            }
            let fresh = Self::create_cached_io(shape, *dtype)?;
            if index < cache.len() {
                cache[index] = fresh;
            } else {
                cache.push(fresh);
            }
        }
        Ok(())
    }

    /// Creates one cached `AiTensor` (ION memory) for a `(shape, dtype)` slot.
    fn create_cached_io(shape: &[u32], dtype: i32) -> Result<CachedIo> {
        let dimension = IoTensorDim::new(unsafe {
            ddk_cann_io_tensor_dim_create_nd(shape.as_ptr(), count_i32(shape.len(), "shape rank")?)
        })
        .ok_or(Error::Null {
            what: "IO tensor dimension",
        })?;
        let tensor = IoTensor::new(unsafe { ddk_cann_io_tensor_create() })
            .ok_or(Error::Null { what: "IO tensor" })?;
        let dt = data_type(dtype)?;
        let status = unsafe { ddk_cann_io_tensor_init(tensor.as_ptr(), dimension.as_ptr(), dt) };
        if status != 0 {
            return Err(Error::Cann {
                operation: "initialize IO tensor",
                status: CannStatus::from_code(status),
            });
        }
        Ok(CachedIo {
            tensor,
            dimension,
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Pre-creates the DDK IO tensors (ION memory) at build/load time, so the
    /// per-dispatch ION allocation (the `out_init`/input-alloc cost) is paid
    /// once outside the inference window. This matches Chromium's architecture
    /// (`BuildModel` + `LoadModelsByHcl` in `build()`, `dispatch()` only runs).
    ///
    /// `slots` are `(shape, dtype)` pairs. They must match what `dispatch` will
    /// later receive; on a mismatch `dispatch` lazily recreates the tensor
    /// (current behavior), so this is purely an optimization.
    pub fn prepare_io(
        &mut self,
        input_slots: &[(Vec<u32>, i32)],
        output_slots: &[(Vec<u32>, i32)],
    ) -> Result<()> {
        self.ensure_cached(IoKind::Input, input_slots)?;
        self.ensure_cached(IoKind::Output, output_slots)?;
        Ok(())
    }

    /// Runs one inference on the loaded model.
    pub fn dispatch(
        &mut self,
        inputs: &[InputDesc<'_>],
        outputs: &mut [OutputDesc<'_>],
    ) -> Result<()> {
        // `Instant::now()` is only captured when timing is enabled; otherwise
        // every `tN` is `None` and the readback timers below are skipped.
        let timing = timing_enabled();
        let t0 = timing.then(std::time::Instant::now);

        // 1. Inputs: pre-create/reuse the cached DDK ION tensors, then copy the
        //    caller's bytes into them.
        let input_slots: Vec<(Vec<u32>, i32)> =
            inputs.iter().map(|i| (i.shape.clone(), i.dtype)).collect();
        self.ensure_cached(IoKind::Input, &input_slots)?;
        for (input, cached) in inputs.iter().zip(self.io_cache.inputs.iter()) {
            // SetData takes a `uint32_t` size; guard against truncation.
            let input_len: u32 = input.data.len().try_into().map_err(|_| Error::TooLarge {
                what: "input tensor",
            })?;
            let status = unsafe {
                ddk_cann_io_tensor_set_data(
                    cached.tensor.as_ptr(),
                    input.data.as_ptr() as *const _,
                    input_len,
                )
            };
            if status != 0 {
                return Err(Error::Cann {
                    operation: "write input tensor data",
                    status: CannStatus::from_code(status),
                });
            }
        }
        let t1 = timing.then(std::time::Instant::now);

        // 2. Outputs: pre-create/reuse the cached DDK ION tensors.
        let output_slots: Vec<(Vec<u32>, i32)> =
            outputs.iter().map(|o| (o.shape.clone(), o.dtype)).collect();
        self.ensure_cached(IoKind::Output, &output_slots)?;
        let t2 = timing.then(std::time::Instant::now);

        // 3. Run inference using the cached context (no per-call `AiContext`
        // construction). `process` takes raw `*mut ddk_CannIOTensorHandle`
        // arrays, collected from the cached tensors.
        let mut raw_inputs: Vec<ddk_CannIOTensorHandle> = self
            .io_cache
            .inputs
            .iter()
            .map(|c| c.tensor.as_ptr())
            .collect();
        let mut raw_outputs: Vec<ddk_CannIOTensorHandle> = self
            .io_cache
            .outputs
            .iter()
            .map(|c| c.tensor.as_ptr())
            .collect();

        // `stamp` is a DDK out-param (the inference timestamp); we don't use it.
        let mut stamp: i32 = 0;
        let status = unsafe {
            ddk_cann_model_manager_process(
                self.manager.as_ptr(),
                self.context.as_ptr(),
                raw_inputs.as_mut_ptr(),
                count_i32(raw_inputs.len(), "input count")?,
                raw_outputs.as_mut_ptr(),
                count_i32(raw_outputs.len(), "output count")?,
                1000,
                &mut stamp,
            )
        };
        if status != 0 {
            return Err(Error::Cann {
                operation: "run inference",
                status: CannStatus::from_code(status),
            });
        }
        let t3 = timing.then(std::time::Instant::now);

        // 4. Read outputs back: copy out of the DDK ION buffer into the caller's
        //    buffer.
        let mut rb_sync_ms = 0.0f64;
        let mut rb_copy_ms = 0.0f64;
        let mut rb_first_ms = 0.0f64;
        let mut rb_rest_ms = 0.0f64;
        for (output, handle) in outputs.iter_mut().zip(raw_outputs.iter().copied()) {
            let sync_start = timing.then(std::time::Instant::now);
            let buffer = unsafe { ddk_cann_io_tensor_get_buffer(handle) };
            if buffer.is_null() {
                return Err(Error::OutputBufferNull);
            }
            // Copy exactly the tensor's reported byte size, not the caller's buffer
            // length, so we never read past the DDK's buffer.
            let actual = unsafe { ddk_cann_io_tensor_get_size(handle) } as usize;
            if let Some(start) = sync_start {
                rb_sync_ms += start.elapsed().as_secs_f64() * 1e3;
            }
            if output.data.len() < actual {
                return Err(Error::OutputBufferTooSmall);
            }
            // SAFETY: `buffer` is non-null and `ddk_cann_io_tensor_get_size` reports
            // `actual` bytes of valid memory for this tensor.
            let copy_start = timing.then(std::time::Instant::now);
            let source = unsafe { std::slice::from_raw_parts(buffer as *const u8, actual) };
            if timing && !source.is_empty() {
                // First read: touching the first byte triggers any lazy NPU-drain /
                // cache-coherency sync. Time it separately from the bulk copy to
                // distinguish a device-sync wait from a slow cold-buffer read.
                let first_start = std::time::Instant::now();
                std::hint::black_box(source[0]);
                rb_first_ms += first_start.elapsed().as_secs_f64() * 1e3;
                let rest_start = std::time::Instant::now();
                output.data[..actual].copy_from_slice(source);
                rb_rest_ms += rest_start.elapsed().as_secs_f64() * 1e3;
            } else {
                output.data[..actual].copy_from_slice(source);
            }
            if let Some(start) = copy_start {
                rb_copy_ms += start.elapsed().as_secs_f64() * 1e3;
            }
            output.actual_len = actual;
        }
        let t4 = timing.then(std::time::Instant::now);

        if timing {
            log::debug!(
                "[cann-timing] set_data={:.2}ms out_init={:.2}ms process={:.2}ms readback={:.2}ms sync={:.2}ms copy={:.2}ms first={:.3}ms rest={:.2}ms",
                elapsed_ms(t0, t1),
                elapsed_ms(t1, t2),
                elapsed_ms(t2, t3),
                elapsed_ms(t3, t4),
                rb_sync_ms,
                rb_copy_ms,
                rb_first_ms,
                rb_rest_ms,
            );
        }

        Ok(())
    }
}

/// One-shot dispatch: loads the model, runs one inference, and drops the
/// session. Prefer [`Session`] for repeated inference.
pub fn dispatch(
    model_bytes: &[u8],
    inputs: &[InputDesc<'_>],
    outputs: &mut [OutputDesc<'_>],
) -> Result<()> {
    let mut session = Session::load(model_bytes)?;
    session.dispatch(inputs, outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_type_maps_valid_codes() {
        assert_eq!(data_type(0).unwrap(), ddk_CannDataType::CANN_DT_FLOAT);
        assert_eq!(data_type(1).unwrap(), ddk_CannDataType::CANN_DT_FLOAT16);
        assert_eq!(
            data_type(15).unwrap(),
            ddk_CannDataType::CANN_DT_DUAL_SUB_UINT8
        );
    }

    #[test]
    fn data_type_rejects_invalid_codes() {
        assert!(matches!(
            data_type(-1),
            Err(Error::InvalidDataType { value: -1 })
        ));
        assert!(matches!(
            data_type(9999),
            Err(Error::InvalidDataType { value: 9999 })
        ));
    }
}
