/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

//! Model dispatch over the HiAI DDK.
//!
//! `dispatch` is a thin, safe wrapper over the raw [`hiai_rs_sys`] bindings:
//! it loads a prebuilt offline model, feeds input tensors, runs inference on
//! the NPU, and copies the output tensors back.

use crate::error::{CannStatus, Error, Result};
use hiai_rs_sys::*;

/// Rust-side tensor descriptor passed between the caller and the HiAI runtime.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    /// Raw tensor bytes (little-endian).
    pub data: Vec<u8>,
    /// Tensor shape (dimension sizes).
    pub shape: Vec<u32>,
    /// Raw CANN data-type value (`CANN_DT_*`). Converted to the bindgen
    /// `ddk_CannDataType` enum at the FFI boundary.
    pub dtype: i32,
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
        17 => Ok(ddk_CannDataType::CANN_DT_UNDEFINED),
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
        41 => Ok(ddk_CannDataType::CANN_DT_MAX),
        _ => Err(Error::InvalidDataType { value }),
    }
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

/// Dispatches a prebuilt offline model on the NPU.
///
/// # Arguments
///
/// - `model_bytes`: offline model bytes produced by the CANN IR builder.
/// - `inputs`: input tensors, in the model's canonical input order.
/// - `outputs`: output tensors, in the model's canonical output order. Each
///   entry's `data` buffer must be large enough to receive the output; on
///   success it is truncated to the actual output size and overwritten with
///   the computed values.
pub fn dispatch(
    model_bytes: &[u8],
    inputs: &[TensorDesc],
    outputs: &mut [TensorDesc],
) -> Result<()> {
    use std::ffi::CString;

    // 1. Init model manager
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

    // 2. Load model
    let name = CString::new("webnn_model").unwrap();
    let mut descriptor = ModelDesc::new(unsafe {
        ddk_cann_model_desc_create(
            name.as_ptr(),
            3, // AiModelDescription_Frequency_HIGH
            0, // HIAI_FRAMEWORK_NONE
            0, // HIAI_MODELTYPE_OFFLINE
            0, // AiModelDescription_DeviceType_NPU
        )
    })
    .ok_or(Error::Null {
        what: "model descriptor",
    })?;

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

    // 3. Build input tensors
    let mut input_handles: Vec<IoTensor> = Vec::new();
    let mut dimension_handles: Vec<IoTensorDim> = Vec::new();

    for input in inputs.iter() {
        let dimension = IoTensorDim::new(unsafe {
            ddk_cann_io_tensor_dim_create_nd(input.shape.as_ptr(), input.shape.len() as i32)
        })
        .ok_or(Error::Null {
            what: "IO tensor dimension",
        })?;

        let tensor = IoTensor::new(unsafe { ddk_cann_io_tensor_create() })
            .ok_or(Error::Null { what: "IO tensor" })?;

        let dtype = data_type(input.dtype)?;
        let status = unsafe { ddk_cann_io_tensor_init(tensor.as_ptr(), dimension.as_ptr(), dtype) };
        if status != 0 {
            return Err(Error::Cann {
                operation: "initialize IO tensor",
                status: CannStatus::from_code(status),
            });
        }

        // SetData takes a `uint32_t` size; guard against truncation.
        let input_len: u32 = input.data.len().try_into().map_err(|_| Error::TooLarge {
            what: "input tensor",
        })?;
        let status = unsafe {
            ddk_cann_io_tensor_set_data(tensor.as_ptr(), input.data.as_ptr() as *const _, input_len)
        };
        if status != 0 {
            return Err(Error::Cann {
                operation: "write input tensor data",
                status: CannStatus::from_code(status),
            });
        }

        dimension_handles.push(dimension);
        input_handles.push(tensor);
    }

    // 4. Build output tensors
    let mut output_handles: Vec<IoTensor> = Vec::new();
    for output in outputs.iter() {
        let dimension = IoTensorDim::new(unsafe {
            ddk_cann_io_tensor_dim_create_nd(output.shape.as_ptr(), output.shape.len() as i32)
        })
        .ok_or(Error::Null {
            what: "IO tensor dimension",
        })?;

        let tensor = IoTensor::new(unsafe { ddk_cann_io_tensor_create() })
            .ok_or(Error::Null { what: "IO tensor" })?;

        let dtype = data_type(output.dtype)?;
        let status = unsafe { ddk_cann_io_tensor_init(tensor.as_ptr(), dimension.as_ptr(), dtype) };
        if status != 0 {
            return Err(Error::Cann {
                operation: "initialize IO tensor",
                status: CannStatus::from_code(status),
            });
        }

        dimension_handles.push(dimension);
        output_handles.push(tensor);
    }

    // 5. Run inference
    let context = Context::new(unsafe { ddk_cann_context_create() }).ok_or(Error::Null {
        what: "inference context",
    })?;
    let model_name = CString::new("model_name").unwrap();
    let model_val = CString::new("webnn_model").unwrap();
    unsafe { ddk_cann_context_set_para(context.as_ptr(), model_name.as_ptr(), model_val.as_ptr()) };

    // `process` takes raw `*mut ddk_CannIOTensorHandle` arrays; build them from
    // the guards (which still own the handles and clean them up on drop).
    let mut raw_inputs: Vec<ddk_CannIOTensorHandle> =
        input_handles.iter().map(|t| t.as_ptr()).collect();
    let mut raw_outputs: Vec<ddk_CannIOTensorHandle> =
        output_handles.iter().map(|t| t.as_ptr()).collect();

    let mut stamp: i32 = 0;
    let status = unsafe {
        ddk_cann_model_manager_process(
            manager.as_ptr(),
            context.as_ptr(),
            raw_inputs.as_mut_ptr(),
            raw_inputs.len() as i32,
            raw_outputs.as_mut_ptr(),
            raw_outputs.len() as i32,
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

    // 6. Read outputs back
    for (output, handle) in outputs.iter_mut().zip(raw_outputs.iter().copied()) {
        let buffer = unsafe { ddk_cann_io_tensor_get_buffer(handle) };
        if buffer.is_null() {
            return Err(Error::OutputBufferNull);
        }
        // Copy exactly the tensor's reported byte size, not the caller's buffer
        // length, so we never read past the DDK's buffer.
        let actual = unsafe { ddk_cann_io_tensor_get_size(handle) } as usize;
        if output.data.len() < actual {
            return Err(Error::OutputBufferTooSmall);
        }
        // SAFETY: `buffer` is non-null and `ddk_cann_io_tensor_get_size` reports
        // `actual` bytes of valid memory for this tensor.
        let source = unsafe { std::slice::from_raw_parts(buffer as *const u8, actual) };
        // Trim the caller's buffer to the actual output size so no stale bytes
        // remain past the end of the produced tensor.
        output.data.truncate(actual);
        output.data.copy_from_slice(source);
    }

    Ok(())
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
