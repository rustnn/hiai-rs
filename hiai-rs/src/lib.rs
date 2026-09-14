/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

//! Safe Rust bindings to the Huawei HiAI (CANN) DDK.
//!
//! This crate is in early experimental development. The API is unstable and
//! will change. This is NOT production-ready software. Use at your own risk.
//!
//! This crate provides a safe, ergonomic Rust API for running prebuilt neural
//! network graphs on Huawei Ascend NPUs from OpenHarmony, via the HiAI (CANN)
//! DDK.
//!
//! # Overview
//!
//! Graph construction (WebNN → CANN IR) is handled by
//! [rustnn](https://github.com/rustnn/rustnn). This crate covers the runtime
//! side: it loads an offline model and dispatches inference on the NPU.
//!
//! # Example
//!
//! ```rust,no_run
//! use hiai_rs::{InputDesc, OutputDesc, dispatch};
//!
//! # fn main() -> Result<(), hiai_rs::Error> {
//! // Prebuilt offline model bytes (produced by the rustnn CANN converter).
//! let model_bytes: &[u8] = todo!();
//!
//! let input_data = [1.0f32; 4];
//! let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();
//! let inputs = vec![InputDesc {
//!     data: &input_bytes,
//!     shape: vec![1, 4],
//!     dtype: 0, // CANN_DT_FLOAT
//! }];
//!
//! let mut output_data = vec![0u8; 16];
//! let mut outputs = vec![OutputDesc {
//!     data: &mut output_data,
//!     shape: vec![1, 4],
//!     dtype: 0,
//!     actual_len: 0,
//! }];
//!
//! dispatch(model_bytes, &inputs, &mut outputs)?;
//! # Ok(())
//! # }
//! ```

mod dispatch;
mod error;

pub use dispatch::{InputDesc, OutputDesc, Session, dispatch};
pub use error::{CannStatus, Error, Result};

// Re-export the raw bindings for advanced use (e.g. graph building).
pub use hiai_rs_sys as sys;
