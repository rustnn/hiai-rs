/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

//! Raw FFI bindings to the Huawei HiAI (CANN) DDK.
//!
//! This crate provides low-level, `unsafe` bindings to a pure-C adapter that
//! wraps the HiAI DDK's C++ API (`hiai::op::*`, the GE graph engine, and the
//! model manager). For a safe, ergonomic API, use the [`hiai-rs`](https://crates.io/crates/hiai-rs)
//! crate instead.
//!
//! # Safety
//!
//! All functions in this crate are `unsafe` as they call directly into C++
//! code and perform no safety checks. Callers must ensure:
//!
//! - Handles are valid and obtained from the correct factory function.
//! - Handles are not used after they are destroyed.
//! - Attribute/input names point to valid NUL-terminated C strings.
//!
//! # Availability
//!
//! On the OpenHarmony target (`aarch64-unknown-linux-ohos`), the build also
//! compiles and links the C++ adapter against `libhiai`; on other targets it
//! only generates the bindings (the adapter headers are self-contained).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

pub mod sys {
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(non_upper_case_globals)]
    #![allow(dead_code)]
    #![allow(clippy::missing_safety_doc)]
    include!(concat!(env!("OUT_DIR"), "/cann_bindings.rs"));
}

pub use sys::*;
