/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types for HiAI (CANN) DDK operations.

use std::fmt;
use thiserror::Error;

/// Result type for HiAI DDK operations.
pub type Result<T> = std::result::Result<T, Error>;

/// A status code returned by the HiAI DDK (the adapter's `CannStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CannStatus(i32);

impl CannStatus {
    /// Wraps a raw status code.
    pub fn from_code(code: i32) -> Self {
        Self(code)
    }

    /// The raw status code.
    pub fn code(self) -> i32 {
        self.0
    }

    /// The symbolic name of a known status code, or `None` if unknown.
    pub fn name(self) -> Option<&'static str> {
        match self.0 {
            0 => Some("kSuccess"),
            1 => Some("kFailed"),
            2 => Some("kNotInit"),
            3 => Some("kInvalidPara"),
            7 => Some("kInvalidApi"),
            8 => Some("kInvalidPtr"),
            _ => None,
        }
    }
}

impl fmt::Display for CannStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(f, "{name} ({})", self.0),
            None => write!(f, "unknown status ({})", self.0),
        }
    }
}

/// Errors that can occur when using the HiAI DDK.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// A CANN call returned a non-success status (the CANN error itself).
    #[error("{operation} failed with {status}")]
    Cann {
        operation: &'static str,
        status: CannStatus,
    },

    /// A CANN factory returned a null handle — no status code is produced, so
    /// this complements [`Error::Cann`].
    #[error("failed to create {what}: null handle")]
    Null { what: &'static str },

    /// A value exceeds the DDK's 32-bit size limit (a Rust-side guard).
    #[error("{what} exceeds u32::MAX (4 GiB)")]
    TooLarge { what: &'static str },

    /// A count exceeds the DDK's 32-bit signed limit (a Rust-side guard).
    #[error("{what} exceeds i32::MAX")]
    TooMany { what: &'static str },

    /// The output tensor buffer returned by the DDK is null.
    #[error("output tensor buffer is null")]
    OutputBufferNull,

    /// The caller's output buffer is too small for the tensor.
    #[error("output buffer is too small")]
    OutputBufferTooSmall,

    /// The `TensorDesc.dtype` value is not a known CANN data-type code.
    #[error("invalid CANN data type: {value}")]
    InvalidDataType { value: i32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::Cann {
            operation: "load model",
            status: CannStatus::from_code(2),
        };
        assert_eq!(err.to_string(), "load model failed with kNotInit (2)");
    }
}
