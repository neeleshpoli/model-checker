use std::{
    ffi::{CStr, NulError},
    ptr::NonNull,
    str::Utf8Error,
};

use ort_genai_sys::{OgaDestroyResult, OgaResult as OgaResultCType, OgaResultGetError};

/// Alias used throughout the crate for FFI call results.
pub type OgaResult<T> = Result<T, Error>;

/// Errors returned by the ONNX Runtime GenAI FFI bindings.
///
/// These errors cover both native API failures and Rust-side validation issues.
/// The native `OgaResult` pointer is always destroyed after extraction, so the
/// conversion path never leaks the underlying C result object.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// A native GenAI call failed and returned an error string.
    #[error("onnxruntime-genai error: {0}")]
    OgaError(String),
    /// A required pointer was null when a non-null value was expected.
    #[error("Null found in parameter!")]
    NulError,
    /// A native string could not be decoded as valid UTF-8.
    #[error("Invalid UTF8: {0}")]
    Utf8Error(Utf8Error),
    /// A tensor element type does not match the requested Rust type.
    #[error("tensor element type does not match the requested Rust type")]
    TensorTypeMismatch,
    /// A tensor had an invalid shape for the operation being performed.
    #[error("invalid tensor shape")]
    InvalidTensorShape,
}

/// Converts a Rust null-pointer trap into the crate's error type.
impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::NulError
    }
}

/// Converts a UTF-8 conversion failure into the crate's error type.
impl From<Utf8Error> for Error {
    fn from(value: Utf8Error) -> Self {
        Self::Utf8Error(value)
    }
}

/// Extension trait for converting native GenAI result pointers into Rust Result values.
///
/// This is the main boundary between the C API and Rust ownership. A non-null native
/// result is interpreted as an error payload, and it is always destroyed before being
/// converted into an `Error::OgaError` value to avoid leaking native memory.
pub(crate) trait OgaResultExt<T> {
    fn to_result(self) -> Result<T, Error>;
}

impl OgaResultExt<()> for *mut OgaResultCType {
    fn to_result(self) -> Result<(), Error> {
        if self.is_null() {
            return Ok(());
        }

        unsafe {
            let err_msg = CStr::from_ptr(OgaResultGetError(self))
                .to_string_lossy()
                .into_owned();

            OgaDestroyResult(self);

            Err(Error::OgaError(err_msg))
        }
    }
}

// Backward-compatible alias for the existing style used across the crate.
impl<T> OgaResultExt<NonNull<T>> for Option<NonNull<T>> {
    fn to_result(self) -> Result<NonNull<T>, Error> {
        self.ok_or(Error::NulError)
    }
}
