use std::ffi::CStr;

use ort_genai_sys::{OgaDestroyResult, OgaResult, OgaResultGetError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("ONNX Runtime GenAI error: {0}")]
    OgaError(String),
    #[error("Null pointer error")]
    NullPointer,
    #[error("String conversion error: {0}")]
    StringConversion(#[from] std::ffi::NulError),
    #[error("UTF-8 conversion error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
    #[error("Mutex lock poisoned")]
    PoisonError,
}

pub type Result<T> = std::result::Result<T, Error>;

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Error::PoisonError
    }
}

pub(crate) fn check_status(result: *mut OgaResult) -> Result<()> {
    if result.is_null() {
        return Ok(());
    }

    let error_msg = unsafe {
        let msg_ptr = OgaResultGetError(result);
        if msg_ptr.is_null() {
            "Unknown error".to_string()
        } else {
            CStr::from_ptr(msg_ptr).to_string_lossy().into_owned()
        }
    };

    unsafe {
        OgaDestroyResult(result);
    }

    Err(Error::OgaError(error_msg))
}
