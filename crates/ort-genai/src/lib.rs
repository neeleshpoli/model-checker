use std::{ffi::CString, mem::MaybeUninit};

use ort_genai_sys::{
    OgaGetCurrentGpuDeviceId, OgaRegisterExecutionProviderLibrary, OgaSetCurrentGpuDeviceId,
    OgaSetTelemetryEnabled, OgaShutdown, OgaUnregisterExecutionProviderLibrary,
};

use crate::error::{OgaResult, OgaResultExt};

mod adapter;
mod audios;
mod config;
mod element_type;
mod engine;
mod error;
mod generator;
mod generator_params;
mod images;
mod model;
mod multi_modal_processor;
mod named_tensors;
mod request;
mod runtime_settings;
mod sequences;
mod streaming_processor;
mod string_array;
mod tensor;
mod tokenizer;
mod tokenizer_stream;

pub use adapter::*;
pub use audios::*;
pub use config::*;
pub use element_type::*;
pub use engine::*;
pub use error::*;
pub use generator::*;
pub use generator_params::*;
pub use images::*;
pub use model::*;
pub use multi_modal_processor::*;
pub use named_tensors::*;
pub use request::*;
pub use runtime_settings::*;
pub use sequences::*;
pub use streaming_processor::*;
pub use tensor::*;
pub use tokenizer::*;
pub use tokenizer_stream::*;

#[macro_export]
/// Implements `Drop` for an FFI-backed wrapper.
///
/// The generated implementation calls the supplied native destruction
/// function with the wrapper's internal pointer.
macro_rules! impl_drop {
    ($struct_name:ident, $destroy_fn:ident) => {
        impl Drop for $struct_name {
            fn drop(&mut self) {
                unsafe {
                    $destroy_fn(self.ptr.as_ptr());
                }
            }
        }
    };
    ($struct_name:ident<$($generic:tt),+>, $destroy_fn:ident) => {
        impl<$($generic),+> Drop for $struct_name<$($generic),+> {
            fn drop(&mut self) {
                unsafe {
                    $destroy_fn(self.ptr.as_ptr());
                }
            }
        }
    };
}

#[macro_export]
/// Implements conversions from an FFI-backed wrapper to native pointers.
///
/// The generated implementations expose the wrapper pointer as either a
/// const or mutable pointer, depending on whether the wrapper is borrowed
/// immutably or mutably.
macro_rules! impl_into_ptr {
    ($struct_name: ident, $c_type: ident) => {
        impl Into<*const $c_type> for &$struct_name {
            #[inline]
            fn into(self) -> *const $c_type {
                self.ptr.as_ptr()
            }
        }

        impl Into<*const $c_type> for $struct_name {
            #[inline]
            fn into(self) -> *const $c_type {
                self.ptr.as_ptr()
            }
        }

        impl Into<*mut $c_type> for &mut $struct_name {
            #[inline]
            fn into(self) -> *mut $c_type {
                self.ptr.as_ptr()
            }
        }

        impl Into<*mut $c_type> for $struct_name {
            #[inline]
            fn into(self) -> *mut $c_type {
                self.ptr.as_ptr()
            }
        }
    };
    ($struct_name:ident<$($generic:tt),+>, $c_type:ident) => {
        impl<$($generic),+> Into<*const $c_type> for &$struct_name<$($generic),+> {
            #[inline]
            fn into(self) -> *const $c_type {
                self.ptr.as_ptr()
            }
        }

        impl<$($generic),+> Into<*const $c_type> for $struct_name<$($generic),+> {
            #[inline]
            fn into(self) -> *const $c_type {
                self.ptr.as_ptr()
            }
        }

        impl<$($generic),+> Into<*mut $c_type> for &mut $struct_name<$($generic),+> {
            #[inline]
            fn into(self) -> *mut $c_type {
                self.ptr.as_ptr()
            }
        }

        impl<$($generic),+> Into<*mut $c_type> for $struct_name<$($generic),+> {
            #[inline]
            fn into(self) -> *mut $c_type {
                self.ptr.as_ptr()
            }
        }
    };
}

#[macro_export]
/// Implements `TryFrom<*mut T>` for an FFI-backed wrapper.
///
/// A null native pointer is converted into the crate's error type.
macro_rules! impl_from_ptr {
    ($struct_name:ident, $c_type:ident) => {
        impl TryFrom<*mut $c_type> for $struct_name {
            type Error = $crate::error::Error;

            fn try_from(value: *mut $c_type) -> Result<Self, Self::Error> {
                let ptr =
                    <Option<std::ptr::NonNull<$c_type>> as $crate::error::OgaResultExt<_>>::to_result(
                        std::ptr::NonNull::new(value),
                    )?;

                Ok(Self {
                    ptr,
                    _marker: std::marker::PhantomData,
                })
            }
        }
    };
    ($struct_name:ident<$($generic:tt),+>, $c_type:ident) => {
        impl<$($generic),+> TryFrom<*mut $c_type> for $struct_name<$($generic),+> {
            type Error = $crate::error::Error;

            fn try_from(value: *mut $c_type) -> Result<Self, Self::Error> {
                let ptr =
                    <Option<std::ptr::NonNull<$c_type>> as $crate::error::OgaResultExt<_>>::to_result(
                        std::ptr::NonNull::new(value),
                    )?;

                Ok(Self {
                    ptr,
                    _type: std::marker::PhantomData,
                    _marker: std::marker::PhantomData,
                })
            }
        }
    };
}

/// Shuts down the GenAI library and releases its ONNX Runtime globals.
///
/// All GenAI objects must be destroyed before calling this function. In
/// particular, no [`Model`], [`Generator`], [`Tokenizer`], [`Tensor`],
/// [`Engine`], or [`Request`] may outlive the shutdown call. Calling this
/// function while such objects are alive is undefined behavior.
///
/// The library can be initialized again by a later GenAI call after shutdown.
pub unsafe fn shutdown() {
    unsafe {
        OgaShutdown();
    }
}

/// Enables or disables non-essential telemetry event collection.
///
/// # Parameters
/// * `enabled` - `true` to enable telemetry, or `false` to disable it.
pub fn telemetry_enabled(enabled: bool) {
    unsafe {
        OgaSetTelemetryEnabled(enabled);
    }
}

/// Sets the current GPU device ID used by the library.
///
/// # Parameters
/// * `device_id` - The GPU device ID to select.
pub fn set_current_gpu_device_id(device_id: i32) -> OgaResult<()> {
    unsafe { OgaSetCurrentGpuDeviceId(device_id) }.to_result()
}

/// Returns the current GPU device ID used by the library.
pub fn get_current_gpu_device_id() -> OgaResult<i32> {
    let mut id = MaybeUninit::uninit();
    unsafe { OgaGetCurrentGpuDeviceId(id.as_mut_ptr()) }.to_result()?;

    Ok(unsafe { id.assume_init() })
}

/// Registers an execution-provider library with ONNX Runtime.
///
/// # Parameters
/// * `registration_name` - The name used to register the provider library.
/// * `library_path` - The path to the provider library.
pub fn register_execution_provider_library(registration_name: &str, library_path: &str) {
    let registration_name = CString::new(registration_name).expect("Registration name has nul!");
    let library_path = CString::new(library_path).expect("Library path has null!");

    unsafe {
        OgaRegisterExecutionProviderLibrary(registration_name.as_ptr(), library_path.as_ptr())
    }
}

/// Unregisters an execution-provider library from ONNX Runtime.
///
/// # Parameters
/// * `registration_name` - The name used when the provider library was registered.
pub fn unregister_execution_provider_library(registration_name: &str) {
    let registration_name = CString::new(registration_name).expect("Registration name has nul!");

    unsafe { OgaUnregisterExecutionProviderLibrary(registration_name.as_ptr()) }
}
