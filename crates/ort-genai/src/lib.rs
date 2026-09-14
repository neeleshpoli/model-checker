pub mod adapters;
pub mod audios;
pub mod config;
pub mod engine;
pub mod error;
pub mod generator;
pub mod images;
pub mod model;
pub mod processor;
pub mod request;
pub mod runtime_settings;
pub mod sequences;
pub mod streaming_processor;
pub mod string_array;
pub mod tensor;
pub mod tokenizer;

pub use adapters::*;
pub use audios::*;
pub use config::*;
pub use engine::*;
pub use error::*;
pub use generator::*;
pub use images::*;
pub use model::*;
pub use processor::*;
pub use request::*;
pub use runtime_settings::*;
pub use sequences::*;
pub use streaming_processor::*;
pub use string_array::*;
pub use tensor::*;
pub use tokenizer::*;

use ort_genai_sys::{
    OgaGetCurrentGpuDeviceId, OgaRegisterExecutionProviderLibrary, OgaSetCurrentGpuDeviceId,
    OgaSetLogBool, OgaSetLogString, OgaSetTelemetryEnabled, OgaShutdown,
    OgaUnregisterExecutionProviderLibrary,
};
use std::ffi::CString;

/// Shuts down the GenAI library and releases all GenAI-owned ONNX Runtime globals.
/// \warning Callers SHOULD invoke OgaShutdown() before process exit. If it is not called, GenAI's globals are destroyed
/// at static-destruction time in undefined order, which may crash.
/// \note C++ callers should prefer the OgaHandle RAII wrapper in ort_genai.h; C# callers should prefer the
/// OgaHandle IDisposable wrapper. Both invoke OgaShutdown() on destruction.
/// \note Lifetime contract: no Model / Generator / Tokenizer / Tensor / Engine / Request, or any
/// object that holds device memory, may outlive OgaShutdown(). The caller MUST destroy every such object before
/// calling OgaShutdown(). Calling OgaShutdown() with such objects still alive is undefined behavior (typically a
/// crash when the buffer is freed through a now-invalid allocator).
/// \note Re-initialization: OgaShutdown() is a full teardown -- it destroys GenAI's ONNX Runtime environment and unloads
/// GenAI's add-on libraries. GenAI may be used again after OgaShutdown(); the next GenAI call re-initializes with a
/// fresh environment.
/// \note If a host registered execution provider libraries directly on its own OrtEnv reference, it should unregister
/// them (on that reference) and release the reference after OgaShutdown(), once all GenAI usage is finished.
/// /
/// Shuts down the GenAI library and releases all GenAI-owned ONNX Runtime globals.
/// \warning Callers SHOULD invoke OgaShutdown() before process exit. If it is not called, GenAI's globals are destroyed
/// at static-destruction time in undefined order, which may crash.
/// \note C++ callers should prefer the OgaHandle RAII wrapper in ort_genai.h; C# callers should prefer the
/// OgaHandle IDisposable wrapper. Both invoke OgaShutdown() on destruction.
/// \note Lifetime contract: no Model / Generator / Tokenizer / Tensor / Engine / Request, or any
/// object that holds device memory, may outlive OgaShutdown(). The caller MUST destroy every such object before
/// calling OgaShutdown(). Calling OgaShutdown() with such objects still alive is undefined behavior (typically a
/// crash when the buffer is freed through a now-invalid allocator).
/// \note Re-initialization: OgaShutdown() is a full teardown -- it destroys GenAI's ONNX Runtime environment and unloads
/// GenAI's add-on libraries. GenAI may be used again after OgaShutdown(); the next GenAI call re-initializes with a
/// fresh environment.
/// \note If a host registered execution provider libraries directly on its own OrtEnv reference, it should unregister
/// them (on that reference) and release the reference after OgaShutdown(), once all GenAI usage is finished.
/// /
/// Shuts down the GenAI library and releases all GenAI-owned ONNX Runtime globals.
/// \warning Callers SHOULD invoke OgaShutdown() before process exit. If it is not called, GenAI's globals are destroyed
/// at static-destruction time in undefined order, which may crash.
/// \note C++ callers should prefer the OgaHandle RAII wrapper in ort_genai.h; C# callers should prefer the
/// OgaHandle IDisposable wrapper. Both invoke OgaShutdown() on destruction.
/// \note Lifetime contract: no Model / Generator / Tokenizer / Tensor / Engine / Request, or any
/// object that holds device memory, may outlive OgaShutdown(). The caller MUST destroy every such object before
/// calling OgaShutdown(). Calling OgaShutdown() with such objects still alive is undefined behavior (typically a
/// crash when the buffer is freed through a now-invalid allocator).
/// \note Re-initialization: OgaShutdown() is a full teardown -- it destroys GenAI's ONNX Runtime environment and unloads
/// GenAI's add-on libraries. GenAI may be used again after OgaShutdown(); the next GenAI call re-initializes with a
/// fresh environment.
/// \note If a host registered execution provider libraries directly on its own OrtEnv reference, it should unregister
/// them (on that reference) and release the reference after OgaShutdown(), once all GenAI usage is finished.
/// /
/// Shuts down the GenAI library and releases all GenAI-owned ONNX Runtime globals.
/// \warning Callers SHOULD invoke OgaShutdown() before process exit. If it is not called, GenAI's globals are destroyed
/// at static-destruction time in undefined order, which may crash.
/// \note C++ callers should prefer the OgaHandle RAII wrapper in ort_genai.h; C# callers should prefer the
/// OgaHandle IDisposable wrapper. Both invoke OgaShutdown() on destruction.
/// \note Lifetime contract: no Model / Generator / Tokenizer / Tensor / Engine / Request, or any
/// object that holds device memory, may outlive OgaShutdown(). The caller MUST destroy every such object before
/// calling OgaShutdown(). Calling OgaShutdown() with such objects still alive is undefined behavior (typically a
/// crash when the buffer is freed through a now-invalid allocator).
/// \note Re-initialization: OgaShutdown() is a full teardown -- it destroys GenAI's ONNX Runtime environment and unloads
/// GenAI's add-on libraries. GenAI may be used again after OgaShutdown(); the next GenAI call re-initializes with a
/// fresh environment.
/// \note If a host registered execution provider libraries directly on its own OrtEnv reference, it should unregister
/// them (on that reference) and release the reference after OgaShutdown(), once all GenAI usage is finished.
/// /
/// Shuts down the GenAI library and releases all GenAI-owned ONNX Runtime globals.
/// \warning Callers SHOULD invoke OgaShutdown() before process exit. If it is not called, GenAI's globals are destroyed
/// at static-destruction time in undefined order, which may crash.
/// \note C++ callers should prefer the OgaHandle RAII wrapper in ort_genai.h; C# callers should prefer the
/// OgaHandle IDisposable wrapper. Both invoke OgaShutdown() on destruction.
/// \note Lifetime contract: no Model / Generator / Tokenizer / Tensor / Engine / Request, or any
/// object that holds device memory, may outlive OgaShutdown(). The caller MUST destroy every such object before
/// calling OgaShutdown(). Calling OgaShutdown() with such objects still alive is undefined behavior (typically a
/// crash when the buffer is freed through a now-invalid allocator).
/// \note Re-initialization: OgaShutdown() is a full teardown -- it destroys GenAI's ONNX Runtime environment and unloads
/// GenAI's add-on libraries. GenAI may be used again after OgaShutdown(); the next GenAI call re-initializes with a
/// fresh environment.
/// \note If a host registered execution provider libraries directly on its own OrtEnv reference, it should unregister
/// them (on that reference) and release the reference after OgaShutdown(), once all GenAI usage is finished.
/// /
pub fn shutdown() {
    unsafe {
        OgaShutdown();
    }
}

/// Enable or disable non-essential telemetry event collection.
/// Telemetry can be fully disabled at compile time (ENABLE_TELEMETRY=OFF) or by setting
/// ORT_DISABLE_TELEMETRY=1 before initialization. The environment variable prevents the uploader,
/// events, and persistent device identifier from being created for the process lifetime. Runtime
/// controls suppress new detailed model, generation, adapter, and error telemetry; the process
/// information event may still be emitted.
/// * enabled true to enable telemetry, false to disable.
/// /
/// Enable or disable non-essential telemetry event collection.
/// Telemetry can be fully disabled at compile time (ENABLE_TELEMETRY=OFF) or by setting
/// ORT_DISABLE_TELEMETRY=1 before initialization. The environment variable prevents the uploader,
/// events, and persistent device identifier from being created for the process lifetime. Runtime
/// controls suppress new detailed model, generation, adapter, and error telemetry; the process
/// information event may still be emitted.
/// * enabled true to enable telemetry, false to disable.
/// /
/// Enable or disable non-essential telemetry event collection.
/// Telemetry can be fully disabled at compile time (ENABLE_TELEMETRY=OFF) or by setting
/// ORT_DISABLE_TELEMETRY=1 before initialization. The environment variable prevents the uploader,
/// events, and persistent device identifier from being created for the process lifetime. Runtime
/// controls suppress new detailed model, generation, adapter, and error telemetry; the process
/// information event may still be emitted.
/// * enabled true to enable telemetry, false to disable.
/// /
/// Enable or disable non-essential telemetry event collection.
/// Telemetry can be fully disabled at compile time (ENABLE_TELEMETRY=OFF) or by setting
/// ORT_DISABLE_TELEMETRY=1 before initialization. The environment variable prevents the uploader,
/// events, and persistent device identifier from being created for the process lifetime. Runtime
/// controls suppress new detailed model, generation, adapter, and error telemetry; the process
/// information event may still be emitted.
/// * enabled true to enable telemetry, false to disable.
/// /
/// Enable or disable non-essential telemetry event collection.
/// Telemetry can be fully disabled at compile time (ENABLE_TELEMETRY=OFF) or by setting
/// ORT_DISABLE_TELEMETRY=1 before initialization. The environment variable prevents the uploader,
/// events, and persistent device identifier from being created for the process lifetime. Runtime
/// controls suppress new detailed model, generation, adapter, and error telemetry; the process
/// information event may still be emitted.
/// * enabled true to enable telemetry, false to disable.
/// /
pub fn set_telemetry_enabled(enabled: bool) {
    unsafe {
        OgaSetTelemetryEnabled(enabled);
    }
}

pub fn set_current_gpu_device_id(device_id: i32) -> error::Result<()> {
    unsafe {
        error::check_status(OgaSetCurrentGpuDeviceId(device_id))?;
    }
    Ok(())
}

pub fn get_current_gpu_device_id() -> error::Result<i32> {
    let mut device_id: i32 = 0;
    unsafe {
        error::check_status(OgaGetCurrentGpuDeviceId(&mut device_id))?;
    }
    Ok(device_id)
}

/// Registers an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// * path provider path.
/// /
/// Registers an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// * path provider path.
/// /
/// Registers an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// * path provider path.
/// /
/// Registers an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// * path provider path.
/// /
/// Registers an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// * path provider path.
/// /
pub fn register_execution_provider_library(registration_name: &str, library_path: &str) {
    if let (Ok(c_name), Ok(c_path)) = (CString::new(registration_name), CString::new(library_path))
    {
        unsafe {
            OgaRegisterExecutionProviderLibrary(c_name.as_ptr(), c_path.as_ptr());
        }
    }
}

/// Unregisters an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// /
/// Unregisters an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// /
/// Unregisters an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// /
/// Unregisters an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// /
/// Unregisters an execution provider library with ONNXRuntime API.
/// * registration_name name for registration.
/// /
pub fn unregister_execution_provider_library(registration_name: &str) {
    if let Ok(c_name) = CString::new(registration_name) {
        unsafe {
            OgaUnregisterExecutionProviderLibrary(c_name.as_ptr());
        }
    }
}

pub fn set_log_bool(name: &str, value: bool) -> error::Result<()> {
    let c_name = CString::new(name)?;
    unsafe {
        error::check_status(OgaSetLogBool(c_name.as_ptr(), value))?;
    }
    Ok(())
}

pub fn set_log_string(name: &str, value: &str) -> error::Result<()> {
    let c_name = CString::new(name)?;
    let c_value = CString::new(value)?;
    unsafe {
        error::check_status(OgaSetLogString(c_name.as_ptr(), c_value.as_ptr()))?;
    }
    Ok(())
}

static mut LOG_CALLBACK: Option<fn(&str, usize)> = None;

unsafe extern "C" fn log_callback_trampoline(string: *const std::ffi::c_char, length: usize) {
    unsafe {
        if let Some(cb) = LOG_CALLBACK {
            if !string.is_null() {
                let c_str = std::ffi::CStr::from_ptr(string);
                if let Ok(rust_str) = c_str.to_str() {
                    cb(rust_str, length);
                }
            }
        }
    }
}

pub fn set_log_callback(callback: Option<fn(&str, usize)>) -> error::Result<()> {
    unsafe {
        LOG_CALLBACK = callback;
        let cb_ptr = if callback.is_some() {
            Some(log_callback_trampoline as unsafe extern "C" fn(*const std::ffi::c_char, usize))
        } else {
            None
        };
        error::check_status(ort_genai_sys::OgaSetLogCallback(cb_ptr))?;
    }
    Ok(())
}
