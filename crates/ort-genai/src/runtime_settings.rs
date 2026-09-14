use std::ffi::CString;
use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaCreateRuntimeSettings, OgaDestroyRuntimeSettings, OgaRuntimeSettings,
    OgaRuntimeSettingsSetHandle,
};

use crate::error::{check_status, Result};

pub struct RuntimeSettings {
    pub(crate) ptr: Arc<Mutex<*mut OgaRuntimeSettings>>,
}

unsafe impl Send for RuntimeSettings {}
unsafe impl Sync for RuntimeSettings {}

impl RuntimeSettings {
    pub fn new() -> Result<Self> {
        let mut ptr: *mut OgaRuntimeSettings = std::ptr::null_mut();
        unsafe {
            check_status(OgaCreateRuntimeSettings(&mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn set_handle(&self, handle_name: &str, handle: usize) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(handle_name)?;
        unsafe {
            check_status(OgaRuntimeSettingsSetHandle(
                *ptr,
                c_name.as_ptr(),
                handle as *mut std::ffi::c_void,
            ))?;
        }
        Ok(())
    }
}

impl Drop for RuntimeSettings {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyRuntimeSettings(*ptr);
                }
            }
        }
    }
}
