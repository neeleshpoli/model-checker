use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaCreateModel, OgaCreateModelFromConfig, OgaCreateModelWithRuntimeSettings, OgaDestroyModel,
    OgaModel, OgaModelGetDeviceType, OgaModelGetType,
};

use crate::config::Config;
use crate::error::{check_status, Result};
use crate::runtime_settings::RuntimeSettings;

pub struct Model {
    pub(crate) ptr: Arc<Mutex<*mut OgaModel>>,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn new(config_path: &str) -> Result<Self> {
        let mut ptr: *mut OgaModel = std::ptr::null_mut();
        let c_path = CString::new(config_path)?;
        unsafe {
            check_status(OgaCreateModel(c_path.as_ptr(), &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn from_config(config: &Config) -> Result<Self> {
        let mut ptr: *mut OgaModel = std::ptr::null_mut();
        let config_ptr = config.ptr.lock()?;
        unsafe {
            check_status(OgaCreateModelFromConfig(*config_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn with_runtime_settings(config_path: &str, settings: &RuntimeSettings) -> Result<Self> {
        let mut ptr: *mut OgaModel = std::ptr::null_mut();
        let c_path = CString::new(config_path)?;
        let settings_ptr = settings.ptr.lock()?;
        unsafe {
            check_status(OgaCreateModelWithRuntimeSettings(
                c_path.as_ptr(),
                *settings_ptr,
                &mut ptr,
            ))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn get_type(&self) -> Result<String> {
        let ptr = self.ptr.lock()?;
        let mut type_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaModelGetType(*ptr, &mut type_ptr))?;
            let c_str = CStr::from_ptr(type_ptr);
            Ok(c_str.to_string_lossy().into_owned())
        }
    }

    pub fn get_device_type(&self) -> Result<String> {
        let ptr = self.ptr.lock()?;
        let mut device_type_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaModelGetDeviceType(*ptr, &mut device_type_ptr))?;
            let c_str = CStr::from_ptr(device_type_ptr);
            Ok(c_str.to_string_lossy().into_owned())
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyModel(*ptr);
                }
            }
        }
    }
}
