use std::ffi::{CStr, CString};

use ort_genai_sys::{
    OgaCreateModel, OgaCreateModelFromConfig, OgaCreateModelWithRuntimeSettings, OgaDestroyModel,
    OgaModel, OgaModelGetDeviceType, OgaModelGetType,
};

use crate::config::Config;
use crate::error::{check_status, Result};
use crate::runtime_settings::RuntimeSettings;

pub struct Model {
    pub(crate) ptr: *mut OgaModel,
}

impl Model {
    pub fn new(config_path: &str) -> Result<Self> {
        let mut ptr: *mut OgaModel = std::ptr::null_mut();
        let c_path = CString::new(config_path)?;
        unsafe {
            check_status(OgaCreateModel(c_path.as_ptr(), &mut ptr))?;
        }
        Ok(Self { ptr: ptr })
    }

    pub fn from_config(config: &Config) -> Result<Self> {
        let mut ptr: *mut OgaModel = std::ptr::null_mut();
        let config_ptr = config.ptr;
        unsafe {
            check_status(OgaCreateModelFromConfig(config_ptr, &mut ptr))?;
        }
        Ok(Self { ptr: ptr })
    }

    pub fn with_runtime_settings(config_path: &str, settings: &RuntimeSettings) -> Result<Self> {
        let mut ptr: *mut OgaModel = std::ptr::null_mut();
        let c_path = CString::new(config_path)?;
        let settings_ptr = settings.ptr;
        unsafe {
            check_status(OgaCreateModelWithRuntimeSettings(
                c_path.as_ptr(),
                settings_ptr,
                &mut ptr,
            ))?;
        }
        Ok(Self { ptr: ptr })
    }

    pub fn get_type(&self) -> Result<String> {
        let ptr = self.ptr;
        let mut type_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaModelGetType(ptr, &mut type_ptr))?;
            if type_ptr.is_null() {
                return Ok(String::new());
            }
            let c_str = CStr::from_ptr(type_ptr);
            let s = c_str.to_string_lossy().into_owned();
            ort_genai_sys::OgaDestroyString(type_ptr);
            Ok(s)
        }
    }

    pub fn get_device_type(&self) -> Result<String> {
        let ptr = self.ptr;
        let mut device_type_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaModelGetDeviceType(ptr, &mut device_type_ptr))?;
            if device_type_ptr.is_null() {
                return Ok(String::new());
            }
            let c_str = CStr::from_ptr(device_type_ptr);
            let s = c_str.to_string_lossy().into_owned();
            ort_genai_sys::OgaDestroyString(device_type_ptr);
            Ok(s)
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                OgaDestroyModel(self.ptr);
            }
        }
    }
}
