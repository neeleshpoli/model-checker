use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaCreateStreamingProcessor, OgaDestroyStreamingProcessor, OgaStreamingProcessor,
    OgaStreamingProcessorFlush, OgaStreamingProcessorGetOption, OgaStreamingProcessorProcess,
    OgaStreamingProcessorSetOption,
};

use crate::error::{check_status, Result};
use crate::model::Model;
use crate::tensor::NamedTensors;

pub struct StreamingProcessor {
    pub(crate) ptr: Arc<Mutex<*mut OgaStreamingProcessor>>,
}

unsafe impl Send for StreamingProcessor {}
unsafe impl Sync for StreamingProcessor {}

impl StreamingProcessor {
    pub fn new(model: &Model) -> Result<Self> {
        let mut ptr: *mut OgaStreamingProcessor = std::ptr::null_mut();
        let model_ptr = model.ptr.lock()?;
        unsafe {
            check_status(OgaCreateStreamingProcessor(*model_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn process(&self, audio_data: &[f32]) -> Result<NamedTensors> {
        let ptr = self.ptr.lock()?;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaStreamingProcessorProcess(
                *ptr,
                audio_data.as_ptr(),
                audio_data.len(),
                &mut out_ptr,
            ))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn flush(&self) -> Result<NamedTensors> {
        let ptr = self.ptr.lock()?;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaStreamingProcessorFlush(*ptr, &mut out_ptr))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn set_option(&self, key: &str, value: &str) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_key = CString::new(key)?;
        let c_value = CString::new(value)?;
        unsafe {
            check_status(OgaStreamingProcessorSetOption(
                *ptr,
                c_key.as_ptr(),
                c_value.as_ptr(),
            ))?;
        }
        Ok(())
    }

    pub fn get_option(&self, key: &str) -> Result<String> {
        let ptr = self.ptr.lock()?;
        let c_key = CString::new(key)?;
        let mut value_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaStreamingProcessorGetOption(
                *ptr,
                c_key.as_ptr(),
                &mut value_ptr,
            ))?;
            let c_str = CStr::from_ptr(value_ptr);
            Ok(c_str.to_string_lossy().into_owned())
        }
    }
}

impl Drop for StreamingProcessor {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyStreamingProcessor(*ptr);
                }
            }
        }
    }
}
