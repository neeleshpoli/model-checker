use std::ffi::{CStr, CString};

use ort_genai_sys::{
    OgaCreateStreamingProcessor, OgaDestroyStreamingProcessor, OgaStreamingProcessor,
    OgaStreamingProcessorFlush, OgaStreamingProcessorGetOption, OgaStreamingProcessorProcess,
    OgaStreamingProcessorSetOption,
};

use crate::error::{check_status, Result};
use crate::model::Model;
use crate::tensor::NamedTensors;

pub struct StreamingProcessor {
    pub(crate) ptr: *mut OgaStreamingProcessor,
}

impl StreamingProcessor {
    pub fn new(model: &Model) -> Result<Self> {
        let mut ptr: *mut OgaStreamingProcessor = std::ptr::null_mut();
        let model_ptr = model.ptr;
        unsafe {
            check_status(OgaCreateStreamingProcessor(model_ptr, &mut ptr))?;
        }
        Ok(Self { ptr: ptr })
    }

    pub fn process(&self, audio_data: &[f32]) -> Result<NamedTensors<'_>> {
        let ptr = self.ptr;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaStreamingProcessorProcess(
                ptr,
                audio_data.as_ptr(),
                audio_data.len(),
                &mut out_ptr,
            ))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn flush(&self) -> Result<NamedTensors<'_>> {
        let ptr = self.ptr;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaStreamingProcessorFlush(ptr, &mut out_ptr))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn set_option(&self, key: &str, value: &str) -> Result<()> {
        let ptr = self.ptr;
        let c_key = CString::new(key)?;
        let c_value = CString::new(value)?;
        unsafe {
            check_status(OgaStreamingProcessorSetOption(
                ptr,
                c_key.as_ptr(),
                c_value.as_ptr(),
            ))?;
        }
        Ok(())
    }

    pub fn get_option(&self, key: &str) -> Result<String> {
        let ptr = self.ptr;
        let c_key = CString::new(key)?;
        let mut value_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaStreamingProcessorGetOption(
                ptr,
                c_key.as_ptr(),
                &mut value_ptr,
            ))?;
            let c_str = CStr::from_ptr(value_ptr);
            let s = c_str.to_string_lossy().into_owned();
            ort_genai_sys::OgaDestroyString(value_ptr);
            Ok(s)
        }
    }
}

impl Drop for StreamingProcessor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                OgaDestroyStreamingProcessor(self.ptr);
            }
        }
    }
}
