use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaCreateMultiModalProcessor, OgaCreateTokenizerStreamFromProcessor,
    OgaDestroyMultiModalProcessor, OgaDestroyString, OgaMultiModalProcessor, OgaProcessorDecode,
    OgaProcessorProcessAudios, OgaProcessorProcessAudiosAndPrompts, OgaProcessorProcessImages,
    OgaProcessorProcessImagesAndAudios, OgaProcessorProcessImagesAndAudiosAndPrompts,
    OgaProcessorProcessImagesAndPrompts,
};

use crate::error::{check_status, Result};
use crate::images::Images;
use crate::model::Model;
use crate::string_array::StringArray;
use crate::tensor::NamedTensors;
use crate::tokenizer::TokenizerStream;

// Assuming Audios is imported here
use crate::audios::Audios;

pub struct MultiModalProcessor {
    pub(crate) ptr: Arc<Mutex<*mut OgaMultiModalProcessor>>,
}

unsafe impl Send for MultiModalProcessor {}
unsafe impl Sync for MultiModalProcessor {}

impl MultiModalProcessor {
    pub fn new(model: &Model) -> Result<Self> {
        let mut ptr: *mut OgaMultiModalProcessor = std::ptr::null_mut();
        let model_ptr = model.ptr.lock()?;
        unsafe {
            check_status(OgaCreateMultiModalProcessor(*model_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn process_images(&self, prompt: &str, images: &Images) -> Result<NamedTensors> {
        let ptr = self.ptr.lock()?;
        let c_prompt = CString::new(prompt)?;
        let images_ptr = images.ptr.lock()?;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaProcessorProcessImages(
                *ptr,
                c_prompt.as_ptr(),
                *images_ptr,
                &mut out_ptr,
            ))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn process_images_and_prompts(
        &self,
        prompts: &StringArray,
        images: &Images,
    ) -> Result<NamedTensors> {
        let ptr = self.ptr.lock()?;
        let prompts_ptr = prompts.ptr.lock()?;
        let images_ptr = images.ptr.lock()?;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaProcessorProcessImagesAndPrompts(
                *ptr,
                *prompts_ptr,
                *images_ptr,
                &mut out_ptr,
            ))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn process_audios(&self, prompt: &str, audios: &Audios) -> Result<NamedTensors> {
        let ptr = self.ptr.lock()?;
        let c_prompt = CString::new(prompt)?;
        let audios_ptr = audios.ptr.lock()?;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaProcessorProcessAudios(
                *ptr,
                c_prompt.as_ptr(),
                *audios_ptr,
                &mut out_ptr,
            ))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn process_audios_and_prompts(
        &self,
        prompts: &StringArray,
        audios: &Audios,
    ) -> Result<NamedTensors> {
        let ptr = self.ptr.lock()?;
        let prompts_ptr = prompts.ptr.lock()?;
        let audios_ptr = audios.ptr.lock()?;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaProcessorProcessAudiosAndPrompts(
                *ptr,
                *prompts_ptr,
                *audios_ptr,
                &mut out_ptr,
            ))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn process_images_and_audios(
        &self,
        prompt: &str,
        images: &Images,
        audios: &Audios,
    ) -> Result<NamedTensors> {
        let ptr = self.ptr.lock()?;
        let c_prompt = CString::new(prompt)?;
        let images_ptr = images.ptr.lock()?;
        let audios_ptr = audios.ptr.lock()?;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaProcessorProcessImagesAndAudios(
                *ptr,
                c_prompt.as_ptr(),
                *images_ptr,
                *audios_ptr,
                &mut out_ptr,
            ))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn process_images_and_audios_and_prompts(
        &self,
        prompts: &StringArray,
        images: &Images,
        audios: &Audios,
    ) -> Result<NamedTensors> {
        let ptr = self.ptr.lock()?;
        let prompts_ptr = prompts.ptr.lock()?;
        let images_ptr = images.ptr.lock()?;
        let audios_ptr = audios.ptr.lock()?;
        let mut out_ptr: *mut ort_genai_sys::OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaProcessorProcessImagesAndAudiosAndPrompts(
                *ptr,
                *prompts_ptr,
                *images_ptr,
                *audios_ptr,
                &mut out_ptr,
            ))?;
        }
        Ok(NamedTensors::from_ptr(out_ptr))
    }

    pub fn decode(&self, tokens: &[i32]) -> Result<String> {
        let ptr = self.ptr.lock()?;
        let mut out_str_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaProcessorDecode(
                *ptr,
                tokens.as_ptr(),
                tokens.len(),
                &mut out_str_ptr,
            ))?;
            let c_str = CStr::from_ptr(out_str_ptr);
            let s = c_str.to_string_lossy().into_owned();
            OgaDestroyString(out_str_ptr);
            Ok(s)
        }
    }

    pub fn create_stream(&self) -> Result<TokenizerStream> {
        let ptr = self.ptr.lock()?;
        let mut stream_ptr: *mut ort_genai_sys::OgaTokenizerStream = std::ptr::null_mut();
        unsafe {
            check_status(OgaCreateTokenizerStreamFromProcessor(*ptr, &mut stream_ptr))?;
        }
        Ok(TokenizerStream {
            ptr: Arc::new(Mutex::new(stream_ptr)),
        })
    }
}

impl Drop for MultiModalProcessor {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyMultiModalProcessor(*ptr);
                }
            }
        }
    }
}
