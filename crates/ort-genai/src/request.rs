use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaCreateRequest, OgaDestroyRequest, OgaRequest, OgaRequestAddTokens, OgaRequestGetOpaqueData,
    OgaRequestGetUnseenToken, OgaRequestHasUnseenTokens, OgaRequestIsDone, OgaRequestSetOpaqueData,
};

use crate::error::{Result, check_status};
use crate::generator::GeneratorParams;
use crate::sequences::Sequences;

pub struct Request {
    pub(crate) ptr: Arc<Mutex<*mut OgaRequest>>,
}

unsafe impl Send for Request {}
unsafe impl Sync for Request {}

impl Request {
    pub fn new(params: &GeneratorParams) -> Result<Self> {
        let mut ptr: *mut OgaRequest = std::ptr::null_mut();
        let params_ptr = params.ptr.lock()?;
        unsafe {
            check_status(OgaCreateRequest(*params_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub(crate) fn from_ptr(ptr: *mut OgaRequest) -> Self {
        Self {
            ptr: Arc::new(Mutex::new(ptr)),
        }
    }

    pub fn add_tokens(&self, tokens: &Sequences) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let tokens_ptr = tokens.ptr.lock()?;
        unsafe {
            check_status(OgaRequestAddTokens(*ptr, *tokens_ptr))?;
        }
        Ok(())
    }

    pub fn set_opaque_data(&self, data: *mut std::ffi::c_void) -> Result<()> {
        let ptr = self.ptr.lock()?;
        unsafe {
            check_status(OgaRequestSetOpaqueData(*ptr, data))?;
        }
        Ok(())
    }

    pub fn get_opaque_data(&self) -> Result<*mut std::ffi::c_void> {
        let ptr = self.ptr.lock()?;
        let mut out: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            check_status(OgaRequestGetOpaqueData(*ptr, &mut out))?;
        }
        Ok(out)
    }

    pub fn has_unseen_tokens(&self) -> Result<bool> {
        let ptr = self.ptr.lock()?;
        let mut out: bool = false;
        unsafe {
            check_status(OgaRequestHasUnseenTokens(*ptr, &mut out))?;
        }
        Ok(out)
    }

    pub fn get_unseen_token(&self) -> Result<i32> {
        let ptr = self.ptr.lock()?;
        let mut out: i32 = 0;
        unsafe {
            check_status(OgaRequestGetUnseenToken(*ptr, &mut out))?;
        }
        Ok(out)
    }

    pub fn is_done(&self) -> Result<bool> {
        let ptr = self.ptr.lock()?;
        let mut out: bool = false;
        unsafe {
            check_status(OgaRequestIsDone(*ptr, &mut out))?;
        }
        Ok(out)
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyRequest(*ptr);
                }
            }
        }
    }
}
