use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaCreateEngine, OgaDestroyEngine, OgaEngine, OgaEngineAddRequest, OgaEngineHasPendingRequests,
    OgaEngineRemoveRequest, OgaEngineStep,
};

use crate::error::{Result, check_status};
use crate::model::Model;
use crate::request::Request;

pub struct Engine {
    pub(crate) ptr: Arc<Mutex<*mut OgaEngine>>,
}

unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    pub fn new(model: &Model) -> Result<Self> {
        let mut ptr: *mut OgaEngine = std::ptr::null_mut();
        let model_ptr = model.ptr.lock()?;
        unsafe {
            check_status(OgaCreateEngine(*model_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn step(&self) -> Result<Request> {
        let ptr = self.ptr.lock()?;
        let mut request_ptr: *mut ort_genai_sys::OgaRequest = std::ptr::null_mut();
        unsafe {
            check_status(OgaEngineStep(*ptr, &mut request_ptr))?;
        }
        Ok(Request::from_ptr(request_ptr))
    }

    pub fn has_pending_requests(&self) -> Result<bool> {
        let ptr = self.ptr.lock()?;
        let mut out: bool = false;
        unsafe {
            check_status(OgaEngineHasPendingRequests(*ptr, &mut out))?;
        }
        Ok(out)
    }

    pub fn add_request(&self, request: &Request) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let req_ptr = request.ptr.lock()?;
        unsafe {
            check_status(OgaEngineAddRequest(*ptr, *req_ptr))?;
        }
        Ok(())
    }

    pub fn remove_request(&self, request: &Request) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let req_ptr = request.ptr.lock()?;
        unsafe {
            check_status(OgaEngineRemoveRequest(*ptr, *req_ptr))?;
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyEngine(*ptr);
                }
            }
        }
    }
}
