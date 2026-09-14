use std::ffi::CString;
use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaAdapters, OgaCreateAdapters, OgaDestroyAdapters, OgaLoadAdapter, OgaUnloadAdapter,
};

use crate::error::{check_status, Result};
use crate::model::Model;

pub struct Adapters {
    pub(crate) ptr: Arc<Mutex<*mut OgaAdapters>>,
}

unsafe impl Send for Adapters {}
unsafe impl Sync for Adapters {}

impl Adapters {
    pub fn new(model: &Model) -> Result<Self> {
        let mut ptr: *mut OgaAdapters = std::ptr::null_mut();
        let model_ptr = model.ptr.lock()?;
        unsafe {
            check_status(OgaCreateAdapters(*model_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn load_adapter(&self, adapter_file_path: &str, adapter_name: &str) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_path = CString::new(adapter_file_path)?;
        let c_name = CString::new(adapter_name)?;
        unsafe {
            check_status(OgaLoadAdapter(*ptr, c_path.as_ptr(), c_name.as_ptr()))?;
        }
        Ok(())
    }

    pub fn unload_adapter(&self, adapter_name: &str) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(adapter_name)?;
        unsafe {
            check_status(OgaUnloadAdapter(*ptr, c_name.as_ptr()))?;
        }
        Ok(())
    }
}

impl Drop for Adapters {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyAdapters(*ptr);
                }
            }
        }
    }
}

impl Adapters {
    pub fn set_active_adapter(
        &self,
        generator: &crate::generator::Generator,
        adapter_name: &str,
    ) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let gen_ptr = generator.ptr.lock()?;
        let c_name = CString::new(adapter_name)?;
        unsafe {
            check_status(ort_genai_sys::OgaSetActiveAdapter(
                *gen_ptr,
                *ptr,
                c_name.as_ptr(),
            ))?;
        }
        Ok(())
    }
}
