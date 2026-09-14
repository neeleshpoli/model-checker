use std::ffi::CString;
use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaDestroyImages, OgaImages, OgaLoadImage, OgaLoadImages, OgaLoadImagesFromBuffers,
};

use crate::error::{check_status, Result};
use crate::string_array::StringArray;

pub struct Images {
    pub(crate) ptr: Arc<Mutex<*mut OgaImages>>,
}

unsafe impl Send for Images {}
unsafe impl Sync for Images {}

impl Images {
    pub fn load(image_path: &str) -> Result<Self> {
        let mut ptr: *mut OgaImages = std::ptr::null_mut();
        let c_path = CString::new(image_path)?;
        unsafe {
            check_status(OgaLoadImage(c_path.as_ptr(), &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn load_multiple(image_paths: &StringArray) -> Result<Self> {
        let mut ptr: *mut OgaImages = std::ptr::null_mut();
        let paths_ptr = image_paths.ptr.lock()?;
        unsafe {
            check_status(OgaLoadImages(*paths_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn load_from_buffers(buffers: &[&[u8]]) -> Result<Self> {
        let mut ptr: *mut OgaImages = std::ptr::null_mut();

        let c_ptrs: Vec<*const std::ffi::c_void> = buffers
            .iter()
            .map(|b| b.as_ptr() as *const std::ffi::c_void)
            .collect();
        let c_sizes: Vec<usize> = buffers.iter().map(|b| b.len()).collect();

        unsafe {
            check_status(OgaLoadImagesFromBuffers(
                c_ptrs.as_ptr() as *mut _,
                c_sizes.as_ptr(),
                buffers.len(),
                &mut ptr,
            ))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }
}

impl Drop for Images {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyImages(*ptr);
                }
            }
        }
    }
}
