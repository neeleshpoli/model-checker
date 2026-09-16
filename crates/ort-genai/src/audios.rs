use std::ffi::CString;

use ort_genai_sys::{
    OgaAudios, OgaDestroyAudios, OgaLoadAudio, OgaLoadAudios, OgaLoadAudiosFromBuffers,
};

use crate::error::{check_status, Result};
use crate::string_array::StringArray;

pub struct Audios {
    pub(crate) ptr: *mut OgaAudios,
}

impl Audios {
    pub fn load(audio_path: &str) -> Result<Self> {
        let mut ptr: *mut OgaAudios = std::ptr::null_mut();
        let c_path = CString::new(audio_path)?;
        unsafe {
            check_status(OgaLoadAudio(c_path.as_ptr(), &mut ptr))?;
        }
        Ok(Self { ptr: ptr })
    }

    pub fn load_multiple(audio_paths: &StringArray) -> Result<Self> {
        let mut ptr: *mut OgaAudios = std::ptr::null_mut();
        let paths_ptr = audio_paths.ptr;
        unsafe {
            check_status(OgaLoadAudios(paths_ptr, &mut ptr))?;
        }
        Ok(Self { ptr: ptr })
    }

    pub fn load_from_buffers(buffers: &[&[u8]]) -> Result<Self> {
        let mut ptr: *mut OgaAudios = std::ptr::null_mut();

        let c_ptrs: Vec<*const std::ffi::c_void> = buffers
            .iter()
            .map(|b| b.as_ptr() as *const std::ffi::c_void)
            .collect();
        let c_sizes: Vec<usize> = buffers.iter().map(|b| b.len()).collect();

        unsafe {
            check_status(OgaLoadAudiosFromBuffers(
                c_ptrs.as_ptr() as *mut _,
                c_sizes.as_ptr(),
                buffers.len(),
                &mut ptr,
            ))?;
        }
        Ok(Self { ptr: ptr })
    }
}

impl Drop for Audios {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                OgaDestroyAudios(self.ptr);
            }
        }
    }
}
