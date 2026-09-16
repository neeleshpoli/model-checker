use std::ffi::{CStr, CString};

use ort_genai_sys::{
    OgaCreateStringArray, OgaCreateStringArrayFromStrings, OgaDestroyStringArray, OgaStringArray,
    OgaStringArrayAddString, OgaStringArrayGetCount, OgaStringArrayGetString,
};

use crate::error::{check_status, Result};

pub struct StringArray {
    pub(crate) ptr: *mut OgaStringArray,
}

impl StringArray {
    pub fn new() -> Result<Self> {
        let mut ptr: *mut OgaStringArray = std::ptr::null_mut();
        unsafe {
            check_status(OgaCreateStringArray(&mut ptr))?;
        }
        Ok(Self { ptr: ptr })
    }

    pub fn from_strings(strings: &[&str]) -> Result<Self> {
        let mut ptr: *mut OgaStringArray = std::ptr::null_mut();

        let c_strings: Vec<CString> = strings
            .iter()
            .map(|s| CString::new(*s).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        let c_ptrs: Vec<*const std::ffi::c_char> = c_strings.iter().map(|cs| cs.as_ptr()).collect();

        unsafe {
            check_status(OgaCreateStringArrayFromStrings(
                c_ptrs.as_ptr(),
                c_ptrs.len(),
                &mut ptr,
            ))?;
        }

        Ok(Self { ptr: ptr })
    }

    pub fn add(&self, string: &str) -> Result<()> {
        let ptr = self.ptr;
        let c_string = CString::new(string)?;
        unsafe {
            check_status(OgaStringArrayAddString(ptr, c_string.as_ptr()))?;
        }
        Ok(())
    }

    pub fn len(&self) -> Result<usize> {
        let ptr = self.ptr;
        let mut count: usize = 0;
        unsafe {
            check_status(OgaStringArrayGetCount(ptr, &mut count))?;
        }
        Ok(count)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn get(&self, index: usize) -> Result<String> {
        let ptr = self.ptr;
        let mut str_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaStringArrayGetString(ptr, index, &mut str_ptr))?;
            if str_ptr.is_null() {
                return Ok(String::new());
            }
            let c_str = CStr::from_ptr(str_ptr);
            let s = c_str.to_string_lossy().into_owned();
            Ok(s)
        }
    }
}

impl Drop for StringArray {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                OgaDestroyStringArray(self.ptr);
            }
        }
    }
}
