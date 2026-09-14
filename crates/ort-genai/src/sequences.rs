use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaAppendTokenSequence, OgaAppendTokenToSequence, OgaCreateSequences, OgaDestroySequences,
    OgaSequences, OgaSequencesCount, OgaSequencesGetSequenceCount, OgaSequencesGetSequenceData,
};

use crate::error::{Result, check_status};

pub struct Sequences {
    pub(crate) ptr: Arc<Mutex<*mut OgaSequences>>,
}

unsafe impl Send for Sequences {}
unsafe impl Sync for Sequences {}

impl Sequences {
    pub fn new() -> Result<Self> {
        let mut ptr: *mut OgaSequences = std::ptr::null_mut();
        unsafe {
            check_status(OgaCreateSequences(&mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn num_sequences(&self) -> Result<usize> {
        let ptr = self.ptr.lock()?;
        unsafe { Ok(OgaSequencesCount(*ptr)) }
    }

    pub fn append_sequence(&self, tokens: &[i32]) -> Result<()> {
        let ptr = self.ptr.lock()?;
        unsafe {
            check_status(OgaAppendTokenSequence(tokens.as_ptr(), tokens.len(), *ptr))?;
        }
        Ok(())
    }

    pub fn append_token(&self, token: i32, sequence_index: usize) -> Result<()> {
        let ptr = self.ptr.lock()?;
        unsafe {
            check_status(OgaAppendTokenToSequence(token, *ptr, sequence_index))?;
        }
        Ok(())
    }

    pub fn sequence_len(&self, sequence_index: usize) -> Result<usize> {
        let ptr = self.ptr.lock()?;
        unsafe { Ok(OgaSequencesGetSequenceCount(*ptr, sequence_index)) }
    }

    pub fn get_sequence(&self, sequence_index: usize) -> Result<Vec<i32>> {
        let ptr = self.ptr.lock()?;
        unsafe {
            let count = OgaSequencesGetSequenceCount(*ptr, sequence_index);
            let data_ptr = OgaSequencesGetSequenceData(*ptr, sequence_index);
            if data_ptr.is_null() {
                return Ok(Vec::new());
            }
            let slice = std::slice::from_raw_parts(data_ptr, count);
            Ok(slice.to_vec())
        }
    }
}

impl Drop for Sequences {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroySequences(*ptr);
                }
            }
        }
    }
}
