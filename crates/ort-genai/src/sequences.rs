use ort_genai_sys::{
    OgaAppendTokenSequence, OgaAppendTokenToSequence, OgaCreateSequences, OgaDestroySequences,
    OgaSequences, OgaSequencesCount, OgaSequencesGetSequenceCount, OgaSequencesGetSequenceData,
};

use crate::error::{check_status, Result};

pub struct Sequences {
    pub(crate) ptr: *mut OgaSequences,
}

impl Sequences {
    pub fn new() -> Result<Self> {
        let mut ptr: *mut OgaSequences = std::ptr::null_mut();
        unsafe {
            check_status(OgaCreateSequences(&mut ptr))?;
        }
        Ok(Self { ptr: ptr })
    }

    /// Returns the number of sequences in the Sequences
    /// * sequences
    /// Returns: The number of sequences in the Sequences
    /// /
    /// Returns the number of sequences in the Sequences
    /// * sequences
    /// Returns: The number of sequences in the Sequences
    /// /
    pub fn num_sequences(&self) -> Result<usize> {
        let _ptr = self.ptr;
        unsafe { Ok(OgaSequencesCount(self.ptr)) }
    }

    pub fn append_sequence(&self, tokens: &[i32]) -> Result<()> {
        let ptr = self.ptr;
        unsafe {
            check_status(OgaAppendTokenSequence(tokens.as_ptr(), tokens.len(), ptr))?;
        }
        Ok(())
    }

    pub fn append_token(&self, token: i32, sequence_index: usize) -> Result<()> {
        let ptr = self.ptr;
        unsafe {
            check_status(OgaAppendTokenToSequence(token, ptr, sequence_index))?;
        }
        Ok(())
    }

    /// Returns the number of tokens in the sequence at the given index.
    /// * sequences Sequences to use.
    /// * sequence_index index of the sequence to use.
    /// Returns: The number of tokens in the sequence at the given index. Returns 0 if
    /// sequence_index is out of bounds (i.e. >= SequencesCount(sequences)).
    /// /
    /// Returns the number of tokens in the sequence at the given index.
    /// * sequences Sequences to use.
    /// * sequence_index index of the sequence to use.
    /// Returns: The number of tokens in the sequence at the given index. Returns 0 if
    /// sequence_index is out of bounds (i.e. >= SequencesCount(sequences)).
    /// /
    pub fn sequence_len(&self, sequence_index: usize) -> Result<usize> {
        let ptr = self.ptr;
        unsafe { Ok(OgaSequencesGetSequenceCount(ptr, sequence_index)) }
    }

    /// Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
    /// is given by SequencesGetSequenceCount
    /// * sequences Sequences to use.
    /// * sequence_index index of the sequence to use.
    /// Returns: The pointer to the sequence data at the given index. The pointer is valid until the Sequences is destroyed.
    /// Returns nullptr if sequence_index is out of bounds (i.e. >= SequencesCount(sequences)).
    /// /
    /// Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
    /// is given by SequencesGetSequenceCount
    /// * sequences Sequences to use.
    /// * sequence_index index of the sequence to use.
    /// Returns: The pointer to the sequence data at the given index. The pointer is valid until the Sequences is destroyed.
    /// Returns nullptr if sequence_index is out of bounds (i.e. >= SequencesCount(sequences)).
    /// /
    pub fn get_sequence(&self, sequence_index: usize) -> Result<Vec<i32>> {
        let ptr = self.ptr;
        unsafe {
            let count = OgaSequencesGetSequenceCount(ptr, sequence_index);
            let data_ptr = OgaSequencesGetSequenceData(ptr, sequence_index);
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
        if !self.ptr.is_null() {
            unsafe {
                OgaDestroySequences(self.ptr);
            }
        }
    }
}
