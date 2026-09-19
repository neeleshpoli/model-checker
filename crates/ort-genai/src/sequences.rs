use std::{
    marker::PhantomData,
    ops::Index,
    ptr::{self, NonNull},
    rc::Rc,
    slice,
};

use ort_genai_sys::{
    OgaAppendTokenSequence, OgaAppendTokenToSequence, OgaCreateSequences, OgaDestroySequences,
    OgaSequences, OgaSequencesCount, OgaSequencesGetSequenceCount, OgaSequencesGetSequenceData,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
};

/// A collection of token sequences used to seed generation or requests.
pub struct Sequences {
    ptr: NonNull<OgaSequences>,
    _marker: PhantomData<Rc<()>>,
}

impl Sequences {
    /// Creates an empty [`Sequences`] collection.
    pub fn new() -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateSequences(&mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Returns the number of sequences in this collection.
    pub fn count(&self) -> usize {
        unsafe { OgaSequencesCount(self.into()) }
    }

    /// Appends all tokens in `tokens` as a new sequence.
    ///
    /// # Parameters
    /// * `tokens` - The token IDs to append as a sequence.
    pub fn append_token_sequence(&mut self, tokens: &[i32]) -> OgaResult<()> {
        unsafe { OgaAppendTokenSequence(tokens.as_ptr(), tokens.len(), self.into()) }.to_result()
    }

    /// Appends a token to the sequence at `index`.
    ///
    /// If `index` equals the current sequence count, a new sequence is created
    /// at that index. Other out-of-range indices return an error.
    ///
    /// # Parameters
    /// * `token` - The token ID to append.
    /// * `index` - The index of the sequence to append to.
    pub fn append_token_to_sequence(&mut self, token: i32, index: usize) -> OgaResult<()> {
        unsafe { OgaAppendTokenToSequence(token, self.into(), index) }.to_result()
    }

    /// Returns the number of tokens in the sequence at `index`.
    ///
    /// Returns `0` when `index` is out of bounds.
    ///
    /// # Parameters
    /// * `index` - The sequence index.
    pub fn get_sequence_count(&self, index: usize) -> usize {
        unsafe { OgaSequencesGetSequenceCount(self.into(), index) }
    }

    /// Returns the token data for the sequence at `index`.
    ///
    /// The returned slice is owned by this collection and remains valid until
    /// the collection is destroyed. Returns `None` when `index` is out of
    /// bounds.
    ///
    /// # Parameters
    /// * `index` - The sequence index.
    pub fn get_sequence_data(&self, index: usize) -> Option<&[i32]> {
        let ptr = unsafe { OgaSequencesGetSequenceData(self.into(), index) };

        if ptr.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(ptr, self.get_sequence_count(index)) })
        }
    }
}

/// Provides indexed access to sequence token data.
///
/// Panics when the sequence index is out of bounds.
impl Index<usize> for Sequences {
    type Output = [i32];

    fn index(&self, index: usize) -> &Self::Output {
        if let Some(data) = self.get_sequence_data(index) {
            data
        } else {
            panic!("Index {index} out of bounds for length {}", self.count())
        }
    }
}

impl_into_ptr! {Sequences, OgaSequences}
impl_from_ptr! {Sequences, OgaSequences}
impl_drop! {Sequences, OgaDestroySequences}
