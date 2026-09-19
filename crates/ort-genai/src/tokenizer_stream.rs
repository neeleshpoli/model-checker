use std::{
    ffi::CStr,
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateTokenizerStream, OgaCreateTokenizerStreamFromProcessor, OgaDestroyTokenizerStream,
    OgaTokenizerStream, OgaTokenizerStreamDecode,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr,
    multi_modal_processor::MultiModalProcessor,
    tokenizer::Tokenizer,
};

/// Decodes generated tokens incrementally into text chunks.
pub struct TokenizerStream {
    ptr: NonNull<OgaTokenizerStream>,
    _marker: PhantomData<Rc<()>>,
}

impl TokenizerStream {
    /// Creates a [`TokenizerStream`] from a [`Tokenizer`].
    ///
    /// # Parameters
    /// * `tokenizer` - The tokenizer used to decode the token stream.
    pub fn new(tokenizer: &Tokenizer) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();
        unsafe { OgaCreateTokenizerStream(tokenizer.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Creates a [`TokenizerStream`] from a [`MultiModalProcessor`].
    ///
    /// # Parameters
    /// * `processor` - The multimodal processor used to decode the token stream.
    pub fn new_from_processor(processor: &MultiModalProcessor) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();
        unsafe { OgaCreateTokenizerStreamFromProcessor(processor.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Decodes one token and returns any newly completed text chunk.
    ///
    /// The caller is responsible for concatenating the returned chunks to
    /// reconstruct the complete decoded text. The returned string slice is
    /// owned by the stream and remains valid until the next call to
    /// [`TokenizerStream::decode`] or until the stream is destroyed.
    ///
    /// # Parameters
    /// * `token` - The token ID to decode.
    pub fn decode(&mut self, token: i32) -> OgaResult<&str> {
        let mut ptr = ptr::null();
        unsafe { OgaTokenizerStreamDecode(self.ptr.as_ptr(), token, &mut ptr) }.to_result()?;

        let string_ptr =
            NonNull::new(ptr as *mut std::os::raw::c_char).ok_or(crate::error::Error::NulError)?;
        let string = unsafe { CStr::from_ptr(string_ptr.as_ptr()) };
        Ok(string.to_str()?)
    }
}

/// Creates a [`TokenizerStream`] from an owned [`Tokenizer`].
impl TryFrom<Tokenizer> for TokenizerStream {
    type Error = crate::error::Error;

    #[inline]
    fn try_from(value: Tokenizer) -> OgaResult<Self> {
        Self::new(&value)
    }
}

/// Creates a [`TokenizerStream`] from an owned [`MultiModalProcessor`].
impl TryFrom<MultiModalProcessor> for TokenizerStream {
    type Error = crate::error::Error;

    #[inline]
    fn try_from(value: MultiModalProcessor) -> OgaResult<Self> {
        Self::new_from_processor(&value)
    }
}

/// Creates a [`TokenizerStream`] from a borrowed [`Tokenizer`].
impl TryFrom<&Tokenizer> for TokenizerStream {
    type Error = crate::error::Error;

    #[inline]
    fn try_from(value: &Tokenizer) -> OgaResult<Self> {
        Self::new(value)
    }
}

/// Creates a [`TokenizerStream`] from a borrowed [`MultiModalProcessor`].
impl TryFrom<&MultiModalProcessor> for TokenizerStream {
    type Error = crate::error::Error;

    #[inline]
    fn try_from(value: &MultiModalProcessor) -> OgaResult<Self> {
        Self::new_from_processor(value)
    }
}

impl_from_ptr! {TokenizerStream, OgaTokenizerStream}
impl_drop! {TokenizerStream, OgaDestroyTokenizerStream}
