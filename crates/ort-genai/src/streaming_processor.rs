use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateStreamingProcessor, OgaDestroyStreamingProcessor, OgaDestroyString,
    OgaStreamingProcessor, OgaStreamingProcessorFlush, OgaStreamingProcessorGetOption,
    OgaStreamingProcessorProcess, OgaStreamingProcessorSetOption,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
    model::Model,
    named_tensors::NamedTensors,
};

/// Processes streaming PCM audio into model input tensors.
pub struct StreamingProcessor {
    ptr: NonNull<OgaStreamingProcessor>,
    _marker: PhantomData<Rc<()>>,
}

impl StreamingProcessor {
    /// Creates a [`StreamingProcessor`] for the given speech [`Model`].
    ///
    /// The model must be a `nemotron_speech` model.
    ///
    /// # Parameters
    /// * `model` - The model to use for mel spectrogram extraction.
    pub fn new(model: Model) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateStreamingProcessor(model.into(), &mut ptr) }.to_result()?;

        ptr.try_into()
    }

    /// Processes a chunk of raw mono PCM audio samples.
    ///
    /// Returns [`NamedTensors`] when enough audio has been buffered to produce
    /// a complete chunk. The returned tensors are owned by the caller. A call
    /// may fail to produce a complete chunk until more audio is provided.
    ///
    /// Audio samples must use the model's sample rate.
    ///
    /// # Parameters
    /// * `data` - The raw PCM audio samples as `f32` values.
    pub fn process(&mut self, data: &[f32]) -> OgaResult<NamedTensors<'_>> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaStreamingProcessorProcess(self.into(), data.as_ptr(), data.len(), &mut ptr) }
            .to_result()?;

        ptr.try_into()
    }

    /// Flushes any remaining buffered audio, padding it with silence.
    ///
    /// Returns the resulting [`NamedTensors`]. If the buffer is empty, the
    /// native processor may report an error rather than producing tensors.
    pub fn flush(&mut self) -> OgaResult<NamedTensors<'_>> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaStreamingProcessorFlush(self.into(), &mut ptr) }.to_result()?;

        ptr.try_into()
    }

    /// Sets a streaming processor option.
    ///
    /// Supported option keys include `use_vad`, `vad_threshold`,
    /// `silence_duration_ms`, and `prefix_padding_ms`.
    ///
    /// # Parameters
    /// * `key` - The option name.
    /// * `value` - The option value, represented as a string.
    pub fn set_option(&mut self, key: &str, value: &str) -> OgaResult<()> {
        let key = CString::new(key)?;
        let value = CString::new(value)?;

        unsafe { OgaStreamingProcessorSetOption(self.into(), key.as_ptr(), value.as_ptr()) }
            .to_result()
    }

    /// Returns a streaming processor option value.
    ///
    /// The returned value is owned by Rust and remains valid independently of
    /// the processor.
    ///
    /// # Parameters
    /// * `key` - The option name.
    pub fn get_option(&self, key: &str) -> OgaResult<String> {
        let key = CString::new(key)?;
        let mut ptr = ptr::null();

        unsafe { OgaStreamingProcessorGetOption(self.into(), key.as_ptr(), &mut ptr) }
            .to_result()?;

        let value_ptr =
            NonNull::new(ptr as *mut std::os::raw::c_char).ok_or(crate::error::Error::NulError)?;
        let value = unsafe { CStr::from_ptr(value_ptr.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        unsafe { OgaDestroyString(value_ptr.as_ptr()) };

        Ok(value)
    }
}

impl_from_ptr! {StreamingProcessor, OgaStreamingProcessor}
impl_into_ptr! {StreamingProcessor, OgaStreamingProcessor}
impl_drop! {StreamingProcessor, OgaDestroyStreamingProcessor}
