use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateMultiModalProcessor, OgaDestroyMultiModalProcessor, OgaDestroyString,
    OgaMultiModalProcessor, OgaProcessorDecode, OgaProcessorProcessAudios,
    OgaProcessorProcessAudiosAndPrompts, OgaProcessorProcessImages,
    OgaProcessorProcessImagesAndAudios, OgaProcessorProcessImagesAndAudiosAndPrompts,
    OgaProcessorProcessImagesAndPrompts,
};

use crate::{
    audios::Audios,
    error::{OgaResult, OgaResultExt},
    images::Images,
    impl_drop, impl_from_ptr, impl_into_ptr,
    model::Model,
    named_tensors::NamedTensors,
    string_array::StringArray,
};

/// Processes image and audio inputs for multimodal model inference.
pub struct MultiModalProcessor {
    ptr: NonNull<OgaMultiModalProcessor>,
    _marker: PhantomData<Rc<()>>,
}

impl MultiModalProcessor {
    /// Creates a [`MultiModalProcessor`] for the given [`Model`].
    ///
    /// # Parameters
    /// * `model` - The model to use for multimodal processing.
    pub fn new(model: &Model) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateMultiModalProcessor(model.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Processes images with a single input prompt.
    ///
    /// Returns the named tensors for the processed inputs.
    ///
    /// # Parameters
    /// * `prompt` - The prompt to use with the images.
    /// * `images` - The images to process.
    pub fn process_images(&self, prompt: &str, images: &Images) -> OgaResult<NamedTensors<'_>> {
        let prompt = CString::new(prompt)?;
        let mut ptr = ptr::null_mut();

        unsafe { OgaProcessorProcessImages(self.into(), prompt.as_ptr(), images.into(), &mut ptr) }
            .to_result()?;

        ptr.try_into()
    }

    /// Processes images with multiple input prompts.
    ///
    /// Returns the named tensors for the processed inputs.
    ///
    /// # Parameters
    /// * `prompts` - The prompts to use with the images.
    /// * `images` - The images to process.
    pub fn process_images_and_prompts<S: AsRef<str>>(
        &self,
        prompts: &[S],
        images: &Images,
    ) -> OgaResult<NamedTensors<'_>> {
        let prompts = TryInto::<StringArray>::try_into(prompts)?;
        let mut ptr = ptr::null_mut();

        unsafe {
            OgaProcessorProcessImagesAndPrompts(
                self.into(),
                prompts.into(),
                images.into(),
                &mut ptr,
            )
        }
        .to_result()?;

        ptr.try_into()
    }

    /// Processes audio inputs with a single input prompt.
    ///
    /// Returns the named tensors for the processed inputs.
    ///
    /// # Parameters
    /// * `prompt` - The prompt to use with the audio inputs.
    /// * `audios` - The audio inputs to process.
    pub fn process_audios(&self, prompt: &str, audios: &Audios) -> OgaResult<NamedTensors<'_>> {
        let prompt = CString::new(prompt)?;
        let mut ptr = ptr::null_mut();

        unsafe { OgaProcessorProcessAudios(self.into(), prompt.as_ptr(), audios.into(), &mut ptr) }
            .to_result()?;

        ptr.try_into()
    }

    /// Processes audio inputs with multiple input prompts.
    ///
    /// Returns the named tensors for the processed inputs.
    ///
    /// # Parameters
    /// * `prompts` - The prompts to use with the audio inputs.
    /// * `audios` - The audio inputs to process.
    pub fn process_audios_and_prompts<S: AsRef<str>>(
        &self,
        prompts: &[S],
        audios: &Audios,
    ) -> OgaResult<NamedTensors<'_>> {
        let prompts = TryInto::<StringArray>::try_into(prompts)?;
        let mut ptr = ptr::null_mut();

        unsafe {
            OgaProcessorProcessAudiosAndPrompts(
                self.into(),
                prompts.into(),
                audios.into(),
                &mut ptr,
            )
        }
        .to_result()?;

        ptr.try_into()
    }

    /// Processes images and audio inputs with a single input prompt.
    ///
    /// Returns the named tensors for the processed inputs.
    ///
    /// # Parameters
    /// * `prompt` - The prompt to use with the images and audio inputs.
    /// * `images` - The images to process.
    /// * `audios` - The audio inputs to process.
    pub fn process_images_and_audios(
        &self,
        prompt: &str,
        images: &Images,
        audios: &Audios,
    ) -> OgaResult<NamedTensors<'_>> {
        let prompt = CString::new(prompt)?;
        let mut ptr = ptr::null_mut();

        unsafe {
            OgaProcessorProcessImagesAndAudios(
                self.into(),
                prompt.as_ptr(),
                images.into(),
                audios.into(),
                &mut ptr,
            )
        }
        .to_result()?;

        ptr.try_into()
    }

    /// Processes images and audio inputs with multiple input prompts.
    ///
    /// Returns the named tensors for the processed inputs.
    ///
    /// # Parameters
    /// * `prompts` - The prompts to use with the images and audio inputs.
    /// * `images` - The images to process.
    /// * `audios` - The audio inputs to process.
    pub fn process_images_and_audios_and_prompts<S: AsRef<str>>(
        &self,
        prompts: &[S],
        images: &Images,
        audios: &Audios,
    ) -> OgaResult<NamedTensors<'_>> {
        let prompts = TryInto::<StringArray>::try_into(prompts)?;
        let mut ptr = ptr::null_mut();

        unsafe {
            OgaProcessorProcessImagesAndAudiosAndPrompts(
                self.into(),
                prompts.into(),
                images.into(),
                audios.into(),
                &mut ptr,
            )
        }
        .to_result()?;

        ptr.try_into()
    }

    /// Decodes a token sequence into a UTF-8 string.
    ///
    /// The returned string is owned by Rust and remains valid independently of
    /// the processor.
    ///
    /// # Parameters
    /// * `tokens` - The token sequence to decode.
    pub fn decode(&self, tokens: &[i32]) -> OgaResult<String> {
        let mut ptr = ptr::null();

        unsafe { OgaProcessorDecode(self.into(), tokens.as_ptr(), tokens.len(), &mut ptr) }
            .to_result()?;

        let value_ptr =
            NonNull::new(ptr as *mut std::os::raw::c_char).ok_or(crate::error::Error::NulError)?;
        let result = unsafe { CStr::from_ptr(value_ptr.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        unsafe { OgaDestroyString(value_ptr.as_ptr()) };

        Ok(result)
    }
}

impl_from_ptr! {MultiModalProcessor, OgaMultiModalProcessor}
impl_into_ptr! {MultiModalProcessor, OgaMultiModalProcessor}
impl_drop! {MultiModalProcessor, OgaDestroyMultiModalProcessor}
