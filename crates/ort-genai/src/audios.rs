use std::{
    ffi::CString,
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaAudios, OgaDestroyAudios, OgaLoadAudio, OgaLoadAudios, OgaLoadAudiosFromBuffers,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
    string_array::StringArray,
};

/// Audio collection wrapper.
pub struct Audios {
    ptr: NonNull<OgaAudios>,
    _marker: PhantomData<Rc<()>>,
}

impl Audios {
    /// Loads a single audio from the given file path.
    pub fn load_audio(path: &str) -> OgaResult<Self> {
        let path = CString::new(path)?;
        let mut ptr = ptr::null_mut();

        unsafe { OgaLoadAudio(path.as_ptr(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Loads multiple audios from a string array of file paths.
    pub fn load_audios<S: AsRef<str>>(audios: &[S]) -> OgaResult<Self> {
        let audios = TryInto::<StringArray>::try_into(audios)?;
        let mut ptr = ptr::null_mut();

        unsafe { OgaLoadAudios(audios.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Load multiple audios from an array of byte buffers.
    ///
    /// # Parameters
    /// * `audio_data` - Array of byte buffers containing the audio data.
    pub fn load_audios_from_buffers(audio_data: Vec<Vec<u8>>) -> OgaResult<Self> {
        let mut data_ptrs = audio_data
            .iter()
            .map(|data| data.as_ptr().cast())
            .collect::<Vec<_>>();
        let mut data_sizes: Vec<usize> = audio_data.iter().map(Vec::len).collect();

        let mut ptr = ptr::null_mut();

        unsafe {
            OgaLoadAudiosFromBuffers(
                data_ptrs.as_mut_ptr(),
                data_sizes.as_mut_ptr(),
                data_ptrs.len(),
                &mut ptr,
            )
        }
        .to_result()?;

        Self::try_from(ptr)
    }
}

impl_from_ptr! {Audios, OgaAudios}
impl_into_ptr! {Audios, OgaAudios}
impl_drop! {Audios, OgaDestroyAudios}
