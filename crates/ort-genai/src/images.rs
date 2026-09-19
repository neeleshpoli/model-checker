use std::{
    ffi::CString,
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaDestroyImages, OgaImages, OgaLoadImage, OgaLoadImages, OgaLoadImagesFromBuffers,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
    string_array::StringArray,
};

/// A collection of images loaded for multimodal processing.
pub struct Images {
    ptr: NonNull<OgaImages>,
    _marker: PhantomData<Rc<()>>,
}

impl Images {
    /// Loads a single image from the given file path.
    ///
    /// # Parameters
    /// * `path` - The path to the image file.
    pub fn load_image(path: &str) -> OgaResult<Self> {
        let path = CString::new(path)?;
        let mut ptr = ptr::null_mut();

        unsafe { OgaLoadImage(path.as_ptr(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Loads multiple images from a string array of file paths.
    ///
    /// # Parameters
    /// * `paths` - The image file paths to load.
    pub fn load_images<S: AsRef<str>>(paths: &[S]) -> OgaResult<Self> {
        let images = TryInto::<StringArray>::try_into(paths)?;
        let mut ptr = ptr::null_mut();

        unsafe { OgaLoadImages(images.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Loads multiple images from byte buffers.
    ///
    /// The image data is consumed during this call; the input buffers do not
    /// need to remain valid after the method returns.
    ///
    /// # Parameters
    /// * `image_data` - The image byte buffers to load.
    pub fn load_images_from_buffers(image_data: &[&[u8]]) -> OgaResult<Self> {
        let mut data_ptrs = image_data
            .iter()
            .map(|data| data.as_ptr().cast())
            .collect::<Vec<_>>();
        let mut data_sizes: Vec<usize> = image_data.iter().map(|image| image.len()).collect();

        let mut ptr = ptr::null_mut();

        unsafe {
            OgaLoadImagesFromBuffers(
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

impl_from_ptr! {Images, OgaImages}
impl_into_ptr! {Images, OgaImages}
impl_drop! {Images, OgaDestroyImages}
