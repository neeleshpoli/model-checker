use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateRequest, OgaDestroyRequest, OgaRequest, OgaRequestAddTokens, OgaRequestGetOpaqueData,
    OgaRequestGetUnseenToken, OgaRequestHasUnseenTokens, OgaRequestIsDone, OgaRequestSetOpaqueData,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    generator_params::GeneratorParams,
    impl_drop, impl_from_ptr, impl_into_ptr,
    sequences::Sequences,
};

/// A model-inference request with application-defined opaque data.
///
/// The lifetime `'data` ties the request to the value supplied through
/// [`Request::set_opaque_data`].
pub struct Request<'data, T> {
    ptr: NonNull<OgaRequest>,
    _type: PhantomData<&'data T>,
    _marker: PhantomData<Rc<()>>,
}

impl<'data, T> Request<'data, T> {
    /// Creates a [`Request`] from generator parameters.
    ///
    /// The request can be submitted to an [`Engine`][crate::Engine] for
    /// processing after its input tokens have been added.
    ///
    /// # Parameters
    /// * `params` - The generator parameters to use for the request.
    pub fn new(params: &mut GeneratorParams) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();
        unsafe { OgaCreateRequest(params.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Adds input sequences to the request to seed generation.
    ///
    /// # Parameters
    /// * `tokens` - The input sequences to add to the request.
    pub fn add_tokens(&mut self, tokens: &Sequences) -> OgaResult<()> {
        unsafe { OgaRequestAddTokens(self.into(), tokens.into()) }.to_result()
    }

    /// Associates caller-owned opaque data with the request.
    ///
    /// The request stores a pointer to `opaque_data`; it does not take
    /// ownership of the value. The value must remain valid and must not be
    /// moved for the lifetime of the request. Use [`Request::get_opaque_data`]
    /// to retrieve it later.
    ///
    /// # Parameters
    /// * `opaque_data` - The caller-owned data to associate with the request.
    pub fn set_opaque_data(&mut self, opaque_data: &'data mut T) -> OgaResult<()> {
        unsafe { OgaRequestSetOpaqueData(self.into(), (opaque_data as *mut T).cast()) }.to_result()
    }

    /// Returns a mutable reference to the opaque data associated with the
    /// request.
    ///
    /// This must only be called after [`Request::set_opaque_data`] has been
    /// called with a value of the same type `T`.
    pub fn get_opaque_data(&mut self) -> OgaResult<&mut T> {
        let mut ptr = ptr::null_mut();
        unsafe { OgaRequestGetOpaqueData(self.into(), &mut ptr) }.to_result()?;

        let mut ptr = NonNull::new(ptr.cast::<T>()).to_result()?;
        Ok(unsafe { ptr.as_mut() })
    }

    /// Returns whether the request has generated tokens that have not yet
    /// been retrieved.
    pub fn has_unseen_tokens(&self) -> OgaResult<bool> {
        let mut unseen = MaybeUninit::uninit();

        unsafe { OgaRequestHasUnseenTokens(self.into(), unseen.as_mut_ptr()) }.to_result()?;

        Ok(unsafe { unseen.assume_init() })
    }

    /// Retrieves the next generated token that has not yet been queried.
    ///
    /// Returns an error if no unseen token is available.
    pub fn get_unseen_token(&mut self) -> OgaResult<i32> {
        let mut token = MaybeUninit::uninit();

        unsafe { OgaRequestGetUnseenToken(self.into(), token.as_mut_ptr()) }.to_result()?;

        Ok(unsafe { token.assume_init() })
    }

    /// Returns whether request processing has finished.
    ///
    /// A request is complete when a termination condition has been reached,
    /// such as an end-of-sequence token or cancellation.
    pub fn is_done(&self) -> OgaResult<bool> {
        let mut done = MaybeUninit::uninit();

        unsafe { OgaRequestIsDone(self.into(), done.as_mut_ptr()) }.to_result()?;

        Ok(unsafe { done.assume_init() })
    }
}

impl_from_ptr! {Request<'data, T>, OgaRequest}
impl_into_ptr! {Request<'data, T>, OgaRequest}
impl_drop! {Request<'data, T>, OgaDestroyRequest}
