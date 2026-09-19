use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateEngine, OgaDestroyEngine, OgaEngine, OgaEngineAddRequest, OgaEngineHasPendingRequests,
    OgaEngineRemoveRequest, OgaEngineStep,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
    model::Model,
    request::Request,
};

/// Manages and schedules multiple [`Request`]s for model inference.
///
/// An engine coordinates batching, caching, and resource management while
/// processing requests. Requests can be added to the engine, processed one
/// step at a time, and removed when they are no longer needed.
pub struct Engine<T> {
    ptr: NonNull<OgaEngine>,
    _type: PhantomData<T>,
    _marker: PhantomData<Rc<()>>,
}

impl<T> Engine<T> {
    /// Creates an [`Engine`] from the given [`Model`].
    ///
    /// The engine uses the model to execute inference for requests added to
    /// it. The model must remain valid for the lifetime of the engine.
    ///
    /// # Parameters
    /// * `model` - The model to use for the engine.
    pub fn new(model: Model) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();
        unsafe { OgaCreateEngine(model.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Runs one processing step for the engine.
    ///
    /// This advances the engine by processing a subset of the currently
    /// pending requests. It schedules and executes model inference for ready
    /// requests, updates their state with generated results, and manages
    /// batching and resource allocation as needed.
    ///
    /// Call this method repeatedly to ensure that all requests are processed.
    /// If a request is ready from a previous step, that request is returned.
    /// Otherwise, the engine schedules another subset of requests and returns
    /// the first request from that subset that is ready to be queried. `None`
    /// is returned when no request is ready.
    pub fn step(&mut self) -> OgaResult<Option<Request<'_, T>>> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaEngineStep(self.into(), &mut ptr) }.to_result()?;

        if ptr.is_null() {
            Ok(None)
        } else {
            Ok(Some(Request::try_from(ptr)?))
        }
    }

    /// Checks whether the engine has any pending requests to process.
    ///
    /// Returns `true` if at least one request has not been fully processed;
    /// otherwise, returns `false`.
    pub fn has_pending_requests(&mut self) -> OgaResult<bool> {
        let mut ptr = MaybeUninit::uninit();

        unsafe { OgaEngineHasPendingRequests(self.into(), ptr.as_mut_ptr()) }.to_result()?;

        Ok(unsafe { ptr.assume_init() })
    }

    /// Adds a [`Request`] to the engine for processing.
    ///
    /// The request is submitted to the engine and processed by subsequent
    /// calls to [`Engine::step`]. The request must remain valid until it is
    /// removed from the engine or has been processed.
    ///
    /// # Parameters
    /// * `request` - The request to add to the engine.
    pub fn add_request(&mut self, request: &mut Request<T>) -> OgaResult<()> {
        unsafe { OgaEngineAddRequest(self.into(), request.into()) }.to_result()
    }

    /// Removes a [`Request`] from the engine.
    ///
    /// The request must have previously been added with
    /// [`Engine::add_request`]. After this call, the request is no longer
    /// processed by the engine and can be cleaned up.
    ///
    /// # Parameters
    /// * `request` - The request to remove from the engine.
    pub fn remove_request(&mut self, request: &mut Request<T>) -> OgaResult<()> {
        unsafe { OgaEngineRemoveRequest(self.into(), request.into()) }.to_result()
    }
}

impl_from_ptr! {Engine<T>, OgaEngine}
impl_into_ptr! {Engine<T>, OgaEngine}
impl_drop! {Engine<T>, OgaDestroyEngine}
