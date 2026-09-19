use std::{
    ffi::CString,
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateRuntimeSettings, OgaDestroyRuntimeSettings, OgaRuntimeSettings,
    OgaRuntimeSettingsSetHandle,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
};

/// Runtime settings used when creating a [`Model`][crate::Model].
pub struct RuntimeSettings {
    ptr: NonNull<OgaRuntimeSettings>,
    _marker: PhantomData<Rc<()>>,
}

impl RuntimeSettings {
    /// Creates empty runtime settings for use when creating a model.
    pub fn new() -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateRuntimeSettings(&mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Sets a named runtime handle.
    ///
    /// # Safety
    /// The caller must ensure that `handle` has the type and validity required
    /// by the runtime for `handle_name`. The pointed-to resource is not owned
    /// or managed by [`RuntimeSettings`] and must remain valid for as long as
    /// the runtime may use it.
    ///
    /// # Parameters
    /// * `handle_name` - The name of the runtime handle to set.
    /// * `handle` - The handle value to associate with the name.
    pub unsafe fn set_handle<T>(&self, handle_name: &str, handle: *mut T) -> OgaResult<()> {
        let handle_name = CString::new(handle_name)?;

        unsafe {
            OgaRuntimeSettingsSetHandle(self.ptr.as_ptr(), handle_name.as_ptr(), handle.cast())
        }
        .to_result()
    }
}

impl_into_ptr! {RuntimeSettings, OgaRuntimeSettings}
impl_from_ptr! {RuntimeSettings, OgaRuntimeSettings}
impl_drop! {RuntimeSettings, OgaDestroyRuntimeSettings}
