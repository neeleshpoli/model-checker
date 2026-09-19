use std::{
    ffi::CString,
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaAdapters, OgaCreateAdapters, OgaDestroyAdapters, OgaLoadAdapter, OgaSetActiveAdapter,
    OgaUnloadAdapter,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    generator::Generator,
    impl_drop, impl_from_ptr, impl_into_ptr,
    model::Model,
};

/// The [`Adapter`] object that manages adapters.
///
/// The [`Adapter`] object is used to load all model adapters and is responsible
/// for reference counting the loaded adapters.
pub struct Adapter {
    ptr: NonNull<OgaAdapters>,
    _marker: PhantomData<Rc<()>>,
}

impl Adapter {
    /// Creates the [`Adapter`] object that manages the adapters.
    pub fn new(model: Model) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateAdapters(model.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Loads the model adapter from the given adapter file path and adapter name.
    ///
    /// # Parameters
    /// * `adapter_file_path` - The file path of the adapter to load.
    /// * `adapter_name` - A unique identifier for the adapter chosen by the
    ///   function invoker. This name is used for querying the adapter.
    pub fn load_adapter(&mut self, adapter_file_path: &str, adapter_name: &str) -> OgaResult<()> {
        let adapter_file_path = CString::new(adapter_file_path)?;
        let adapter_name = CString::new(adapter_name)?;

        unsafe {
            OgaLoadAdapter(
                self.into(),
                adapter_file_path.as_ptr(),
                adapter_name.as_ptr(),
            )
        }
        .to_result()
    }

    /// Unloads the adapter with the given identifier from the previously loaded adapters.
    ///
    /// If the adapter is not found, or if it cannot be unloaded (when it is in use),
    /// an error is returned.
    ///
    /// # Parameters
    /// * `adapter_name` - The name of the adapter to unload.
    pub fn unload_adapter(&mut self, adapter_name: &str) -> OgaResult<()> {
        let adapter_name = CString::new(adapter_name)?;

        unsafe { OgaUnloadAdapter(self.into(), adapter_name.as_ptr()) }.to_result()
    }

    /// Sets the adapter with the given adapter name as active for the given [`Generator`] object.
    ///
    /// # Parameters
    /// * `generator` - The OgaGenerator object to set the active adapter.
    /// * `adapter_name` - The name of the adapter to set as active.
    pub fn set_active_adapter(
        &mut self,
        generator: &mut Generator,
        adapter_name: &str,
    ) -> OgaResult<()> {
        let adapter_name = CString::new(adapter_name)?;
        unsafe { OgaSetActiveAdapter(generator.into(), self.into(), adapter_name.as_ptr()) }
            .to_result()
    }
}

impl_from_ptr! {Adapter, OgaAdapters}
impl_into_ptr! {Adapter, OgaAdapters}
impl_drop! {Adapter, OgaDestroyAdapters}
