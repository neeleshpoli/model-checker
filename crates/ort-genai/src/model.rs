use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateModel, OgaCreateModelFromConfig, OgaCreateModelWithRuntimeSettings, OgaDestroyModel,
    OgaDestroyString, OgaModel, OgaModelGetDeviceType, OgaModelGetType,
};

use crate::{
    config::Config,
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
    runtime_settings::RuntimeSettings,
};

/// A model loaded from a configuration directory or [`Config`].
pub struct Model<'data> {
    ptr: NonNull<OgaModel>,
    _type: PhantomData<&'data [u8]>,
    _marker: PhantomData<Rc<()>>,
}

impl<'data> Model<'data> {
    /// Creates a [`Model`] from the given model configuration directory.
    ///
    /// # Parameters
    /// * `path_to_model` - The path to the model configuration directory. The
    ///   path is expected to be encoded in UTF-8.
    pub fn new(path_to_model: &str) -> OgaResult<Self> {
        let config_path = CString::new(path_to_model).expect("Config path contained a null byte");
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateModel(config_path.as_ptr(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Creates a [`Model`] from a model configuration directory and runtime
    /// settings.
    ///
    /// # Parameters
    /// * `path_to_model` - The path to the model configuration directory. The
    ///   path is expected to be encoded in UTF-8.
    /// * `runtime_settings` - The runtime settings to use for the model.
    pub fn new_with_runtime_settings(
        path_to_model: &str,
        runtime_settings: &RuntimeSettings,
    ) -> OgaResult<Self> {
        let path_to_model = CString::new(path_to_model)?;
        let mut ptr = ptr::null_mut();

        unsafe {
            OgaCreateModelWithRuntimeSettings(
                path_to_model.as_ptr(),
                runtime_settings.into(),
                &mut ptr,
            )
        }
        .to_result()?;

        Self::try_from(ptr)
    }

    /// Creates a [`Model`] from a [`Config`].
    ///
    /// The model borrows the configuration for its lifetime. This also keeps
    /// any caller-owned model data registered with the configuration valid
    /// while the model exists.
    ///
    /// # Parameters
    /// * `config` - The configuration to use for the model.
    pub fn new_with_config(config: &'data Config) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateModelFromConfig(config.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Returns the type of the model.
    ///
    /// The returned string is owned by Rust and remains valid independently of
    /// the model.
    pub fn get_type(&self) -> OgaResult<String> {
        let mut raw_ptr = ptr::null();

        unsafe { OgaModelGetType(self.into(), &mut raw_ptr) }.to_result()?;

        let model_type = NonNull::new(raw_ptr as *mut i8).ok_or(crate::error::Error::NulError)?;
        let value = unsafe { CStr::from_ptr(model_type.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        unsafe { OgaDestroyString(model_type.as_ptr()) };

        Ok(value)
    }

    /// Returns the device type of the model.
    ///
    /// The returned string is owned by Rust and remains valid independently of
    /// the model.
    pub fn get_device_type(&self) -> OgaResult<String> {
        let mut raw_ptr = ptr::null();

        unsafe { OgaModelGetDeviceType(self.into(), &mut raw_ptr) }.to_result()?;

        let model_type = NonNull::new(raw_ptr as *mut i8).ok_or(crate::error::Error::NulError)?;
        let value = unsafe { CStr::from_ptr(model_type.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        unsafe { OgaDestroyString(model_type.as_ptr()) };

        Ok(value)
    }
}

impl_into_ptr! {Model<'data>, OgaModel}
impl_from_ptr! {Model<'data>, OgaModel}
impl_drop! {Model<'data>, OgaDestroyModel}
