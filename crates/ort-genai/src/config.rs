use std::{
    ffi::CString,
    marker::PhantomData,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaConfig, OgaConfigAddModelData, OgaConfigAppendProvider,
    OgaConfigClearDecoderProviderOptionsHardwareDeviceId,
    OgaConfigClearDecoderProviderOptionsHardwareDeviceType,
    OgaConfigClearDecoderProviderOptionsHardwareVendorId, OgaConfigClearProviders,
    OgaConfigOverlay, OgaConfigRemoveModelData, OgaConfigSetDecoderProviderOptionsHardwareDeviceId,
    OgaConfigSetDecoderProviderOptionsHardwareDeviceType,
    OgaConfigSetDecoderProviderOptionsHardwareVendorId, OgaConfigSetProviderOption,
    OgaCreateConfig, OgaCreateConfigFromPackageEp, OgaDestroyConfig,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
};

/// Configuration object used when creating a [`Model`][crate::model::Model].
pub struct Config<'data> {
    ptr: NonNull<OgaConfig>,
    _type: PhantomData<&'data [u8]>,
    _marker: PhantomData<Rc<()>>,
}

impl<'data> Config<'data> {
    /// Creates a [`Config`] from the given configuration directory.
    ///
    /// # Parameters
    /// * `config_path` - The path to the configuration directory. The path is expected to be encoded in UTF-8.
    pub fn new(config_path: &str) -> OgaResult<Self> {
        let config_path = CString::new(config_path)?;
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateConfig(config_path.as_ptr(), &mut ptr) }.to_result()?;

        let mut config = Self::try_from(ptr)?;

        // For memory effciency and to better work with Rust's lifetimes, we will enable
        // use model bytes directly.
        config.overlay(
            r#"{
                "session": {
                    "use_ort_model_bytes_directly": "1"
                }
            }"#,
        )?;

        Ok(config)
    }

    /// Creates a [`Config`] from a model package directory, using the supplied execution provider
    /// to select a variant.
    ///
    /// `config_path` must refer to a model package; passing a flat directory returns an error.
    /// When the package declares exactly one execution provider across its variants, `ep` may be
    /// null or empty (the EP is auto-detected). Otherwise `ep` selects which variant is loaded.
    ///
    /// To load a model with an explicit EP, pass the resulting [`Config`] to
    /// [`Model::new_with_config`][crate::model::Model::new_with_config].
    /// [`Model::new`][crate::model::Model::new] does not accept an EP argument because the config
    /// carries everything needed.
    ///
    /// # Parameters
    /// * `config_path` - Path to the model package, encoded in UTF-8.
    /// * `ep` - Execution provider name, or `None` for auto-detection.
    pub fn new_from_package_ep(config_path: &str, ep: Option<&str>) -> OgaResult<Self> {
        let config_path = CString::new(config_path)?;
        let ep = ep.map(CString::new).transpose()?;
        let mut ptr = ptr::null_mut();
        let ep_ptr = ep.as_ref().map_or(ptr::null(), |ep| ep.as_ptr());

        unsafe { OgaCreateConfigFromPackageEp(config_path.as_ptr(), ep_ptr, &mut ptr) }
            .to_result()?;

        let mut config = Self::try_from(ptr)?;

        // For memory effciency and to better work with Rust's lifetimes, we will enable
        // use model bytes directly.
        config.overlay(
            r#"{
                "session": {
                    "use_ort_model_bytes_directly": "1"
                }
            }"#,
        )?;

        Ok(config)
    }

    /// Clears the list of providers in this configuration.
    pub fn clear_providers(&mut self) -> OgaResult<()> {
        unsafe { OgaConfigClearProviders(self.into()) }.to_result()
    }

    /// Adds the provider at the end of the list of providers in this configuration if it doesn't
    /// already exist. If it already exists, do nothing.
    ///
    /// # Parameters
    /// * `provider` - The provider to add to this configuration.
    pub fn append_provider(&mut self, provider: &str) -> OgaResult<()> {
        let provider = CString::new(provider)?;

        unsafe { OgaConfigAppendProvider(self.into(), provider.as_ptr()) }.to_result()
    }

    /// Sets an option for an execution provider in this configuration.
    ///
    /// # Parameters
    /// * `provider` - The provider to set the option for.
    /// * `key` - The key of the option to set.
    /// * `value` - The value of the option to set.
    pub fn set_provider_option(&mut self, provider: &str, key: &str, value: &str) -> OgaResult<()> {
        let provider = CString::new(provider)?;
        let key = CString::new(key)?;
        let value = CString::new(value)?;

        unsafe {
            OgaConfigSetProviderOption(self.into(), provider.as_ptr(), key.as_ptr(), value.as_ptr())
        }
        .to_result()
    }

    /// Adds model data to load the model from memory.
    ///
    /// Applications may call [`Config::remove_model_data`] to remove the model data when it is no
    /// longer needed.
    ///
    /// `session.use_ort_model_bytes_directly` is force-enabled by the [`Config`] constructors to
    /// avoid an extra copy of the model data in memory. The caller-owned `model_data` must
    /// therefore remain valid until the [`Model`][crate::model::Model] created from this config is
    /// destroyed, as the data may be used directly by the underlying ONNX Runtime session.
    ///
    /// # Parameters
    /// * `model_filename` - The name of the model file as defined in the configuration.
    /// * `model_data` - The model data to add. The data is expected to be valid at least until the model is created.
    pub fn add_model_data(
        &mut self,
        model_filename: &str,
        model_data: &'data [u8],
    ) -> OgaResult<()> {
        let model_filename = CString::new(model_filename)?;

        unsafe {
            OgaConfigAddModelData(
                self.into(),
                model_filename.as_ptr(),
                model_data.as_ptr().cast(),
                model_data.len(),
            )
        }
        .to_result()
    }

    /// Removes model data previously added to this [`Config`].
    ///
    /// # Parameters
    /// * `model_filename` - The name of the model file as defined in the configuration.
    pub fn remove_model_data(&mut self, model_filename: &str) -> OgaResult<()> {
        let model_filename = CString::new(model_filename)?;

        unsafe { OgaConfigRemoveModelData(self.into(), model_filename.as_ptr()) }.to_result()
    }

    /// Filters execution-provider devices by hardware device type.
    ///
    /// # Parameters
    /// * `provider` - The provider to configure.
    /// * `hardware_device_type` - Hardware device type, e.g. CPU, GPU, or NPU.
    pub fn set_decoder_provider_options_hardware_device_type(
        &mut self,
        provider: &str,
        hardware_device_type: HardwareDeviceType,
    ) -> OgaResult<()> {
        let provider = CString::new(provider)?;

        unsafe {
            OgaConfigSetDecoderProviderOptionsHardwareDeviceType(
                self.into(),
                provider.as_ptr(),
                hardware_device_type.into(),
            )
        }
        .to_result()
    }

    /// Filters execution-provider devices by hardware device ID.
    ///
    /// # Parameters
    /// * `provider` - The provider to configure.
    /// * `hardware_device_id` - Hardware device id.
    pub fn set_decoder_provider_options_hardware_device_id(
        &mut self,
        provider: &str,
        hardware_device_id: u32,
    ) -> OgaResult<()> {
        let provider = CString::new(provider)?;

        unsafe {
            OgaConfigSetDecoderProviderOptionsHardwareDeviceId(
                self.into(),
                provider.as_ptr(),
                hardware_device_id,
            )
        }
        .to_result()
    }

    /// Filters execution-provider devices by hardware vendor ID.
    ///
    /// # Parameters
    /// * `provider` - The provider to configure.
    /// * `hardware_vendor_id` - Hardware vendor id.
    pub fn set_decoder_provider_options_hardware_vendor_id(
        &mut self,
        provider: &str,
        hardware_vendor_id: u32,
    ) -> OgaResult<()> {
        let provider = CString::new(provider)?;

        unsafe {
            OgaConfigSetDecoderProviderOptionsHardwareVendorId(
                self.into(),
                provider.as_ptr(),
                hardware_vendor_id,
            )
        }
        .to_result()
    }

    /// Clears the hardware device type property for an execution provider.
    ///
    /// # Parameters
    /// * `provider` - The provider to clear the hardware device type property for.
    pub fn clear_decoder_provider_options_hardware_device_type(
        &mut self,
        provider: &str,
    ) -> OgaResult<()> {
        let provider = CString::new(provider)?;

        unsafe {
            OgaConfigClearDecoderProviderOptionsHardwareDeviceType(self.into(), provider.as_ptr())
        }
        .to_result()
    }

    /// Clears the hardware device ID property for an execution provider.
    ///
    /// # Parameters
    /// * `provider` - The provider to clear the hardware device id property for.
    pub fn clear_decoder_provider_options_hardware_device_id(
        &mut self,
        provider: &str,
    ) -> OgaResult<()> {
        let provider = CString::new(provider)?;

        unsafe {
            OgaConfigClearDecoderProviderOptionsHardwareDeviceId(self.into(), provider.as_ptr())
        }
        .to_result()
    }

    /// Clears the hardware vendor ID property for an execution provider.
    ///
    /// # Parameters
    /// * `provider` - The provider to clear the hardware vendor id property for.
    pub fn clear_decoder_provider_options_hardware_vendor_id(
        &mut self,
        provider: &str,
    ) -> OgaResult<()> {
        let provider = CString::new(provider)?;

        unsafe {
            OgaConfigClearDecoderProviderOptionsHardwareVendorId(self.into(), provider.as_ptr())
        }
        .to_result()
    }

    /// Overlays JSON on top of this configuration.
    ///
    /// # Parameters
    /// * `json` - The JSON to overlay on the config.
    pub fn overlay(&mut self, json: &str) -> OgaResult<()> {
        let json = CString::new(json)?;

        unsafe { OgaConfigOverlay(self.into(), json.as_ptr()) }.to_result()
    }
}

/// Hardware device type used for execution-provider configuration.
pub enum HardwareDeviceType {
    CPU,
    GPU,
    NPU,
}

impl From<HardwareDeviceType> for *const std::ffi::c_char {
    fn from(value: HardwareDeviceType) -> Self {
        match value {
            HardwareDeviceType::CPU => c"CPU".as_ptr(),
            HardwareDeviceType::GPU => c"GPU".as_ptr(),
            HardwareDeviceType::NPU => c"NPU".as_ptr(),
        }
    }
}

impl_into_ptr! {Config<'data>, OgaConfig}
impl_from_ptr! {Config<'data>, OgaConfig}
impl_drop! {Config<'data>, OgaDestroyConfig}
