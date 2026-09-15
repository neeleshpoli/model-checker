use std::ffi::CString;

use ort_genai_sys::{
    OgaConfig, OgaConfigAppendProvider, OgaConfigClearDecoderProviderOptionsHardwareDeviceId,
    OgaConfigClearDecoderProviderOptionsHardwareDeviceType,
    OgaConfigClearDecoderProviderOptionsHardwareVendorId, OgaConfigClearProviders,
    OgaConfigSetDecoderProviderOptionsHardwareDeviceId,
    OgaConfigSetDecoderProviderOptionsHardwareDeviceType,
    OgaConfigSetDecoderProviderOptionsHardwareVendorId, OgaConfigSetProviderOption,
    OgaCreateConfig, OgaCreateConfigFromPackageEp, OgaDestroyConfig,
};

use crate::error::{check_status, Result};

pub struct Config {
    pub(crate) ptr: *mut OgaConfig,
}

impl Config {
    pub fn new(config_path: &str) -> Result<Self> {
        let mut ptr: *mut OgaConfig = std::ptr::null_mut();
        let c_path = CString::new(config_path)?;
        unsafe {
            check_status(OgaCreateConfig(c_path.as_ptr(), &mut ptr))?;
        }
        Ok(Self { ptr: ptr })
    }

    pub fn from_package_ep(config_path: &str, ep: &str) -> Result<Self> {
        let mut ptr: *mut OgaConfig = std::ptr::null_mut();
        let c_path = CString::new(config_path)?;
        let c_ep = CString::new(ep)?;
        unsafe {
            check_status(OgaCreateConfigFromPackageEp(
                c_path.as_ptr(),
                c_ep.as_ptr(),
                &mut ptr,
            ))?;
        }
        Ok(Self { ptr: ptr })
    }

    pub fn clear_providers(&self) -> Result<()> {
        let _ptr = self.ptr;
        unsafe {
            check_status(OgaConfigClearProviders(self.ptr))?;
        }
        Ok(())
    }

    pub fn append_provider(&self, provider: &str) -> Result<()> {
        let ptr = self.ptr;
        let c_provider = CString::new(provider)?;
        unsafe {
            check_status(OgaConfigAppendProvider(ptr, c_provider.as_ptr()))?;
        }
        Ok(())
    }

    pub fn set_provider_option(&self, provider: &str, key: &str, value: &str) -> Result<()> {
        let ptr = self.ptr;
        let c_provider = CString::new(provider)?;
        let c_key = CString::new(key)?;
        let c_value = CString::new(value)?;
        unsafe {
            check_status(OgaConfigSetProviderOption(
                ptr,
                c_provider.as_ptr(),
                c_key.as_ptr(),
                c_value.as_ptr(),
            ))?;
        }
        Ok(())
    }

    pub fn set_decoder_provider_options_hardware_device_type(
        &self,
        provider: &str,
        hardware_device_type: &str,
    ) -> Result<()> {
        let ptr = self.ptr;
        let c_provider = CString::new(provider)?;
        let c_device_type = CString::new(hardware_device_type)?;
        unsafe {
            check_status(OgaConfigSetDecoderProviderOptionsHardwareDeviceType(
                ptr,
                c_provider.as_ptr(),
                c_device_type.as_ptr(),
            ))?;
        }
        Ok(())
    }

    pub fn set_decoder_provider_options_hardware_device_id(
        &self,
        provider: &str,
        hardware_device_id: u32,
    ) -> Result<()> {
        let ptr = self.ptr;
        let c_provider = CString::new(provider)?;
        unsafe {
            check_status(OgaConfigSetDecoderProviderOptionsHardwareDeviceId(
                ptr,
                c_provider.as_ptr(),
                hardware_device_id,
            ))?;
        }
        Ok(())
    }

    pub fn set_decoder_provider_options_hardware_vendor_id(
        &self,
        provider: &str,
        hardware_vendor_id: u32,
    ) -> Result<()> {
        let ptr = self.ptr;
        let c_provider = CString::new(provider)?;
        unsafe {
            check_status(OgaConfigSetDecoderProviderOptionsHardwareVendorId(
                ptr,
                c_provider.as_ptr(),
                hardware_vendor_id,
            ))?;
        }
        Ok(())
    }

    pub fn clear_decoder_provider_options_hardware_device_type(
        &self,
        provider: &str,
    ) -> Result<()> {
        let ptr = self.ptr;
        let c_provider = CString::new(provider)?;
        unsafe {
            check_status(OgaConfigClearDecoderProviderOptionsHardwareDeviceType(
                ptr,
                c_provider.as_ptr(),
            ))?;
        }
        Ok(())
    }

    pub fn clear_decoder_provider_options_hardware_device_id(&self, provider: &str) -> Result<()> {
        let ptr = self.ptr;
        let c_provider = CString::new(provider)?;
        unsafe {
            check_status(OgaConfigClearDecoderProviderOptionsHardwareDeviceId(
                ptr,
                c_provider.as_ptr(),
            ))?;
        }
        Ok(())
    }

    pub fn clear_decoder_provider_options_hardware_vendor_id(&self, provider: &str) -> Result<()> {
        let ptr = self.ptr;
        let c_provider = CString::new(provider)?;
        unsafe {
            check_status(OgaConfigClearDecoderProviderOptionsHardwareVendorId(
                ptr,
                c_provider.as_ptr(),
            ))?;
        }
        Ok(())
    }
}

impl Drop for Config {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                OgaDestroyConfig(self.ptr);
            }
        }
    }
}

impl Config {
    pub fn add_model_data(&self, model_filename: &str, model_data: &[u8]) -> Result<()> {
        let ptr = self.ptr;
        let c_filename = CString::new(model_filename)?;
        unsafe {
            check_status(ort_genai_sys::OgaConfigAddModelData(
                ptr,
                c_filename.as_ptr(),
                model_data.as_ptr() as *const std::ffi::c_void,
                model_data.len(),
            ))?;
        }
        Ok(())
    }

    pub fn remove_model_data(&self, model_filename: &str) -> Result<()> {
        let ptr = self.ptr;
        let c_filename = CString::new(model_filename)?;
        unsafe {
            check_status(ort_genai_sys::OgaConfigRemoveModelData(
                ptr,
                c_filename.as_ptr(),
            ))?;
        }
        Ok(())
    }

    pub fn overlay(&self, json: &str) -> Result<()> {
        let ptr = self.ptr;
        let c_json = CString::new(json)?;
        unsafe {
            check_status(ort_genai_sys::OgaConfigOverlay(ptr, c_json.as_ptr()))?;
        }
        Ok(())
    }
}
