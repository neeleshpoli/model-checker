use std::{
    ffi::CString,
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateGeneratorParams, OgaDestroyGeneratorParams, OgaGeneratorParams,
    OgaGeneratorParamsGetSearchBool, OgaGeneratorParamsGetSearchNumber,
    OgaGeneratorParamsSetGuidance, OgaGeneratorParamsSetSearchBool,
    OgaGeneratorParamsSetSearchNumber,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
    model::Model,
};

/// Generation parameters associated with a [`Model`].
pub struct GeneratorParams {
    ptr: NonNull<OgaGeneratorParams>,
    _marker: PhantomData<Rc<()>>,
}

impl GeneratorParams {
    /// Creates [`GeneratorParams`] from the given [`Model`].
    ///
    /// # Parameters
    /// * `model` - The model to use for generation.
    pub fn new(model: &Model) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateGeneratorParams(model.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Sets a numerical value for a search parameter.
    ///
    /// # Parameters
    /// * `name` - The name of the search parameter.
    /// * `value` - The value of the search parameter.
    pub fn set_search_number(&mut self, name: &str, value: f64) -> OgaResult<()> {
        let name = CString::new(name)?;

        unsafe { OgaGeneratorParamsSetSearchNumber(self.into(), name.as_ptr(), value) }.to_result()
    }

    /// Sets a boolean value for a search parameter.
    ///
    /// # Parameters
    /// * `name` - The name of the search parameter.
    /// * `value` - The value of the search parameter.
    pub fn set_search_bool(&mut self, name: &str, value: bool) -> OgaResult<()> {
        let name = CString::new(name)?;

        unsafe { OgaGeneratorParamsSetSearchBool(self.into(), name.as_ptr(), value) }.to_result()
    }

    /// Sets the guidance type and data for the generation parameters.
    ///
    /// The supported guidance types are JSON schema, regular expressions, and
    /// Lark grammar. When enabled, forced-forward token generation can force
    /// tokens that satisfy the input grammar without calling the model, which
    /// can speed up generation. This option is only valid when guidance is
    /// enabled and both the batch size and beam size are 1.
    ///
    /// # Parameters
    /// * `guidance_type` - The type of guidance to use.
    /// * `data` - The guidance data.
    /// * `enable_ff_tokens` - Whether to enable forced-forward token generation.
    pub fn set_guidance(
        &mut self,
        guidance_type: GuidanceTypes,
        data: &str,
        enable_ff_tokens: bool,
    ) -> OgaResult<()> {
        let data = CString::new(data)?;

        unsafe {
            OgaGeneratorParamsSetGuidance(
                self.into(),
                guidance_type.into(),
                data.as_ptr(),
                enable_ff_tokens,
            )
        }
        .to_result()
    }

    /// Gets a numerical value for a search parameter.
    ///
    /// # Parameters
    /// * `name` - The name of the search parameter.
    pub fn get_search_number(&self, name: &str) -> OgaResult<f64> {
        let name = CString::new(name)?;
        let mut value = MaybeUninit::<f64>::uninit();

        unsafe {
            OgaGeneratorParamsGetSearchNumber(self.into(), name.as_ptr(), value.as_mut_ptr())
        }
        .to_result()?;

        Ok(unsafe { value.assume_init() })
    }

    /// Gets a boolean value for a search parameter.
    ///
    /// # Parameters
    /// * `name` - The name of the search parameter.
    pub fn get_search_bool(&self, name: &str) -> OgaResult<bool> {
        let name = CString::new(name)?;
        let mut value = MaybeUninit::<bool>::uninit();

        unsafe { OgaGeneratorParamsGetSearchBool(self.into(), name.as_ptr(), value.as_mut_ptr()) }
            .to_result()?;

        Ok(unsafe { value.assume_init() })
    }
}

/// Guidance formats supported by [`GeneratorParams::set_guidance`].
pub enum GuidanceTypes {
    /// A JSON Schema describing the expected output.
    JsonSchema,
    /// A regular expression describing the expected output.
    Regex,
    /// A Lark grammar describing the expected output.
    LarkGrammer,
}

impl From<GuidanceTypes> for *const std::ffi::c_char {
    fn from(value: GuidanceTypes) -> Self {
        match value {
            GuidanceTypes::JsonSchema => c"json_schema".as_ptr(),
            GuidanceTypes::Regex => c"regex".as_ptr(),
            GuidanceTypes::LarkGrammer => c"lark_grammer".as_ptr(),
        }
    }
}

impl_into_ptr! {GeneratorParams, OgaGeneratorParams}
impl_from_ptr! {GeneratorParams, OgaGeneratorParams}
impl_drop! {GeneratorParams, OgaDestroyGeneratorParams}
