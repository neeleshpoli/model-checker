use std::ffi::CString;
use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaCreateGenerator, OgaCreateGeneratorParams, OgaDestroyGenerator, OgaDestroyGeneratorParams,
    OgaGenerator, OgaGenerator_AppendTokenSequences, OgaGenerator_AppendTokens,
    OgaGenerator_GenerateNextToken, OgaGenerator_GetInput, OgaGenerator_GetLogits,
    OgaGenerator_GetNextTokens, OgaGenerator_GetOutput, OgaGenerator_GetSequenceCount,
    OgaGenerator_GetSequenceData, OgaGenerator_IsDone, OgaGenerator_RewindTo,
    OgaGenerator_SetInputs, OgaGenerator_SetLogits, OgaGenerator_SetModelInput,
    OgaGenerator_SetRuntimeOption, OgaGenerator_TokenCount, OgaGeneratorParams,
    OgaGeneratorParamsGetSearchBool, OgaGeneratorParamsGetSearchNumber,
    OgaGeneratorParamsSetGuidance, OgaGeneratorParamsSetSearchBool,
    OgaGeneratorParamsSetSearchNumber,
};

use crate::error::{Result, check_status};
use crate::model::Model;
use crate::sequences::Sequences;
use crate::tensor::{NamedTensors, Tensor};

pub struct GeneratorParams {
    pub(crate) ptr: Arc<Mutex<*mut OgaGeneratorParams>>,
}

unsafe impl Send for GeneratorParams {}
unsafe impl Sync for GeneratorParams {}

impl GeneratorParams {
    pub fn new(model: &Model) -> Result<Self> {
        let mut ptr: *mut OgaGeneratorParams = std::ptr::null_mut();
        let model_ptr = model.ptr.lock()?;
        unsafe {
            check_status(OgaCreateGeneratorParams(*model_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn set_search_number(&self, name: &str, value: f64) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        unsafe {
            check_status(OgaGeneratorParamsSetSearchNumber(
                *ptr,
                c_name.as_ptr(),
                value,
            ))?;
        }
        Ok(())
    }

    pub fn set_search_bool(&self, name: &str, value: bool) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        unsafe {
            check_status(OgaGeneratorParamsSetSearchBool(
                *ptr,
                c_name.as_ptr(),
                value,
            ))?;
        }
        Ok(())
    }

    pub fn set_guidance(&self, type_: &str, data: &str, enable_ff_tokens: bool) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_type = CString::new(type_)?;
        let c_data = CString::new(data)?;
        unsafe {
            check_status(OgaGeneratorParamsSetGuidance(
                *ptr,
                c_type.as_ptr(),
                c_data.as_ptr(),
                enable_ff_tokens,
            ))?;
        }
        Ok(())
    }

    pub fn get_search_number(&self, name: &str) -> Result<f64> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        let mut value = 0.0;
        unsafe {
            check_status(OgaGeneratorParamsGetSearchNumber(
                *ptr,
                c_name.as_ptr(),
                &mut value,
            ))?;
        }
        Ok(value)
    }

    pub fn get_search_bool(&self, name: &str) -> Result<bool> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        let mut value = false;
        unsafe {
            check_status(OgaGeneratorParamsGetSearchBool(
                *ptr,
                c_name.as_ptr(),
                &mut value,
            ))?;
        }
        Ok(value)
    }
}

impl Drop for GeneratorParams {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyGeneratorParams(*ptr);
                }
            }
        }
    }
}

pub struct Generator {
    pub(crate) ptr: Arc<Mutex<*mut OgaGenerator>>,
}

unsafe impl Send for Generator {}
unsafe impl Sync for Generator {}

impl Generator {
    pub fn new(model: &Model, params: &GeneratorParams) -> Result<Self> {
        let mut ptr: *mut OgaGenerator = std::ptr::null_mut();
        let model_ptr = model.ptr.lock()?;
        let params_ptr = params.ptr.lock()?;
        unsafe {
            check_status(OgaCreateGenerator(*model_ptr, *params_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn is_done(&self) -> Result<bool> {
        let ptr = self.ptr.lock()?;
        unsafe { Ok(OgaGenerator_IsDone(*ptr)) }
    }

    pub fn generate_next_token(&self) -> Result<()> {
        let ptr = self.ptr.lock()?;
        unsafe {
            check_status(OgaGenerator_GenerateNextToken(*ptr))?;
        }
        Ok(())
    }

    pub fn append_tokens(&self, tokens: &[i32]) -> Result<()> {
        let ptr = self.ptr.lock()?;
        unsafe {
            check_status(OgaGenerator_AppendTokens(
                *ptr,
                tokens.as_ptr(),
                tokens.len(),
            ))?;
        }
        Ok(())
    }

    pub fn append_token_sequences(&self, sequences: &Sequences) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let seq_ptr = sequences.ptr.lock()?;
        unsafe {
            check_status(OgaGenerator_AppendTokenSequences(*ptr, *seq_ptr))?;
        }
        Ok(())
    }

    pub fn set_model_input(&self, name: &str, tensor: &Tensor) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        let tensor_ptr = tensor.ptr.lock()?;
        unsafe {
            check_status(OgaGenerator_SetModelInput(
                *ptr,
                c_name.as_ptr(),
                *tensor_ptr,
            ))?;
        }
        Ok(())
    }

    pub fn set_inputs(&self, inputs: &NamedTensors) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let inputs_ptr = inputs.ptr.lock()?;
        unsafe {
            check_status(OgaGenerator_SetInputs(*ptr, *inputs_ptr))?;
        }
        Ok(())
    }

    pub fn token_count(&self) -> Result<usize> {
        let ptr = self.ptr.lock()?;
        unsafe { Ok(OgaGenerator_TokenCount(*ptr)) }
    }

    pub fn get_next_tokens(&self) -> Result<Vec<i32>> {
        let ptr = self.ptr.lock()?;
        let mut out_ptr: *const i32 = std::ptr::null();
        let mut out_count: usize = 0;
        unsafe {
            check_status(OgaGenerator_GetNextTokens(
                *ptr,
                &mut out_ptr,
                &mut out_count,
            ))?;
            let slice = std::slice::from_raw_parts(out_ptr, out_count);
            Ok(slice.to_vec())
        }
    }

    pub fn set_runtime_option(&self, key: &str, value: &str) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_key = CString::new(key)?;
        let c_value = CString::new(value)?;
        unsafe {
            check_status(OgaGenerator_SetRuntimeOption(
                *ptr,
                c_key.as_ptr(),
                c_value.as_ptr(),
            ))?;
        }
        Ok(())
    }

    pub fn rewind_to(&self, new_length: usize) -> Result<()> {
        let ptr = self.ptr.lock()?;
        unsafe {
            check_status(OgaGenerator_RewindTo(*ptr, new_length))?;
        }
        Ok(())
    }

    pub fn get_input(&self, name: &str) -> Result<Tensor> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        let mut tensor_ptr: *mut ort_genai_sys::OgaTensor = std::ptr::null_mut();
        unsafe {
            check_status(OgaGenerator_GetInput(
                *ptr,
                c_name.as_ptr(),
                &mut tensor_ptr,
            ))?;
        }
        Ok(Tensor::from_ptr(tensor_ptr))
    }

    pub fn get_output(&self, name: &str) -> Result<Tensor> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        let mut tensor_ptr: *mut ort_genai_sys::OgaTensor = std::ptr::null_mut();
        unsafe {
            check_status(OgaGenerator_GetOutput(
                *ptr,
                c_name.as_ptr(),
                &mut tensor_ptr,
            ))?;
        }
        Ok(Tensor::from_ptr(tensor_ptr))
    }

    pub fn get_logits(&self) -> Result<Tensor> {
        let ptr = self.ptr.lock()?;
        let mut tensor_ptr: *mut ort_genai_sys::OgaTensor = std::ptr::null_mut();
        unsafe {
            check_status(OgaGenerator_GetLogits(*ptr, &mut tensor_ptr))?;
        }
        Ok(Tensor::from_ptr(tensor_ptr))
    }

    pub fn set_logits(&self, tensor: &Tensor) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let tensor_ptr = tensor.ptr.lock()?;
        unsafe {
            check_status(OgaGenerator_SetLogits(*ptr, *tensor_ptr))?;
        }
        Ok(())
    }

    pub fn get_sequence_count(&self, index: usize) -> Result<usize> {
        let ptr = self.ptr.lock()?;
        unsafe { Ok(OgaGenerator_GetSequenceCount(*ptr, index)) }
    }

    pub fn get_sequence_data(&self, index: usize) -> Result<Vec<i32>> {
        let ptr = self.ptr.lock()?;
        unsafe {
            let count = OgaGenerator_GetSequenceCount(*ptr, index);
            let data_ptr = OgaGenerator_GetSequenceData(*ptr, index);
            if data_ptr.is_null() {
                return Ok(Vec::new());
            }
            let slice = std::slice::from_raw_parts(data_ptr, count);
            Ok(slice.to_vec())
        }
    }
}

impl Drop for Generator {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyGenerator(*ptr);
                }
            }
        }
    }
}

impl Generator {
    pub fn is_session_terminated(&self) -> Result<bool> {
        let ptr = self.ptr.lock()?;
        unsafe { Ok(ort_genai_sys::OgaGenerator_IsSessionTerminated(*ptr)) }
    }
}
