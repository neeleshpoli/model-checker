use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaCreateTokenizer, OgaCreateTokenizerStream, OgaDestroyString, OgaDestroyTokenizer,
    OgaDestroyTokenizerStream, OgaTokenizer, OgaTokenizerApplyChatTemplate, OgaTokenizerDecode,
    OgaTokenizerDecodeBatch, OgaTokenizerEncode, OgaTokenizerEncodeBatch,
    OgaTokenizerGetBorTokenId, OgaTokenizerGetBosTokenId, OgaTokenizerGetBotTokenId,
    OgaTokenizerGetEorTokenId, OgaTokenizerGetEosTokenIds, OgaTokenizerGetEotTokenId,
    OgaTokenizerGetPadTokenId, OgaTokenizerStream, OgaTokenizerStreamDecode, OgaTokenizerToTokenId,
    OgaUpdateTokenizerOptions,
};

use crate::error::{Error, Result, check_status};
use crate::model::Model;
use crate::sequences::Sequences;
use crate::string_array::StringArray;
use crate::tensor::Tensor;

pub struct Tokenizer {
    pub(crate) ptr: Arc<Mutex<*mut OgaTokenizer>>,
}

unsafe impl Send for Tokenizer {}
unsafe impl Sync for Tokenizer {}

impl Tokenizer {
    pub fn new(model: &Model) -> Result<Self> {
        let mut ptr: *mut OgaTokenizer = std::ptr::null_mut();
        let model_ptr = model.ptr.lock()?;
        unsafe {
            check_status(OgaCreateTokenizer(*model_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn update_options(&self, keys: &[&str], values: &[&str]) -> Result<()> {
        if keys.len() != values.len() {
            return Err(Error::OgaError(
                "Keys and values length mismatch".to_string(),
            ));
        }

        let ptr = self.ptr.lock()?;

        let c_keys: Vec<CString> = keys
            .iter()
            .map(|s| CString::new(*s).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        let c_keys_ptrs: Vec<*const std::ffi::c_char> =
            c_keys.iter().map(|cs| cs.as_ptr()).collect();

        let c_values: Vec<CString> = values
            .iter()
            .map(|s| CString::new(*s).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        let c_values_ptrs: Vec<*const std::ffi::c_char> =
            c_values.iter().map(|cs| cs.as_ptr()).collect();

        unsafe {
            check_status(OgaUpdateTokenizerOptions(
                *ptr,
                c_keys_ptrs.as_ptr(),
                c_values_ptrs.as_ptr(),
                keys.len(),
            ))?;
        }
        Ok(())
    }

    pub fn get_bos_token_id(&self) -> Result<i32> {
        let ptr = self.ptr.lock()?;
        let mut token_id: i32 = 0;
        unsafe {
            check_status(OgaTokenizerGetBosTokenId(*ptr, &mut token_id))?;
        }
        Ok(token_id)
    }

    pub fn get_eos_token_ids(&self) -> Result<Vec<i32>> {
        let ptr = self.ptr.lock()?;
        let mut token_ids_ptr: *const i32 = std::ptr::null();
        let mut count: usize = 0;
        unsafe {
            check_status(OgaTokenizerGetEosTokenIds(
                *ptr,
                &mut token_ids_ptr,
                &mut count,
            ))?;
            let slice = std::slice::from_raw_parts(token_ids_ptr, count);
            Ok(slice.to_vec())
        }
    }

    pub fn get_pad_token_id(&self) -> Result<i32> {
        let ptr = self.ptr.lock()?;
        let mut token_id: i32 = 0;
        unsafe {
            check_status(OgaTokenizerGetPadTokenId(*ptr, &mut token_id))?;
        }
        Ok(token_id)
    }

    pub fn get_bot_token_id(&self) -> Result<i32> {
        let ptr = self.ptr.lock()?;
        let mut token_id: i32 = 0;
        unsafe {
            check_status(OgaTokenizerGetBotTokenId(*ptr, &mut token_id))?;
        }
        Ok(token_id)
    }

    pub fn get_eot_token_id(&self) -> Result<i32> {
        let ptr = self.ptr.lock()?;
        let mut token_id: i32 = 0;
        unsafe {
            check_status(OgaTokenizerGetEotTokenId(*ptr, &mut token_id))?;
        }
        Ok(token_id)
    }

    pub fn get_bor_token_id(&self) -> Result<i32> {
        let ptr = self.ptr.lock()?;
        let mut token_id: i32 = 0;
        unsafe {
            check_status(OgaTokenizerGetBorTokenId(*ptr, &mut token_id))?;
        }
        Ok(token_id)
    }

    pub fn get_eor_token_id(&self) -> Result<i32> {
        let ptr = self.ptr.lock()?;
        let mut token_id: i32 = 0;
        unsafe {
            check_status(OgaTokenizerGetEorTokenId(*ptr, &mut token_id))?;
        }
        Ok(token_id)
    }

    pub fn encode(&self, str: &str, sequences: &Sequences) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_str = CString::new(str)?;
        let seq_ptr = sequences.ptr.lock()?;
        unsafe {
            check_status(OgaTokenizerEncode(*ptr, c_str.as_ptr(), *seq_ptr))?;
        }
        Ok(())
    }

    pub fn encode_batch(&self, strings: &[&str]) -> Result<Tensor> {
        let ptr = self.ptr.lock()?;
        let c_strings: Vec<CString> = strings
            .iter()
            .map(|s| CString::new(*s).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        let c_ptrs: Vec<*const std::ffi::c_char> = c_strings.iter().map(|cs| cs.as_ptr()).collect();
        let mut tensor_ptr: *mut ort_genai_sys::OgaTensor = std::ptr::null_mut();

        unsafe {
            check_status(OgaTokenizerEncodeBatch(
                *ptr,
                c_ptrs.as_ptr() as *mut _,
                c_ptrs.len(),
                &mut tensor_ptr,
            ))?;
        }
        Ok(Tensor::from_ptr(tensor_ptr))
    }

    pub fn decode_batch(&self, tensor: &Tensor) -> Result<StringArray> {
        let ptr = self.ptr.lock()?;
        let tensor_ptr = tensor.ptr.lock()?;
        let mut out_ptr: *mut ort_genai_sys::OgaStringArray = std::ptr::null_mut();
        unsafe {
            check_status(OgaTokenizerDecodeBatch(*ptr, *tensor_ptr, &mut out_ptr))?;
        }
        Ok(StringArray {
            ptr: Arc::new(Mutex::new(out_ptr)),
        })
    }

    pub fn to_token_id(&self, str: &str) -> Result<i32> {
        let ptr = self.ptr.lock()?;
        let c_str = CString::new(str)?;
        let mut token_id: i32 = 0;
        unsafe {
            check_status(OgaTokenizerToTokenId(*ptr, c_str.as_ptr(), &mut token_id))?;
        }
        Ok(token_id)
    }

    pub fn decode(&self, tokens: &[i32]) -> Result<String> {
        let ptr = self.ptr.lock()?;
        let mut out_str_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaTokenizerDecode(
                *ptr,
                tokens.as_ptr(),
                tokens.len(),
                &mut out_str_ptr,
            ))?;

            let c_str = CStr::from_ptr(out_str_ptr);
            let s = c_str.to_string_lossy().into_owned();
            OgaDestroyString(out_str_ptr);
            Ok(s)
        }
    }

    pub fn apply_chat_template(
        &self,
        template_str: Option<&str>,
        messages: &str,
        tools: Option<&str>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let ptr = self.ptr.lock()?;

        let c_template = template_str.map(|s| CString::new(s).unwrap());
        let c_messages = CString::new(messages)?;
        let c_tools = tools.map(|s| CString::new(s).unwrap());

        let mut out_str_ptr: *const std::ffi::c_char = std::ptr::null();

        unsafe {
            check_status(OgaTokenizerApplyChatTemplate(
                *ptr,
                c_template.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                c_messages.as_ptr(),
                c_tools.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                add_generation_prompt,
                &mut out_str_ptr,
            ))?;

            let c_str = CStr::from_ptr(out_str_ptr);
            let s = c_str.to_string_lossy().into_owned();
            OgaDestroyString(out_str_ptr);
            Ok(s)
        }
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyTokenizer(*ptr);
                }
            }
        }
    }
}

pub struct TokenizerStream {
    pub(crate) ptr: Arc<Mutex<*mut OgaTokenizerStream>>,
}

unsafe impl Send for TokenizerStream {}
unsafe impl Sync for TokenizerStream {}

impl TokenizerStream {
    pub fn new(tokenizer: &Tokenizer) -> Result<Self> {
        let mut ptr: *mut OgaTokenizerStream = std::ptr::null_mut();
        let tok_ptr = tokenizer.ptr.lock()?;
        unsafe {
            check_status(OgaCreateTokenizerStream(*tok_ptr, &mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub fn decode(&self, token: i32) -> Result<String> {
        let ptr = self.ptr.lock()?;
        let mut out_str_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            check_status(OgaTokenizerStreamDecode(*ptr, token, &mut out_str_ptr))?;
            let c_str = CStr::from_ptr(out_str_ptr);
            Ok(c_str.to_string_lossy().into_owned())
        }
    }
}

impl Drop for TokenizerStream {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyTokenizerStream(*ptr);
                }
            }
        }
    }
}
