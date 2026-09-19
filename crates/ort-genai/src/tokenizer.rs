use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateTokenizer, OgaDestroyString, OgaDestroyTokenizer, OgaTokenizer,
    OgaTokenizerApplyChatTemplate, OgaTokenizerDecode, OgaTokenizerDecodeBatch, OgaTokenizerEncode,
    OgaTokenizerEncodeBatch, OgaTokenizerGetBorTokenId, OgaTokenizerGetBosTokenId,
    OgaTokenizerGetBotTokenId, OgaTokenizerGetEorTokenId, OgaTokenizerGetEosTokenIds,
    OgaTokenizerGetEotTokenId, OgaTokenizerGetPadTokenId, OgaTokenizerToTokenId,
    OgaUpdateTokenizerOptions,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
    model::Model,
    sequences::Sequences,
    string_array::StringArray,
    tensor::Tensor,
};

/// Encodes text into token IDs and decodes token IDs into text.
pub struct Tokenizer {
    ptr: NonNull<OgaTokenizer>,
    _marker: PhantomData<Rc<()>>,
}

impl Tokenizer {
    /// Creates a [`Tokenizer`] for the given [`Model`].
    ///
    /// # Parameters
    /// * `model` - The model whose tokenizer configuration should be used.
    pub fn new(model: &Model) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateTokenizer(model.into(), &mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Updates tokenizer options.
    ///
    /// Supported options are [`TokenizerOption::AddSpecialTokens`] and
    /// [`TokenizerOption::SkipSpecialTokens`]. Unknown options are rejected.
    ///
    /// # Parameters
    /// * `options` - The tokenizer options to apply.
    pub fn update_options(&mut self, options: HashSet<TokenizerOption>) -> OgaResult<()> {
        let mut key_cstrings = Vec::with_capacity(options.len());
        let mut val_cstrings = Vec::with_capacity(options.len());

        for option in options {
            let (key, value) = option.into();
            key_cstrings.push(CString::new(key)?);
            val_cstrings.push(CString::new(value)?);
        }

        let keys: Vec<_> = key_cstrings.iter().map(|s| s.as_ptr()).collect();
        let values: Vec<_> = val_cstrings.iter().map(|s| s.as_ptr()).collect();

        unsafe {
            OgaUpdateTokenizerOptions(self.into(), keys.as_ptr(), values.as_ptr(), keys.len())
        }
        .to_result()
    }

    /// Returns the beginning-of-sequence token ID.
    pub fn get_bos_token_id(&self) -> OgaResult<i32> {
        let mut token = MaybeUninit::<i32>::uninit();

        unsafe { OgaTokenizerGetBosTokenId(self.into(), token.as_mut_ptr()) }.to_result()?;
        Ok(unsafe { token.assume_init() })
    }

    /// Returns the end-of-sequence token IDs.
    ///
    /// The returned slice is owned by the tokenizer and remains valid while
    /// the tokenizer is alive.
    pub fn get_eos_token_ids(&self) -> OgaResult<&[i32]> {
        let mut ptr = ptr::null();
        let mut size = MaybeUninit::<usize>::uninit();

        unsafe { OgaTokenizerGetEosTokenIds(self.into(), &mut ptr, size.as_mut_ptr()) }
            .to_result()?;

        Ok(unsafe { std::slice::from_raw_parts(ptr, size.assume_init()) })
    }

    /// Returns the padding token ID.
    pub fn get_pad_token_id(&self) -> OgaResult<i32> {
        let mut token = MaybeUninit::<i32>::uninit();

        unsafe { OgaTokenizerGetPadTokenId(self.into(), token.as_mut_ptr()) }.to_result()?;
        Ok(unsafe { token.assume_init() })
    }

    /// Returns the beginning-of-tool-call token ID.
    ///
    /// Returns an error if the model does not define one.
    pub fn get_bot_token_id(&self) -> OgaResult<i32> {
        let mut token = MaybeUninit::<i32>::uninit();

        unsafe { OgaTokenizerGetBotTokenId(self.into(), token.as_mut_ptr()) }.to_result()?;
        Ok(unsafe { token.assume_init() })
    }

    /// Returns the end-of-tool-call token ID.
    ///
    /// Returns an error if the model does not define one.
    pub fn get_eot_token_id(&self) -> OgaResult<i32> {
        let mut token = MaybeUninit::<i32>::uninit();

        unsafe { OgaTokenizerGetEotTokenId(self.into(), token.as_mut_ptr()) }.to_result()?;
        Ok(unsafe { token.assume_init() })
    }

    /// Returns the beginning-of-reasoning token ID.
    ///
    /// Returns an error if the model does not define one.
    pub fn get_bor_token_id(&self) -> OgaResult<i32> {
        let mut token = MaybeUninit::<i32>::uninit();

        unsafe { OgaTokenizerGetBorTokenId(self.into(), token.as_mut_ptr()) }.to_result()?;
        Ok(unsafe { token.assume_init() })
    }

    /// Returns the end-of-reasoning token ID.
    ///
    /// Returns an error if the model does not define one.
    pub fn get_eor_token_id(&self) -> OgaResult<i32> {
        let mut token = MaybeUninit::<i32>::uninit();

        unsafe { OgaTokenizerGetEorTokenId(self.into(), token.as_mut_ptr()) }.to_result()?;
        Ok(unsafe { token.assume_init() })
    }

    /// Encodes a string and appends its token sequence to [`Sequences`].
    ///
    /// # Parameters
    /// * `string` - The text to encode.
    /// * `sequences` - The sequence collection to which the encoded tokens are appended.
    pub fn encode(&self, string: &str, sequences: &mut Sequences) -> OgaResult<()> {
        let string = CString::new(string)?;

        unsafe { OgaTokenizerEncode(self.into(), string.as_ptr(), sequences.into()) }.to_result()
    }

    /// Encodes multiple strings into a token-ID [`Tensor`].
    ///
    /// # Parameters
    /// * `strings` - The strings to encode.
    pub fn encode_batch(&self, strings: &mut [&str]) -> OgaResult<Tensor<'static>> {
        let mut tensor = ptr::null_mut();

        let cstrings = strings
            .iter()
            .map(|string| CString::new(*string))
            .collect::<Result<Vec<_>, _>>()?;
        let mut string_ptrs = cstrings
            .iter()
            .map(|cstring| cstring.as_ptr())
            .collect::<Vec<_>>();

        unsafe {
            OgaTokenizerEncodeBatch(
                self.into(),
                string_ptrs.as_mut_ptr(),
                string_ptrs.len(),
                &mut tensor,
            )
        }
        .to_result()?;

        Tensor::try_from(tensor)
    }

    /// Decodes a tensor of token IDs into a [`Vec<String>`].
    ///
    /// # Parameters
    /// * `tensor` - The tensor containing token IDs to decode.
    pub fn decode_batch<'data>(&self, tensor: Tensor<'data>) -> OgaResult<Vec<String>> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaTokenizerDecodeBatch(self.into(), tensor.into(), &mut ptr) }
            .to_result()?;

        TryInto::<StringArray>::try_into(ptr)?.try_into()
    }

    /// Converts a string to a single token ID.
    ///
    /// # Parameters
    /// * `string` - The string to convert.
    pub fn to_token_id(&self, string: &str) -> OgaResult<i32> {
        let string = CString::new(string)?;
        let mut token = MaybeUninit::uninit();

        unsafe {
            OgaTokenizerToTokenId(self.into(), string.as_ptr(), token.as_mut_ptr()).to_result()?;
            Ok(token.assume_init())
        }
    }

    /// Decodes a token sequence into an owned UTF-8 string.
    ///
    /// # Parameters
    /// * `tokens` - The token IDs to decode.
    pub fn decode(&self, tokens: &[i32]) -> OgaResult<String> {
        let mut ptr = ptr::null();

        unsafe { OgaTokenizerDecode(self.into(), tokens.as_ptr(), tokens.len(), &mut ptr) }
            .to_result()?;

        let value_ptr =
            NonNull::new(ptr as *mut std::os::raw::c_char).ok_or(crate::error::Error::NulError)?;
        let result = unsafe { CStr::from_ptr(value_ptr.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        unsafe { OgaDestroyString(value_ptr.as_ptr()) };

        Ok(result)
    }

    /// Applies a chat template to input messages.
    ///
    /// When `template` is `None`, the default template from the tokenizer
    /// configuration is used. When `tools` is `None`, no tool calls are
    /// included. The returned string is owned by Rust.
    ///
    /// # Parameters
    /// * `template` - An optional chat template. `None` uses the configured default.
    /// * `messages` - The input messages to process.
    /// * `tools` - Optional chat tool calls.
    /// * `add_generation_prompt` - Whether to append a generation prompt.
    pub fn apply_chat_template(
        &self,
        template: Option<&str>,
        messages: &str,
        tools: Option<&str>,
        add_generation_prompt: bool,
    ) -> OgaResult<String> {
        let template = template.map(CString::new).transpose()?;
        let messages = CString::new(messages)?;
        let tools = tools.map(CString::new).transpose()?;

        let mut ptr = ptr::null();

        unsafe {
            OgaTokenizerApplyChatTemplate(
                self.into(),
                template.as_deref().map_or(ptr::null(), |s| s.as_ptr()),
                messages.as_ptr(),
                tools.as_deref().map_or(ptr::null(), |s| s.as_ptr()),
                add_generation_prompt,
                &mut ptr,
            )
        }
        .to_result()?;

        let value_ptr =
            NonNull::new(ptr as *mut std::os::raw::c_char).ok_or(crate::error::Error::NulError)?;
        let result = unsafe { CStr::from_ptr(value_ptr.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        unsafe { OgaDestroyString(value_ptr.as_ptr()) };

        Ok(result)
    }
}

/// Options that can be updated on a [`Tokenizer`].
pub enum TokenizerOption {
    /// Controls whether special tokens such as BOS and EOS are added during encoding.
    AddSpecialTokens(bool),
    /// Controls whether special tokens are removed during decoding.
    SkipSpecialTokens(bool),
}

impl Into<(&str, &str)> for TokenizerOption {
    fn into(self) -> (&'static str, &'static str) {
        match self {
            TokenizerOption::AddSpecialTokens(value) => {
                ("add_special_tokens", if value { "1" } else { "0" })
            }
            TokenizerOption::SkipSpecialTokens(value) => {
                ("skip_special_tokens", if value { "1" } else { "0" })
            }
        }
    }
}

impl_from_ptr! {Tokenizer, OgaTokenizer}
impl_into_ptr! {Tokenizer, OgaTokenizer}
impl_drop! {Tokenizer, OgaDestroyTokenizer}
