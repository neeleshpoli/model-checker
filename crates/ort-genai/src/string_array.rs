use std::{
    ffi::{CStr, CString, NulError, c_char}, marker::PhantomData, mem::MaybeUninit, ptr::{self, NonNull}, rc::Rc,
};

use ort_genai_sys::{
    OgaCreateStringArray, OgaDestroyStringArray, OgaStringArray, OgaStringArrayAddString, OgaStringArrayGetCount, OgaStringArrayGetString,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
};

pub struct StringArray {
    ptr: NonNull<OgaStringArray>,
    _marker: PhantomData<Rc<()>>,
}

impl StringArray {
    fn new() -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();
        unsafe { OgaCreateStringArray(&mut ptr) }.to_result()?;
        Self::try_from(ptr)
    }

    fn new_from_buffer<S: AsRef<str>>(strings: &[S]) -> OgaResult<Self> {
        let strings = strings
            .iter()
            .map(|s| CString::new(s.as_ref()))
            .collect::<Result<Vec<_>, NulError>>()?;

        let mut array = Self::new()?;

        for string in strings {
            array.add_string(string.as_ptr())?;
        }

        Ok(array)
    }

    fn add_string(&mut self, string: *const c_char) -> OgaResult<()> {
        unsafe { OgaStringArrayAddString(self.into(), string) }.to_result()
    }

    fn get_string(&self, index: usize) -> OgaResult<String> {
        let mut string = ptr::null();

        unsafe {OgaStringArrayGetString(self.into(), index, &mut string)}.to_result()?;

        Ok(unsafe { CStr::from_ptr(string).to_str()? }.to_owned())
    }

    fn get_count(&self) -> OgaResult<usize> {
        let mut count = MaybeUninit::uninit();

        unsafe { OgaStringArrayGetCount(self.into(), count.as_mut_ptr()) }.to_result()?;

        Ok(unsafe { count.assume_init() })
    }
}

impl<S: AsRef<str>> TryFrom<&[S]> for StringArray {
    type Error = crate::Error;

    fn try_from(value: &[S]) -> OgaResult<Self> {
        Self::new_from_buffer(value)
    }
}

impl TryFrom<StringArray> for Vec<String> {
    type Error = crate::Error;

    fn try_from(value: StringArray) -> Result<Self, Self::Error> {
        let count = value.get_count()?;
        let mut strings = Vec::with_capacity(count);

        for index in 0..count {
            strings.push(value.get_string(index)?);
        }
        
        Ok(strings)
    }
}

impl_from_ptr! {StringArray, OgaStringArray}
impl_into_ptr! {StringArray, OgaStringArray}
impl_drop! {StringArray, OgaDestroyStringArray}
