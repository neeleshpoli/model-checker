use std::{
    ffi::CString,
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateNamedTensors, OgaDestroyNamedTensors, OgaNamedTensors, OgaNamedTensorsCount,
    OgaNamedTensorsDelete, OgaNamedTensorsGet, OgaNamedTensorsGetNames, OgaNamedTensorsSet,
};

use crate::{
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
    string_array::StringArray,
    tensor::Tensor,
};

/// A collection of tensors addressed by name.
pub struct NamedTensors<'data> {
    ptr: NonNull<OgaNamedTensors>,
    _type: PhantomData<&'data mut [u8]>,
    _marker: PhantomData<Rc<()>>,
}

impl<'data> NamedTensors<'data> {
    /// Creates an empty [`NamedTensors`] collection.
    pub fn new() -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();

        unsafe { OgaCreateNamedTensors(&mut ptr) }.to_result()?;

        Self::try_from(ptr)
    }

    /// Looks up a tensor in this collection by name.
    ///
    /// # Parameters
    /// * `name` - The name of the tensor to find.
    pub fn get_tensor(&self, name: &str) -> OgaResult<Tensor<'data>> {
        let name = CString::new(name)?;
        let mut ptr = ptr::null_mut();

        unsafe { OgaNamedTensorsGet(self.ptr.as_ptr(), name.as_ptr(), &mut ptr) }.to_result()?;

        ptr.try_into()
    }

    /// Sets a tensor in this collection by name.
    ///
    /// If a tensor already exists under the name, it is replaced.
    ///
    /// # Parameters
    /// * `name` - The name under which to store the tensor.
    /// * `tensor` - The tensor to store.
    pub fn set_tensor(&self, name: &str, tensor: &mut Tensor) -> OgaResult<()> {
        let name = CString::new(name)?;

        unsafe { OgaNamedTensorsSet(self.ptr.as_ptr(), name.as_ptr(), tensor.into()) }
            .to_result()?;

        Ok(())
    }

    /// Deletes a tensor from this collection by name.
    ///
    /// # Parameters
    /// * `name` - The name of the tensor to remove.
    pub fn delete_tensor(&self, name: &str) -> OgaResult<()> {
        let name = CString::new(name)?;

        unsafe { OgaNamedTensorsDelete(self.ptr.as_ptr(), name.as_ptr()) }.to_result()?;

        Ok(())
    }

    /// Returns the number of tensors in this collection.
    pub fn get_count(&self) -> OgaResult<usize> {
        let mut count = MaybeUninit::<usize>::uninit();
        unsafe { OgaNamedTensorsCount(self.ptr.as_ptr(), count.as_mut_ptr()) }.to_result()?;

        Ok(unsafe { count.assume_init() })
    }

    /// Returns a [`Vec<String>`] containing the names of all tensors in this
    /// collection.
    pub fn get_names(&self) -> OgaResult<Vec<String>> {
        let mut ptr = ptr::null_mut();
        unsafe { OgaNamedTensorsGetNames(self.ptr.as_ptr(), &mut ptr) }.to_result()?;

        TryInto::<StringArray>::try_into(ptr)?.try_into()
    }
}

impl_into_ptr! {NamedTensors<'data>, OgaNamedTensors}
impl_from_ptr! {NamedTensors<'data>, OgaNamedTensors}
impl_drop! {NamedTensors<'data>, OgaDestroyNamedTensors}
