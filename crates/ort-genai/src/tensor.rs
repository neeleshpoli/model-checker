use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    rc::Rc,
};

use ort_genai_sys::{
    OgaCreateTensorFromBuffer, OgaDestroyTensor, OgaTensor, OgaTensorGetData, OgaTensorGetShape,
    OgaTensorGetShapeRank, OgaTensorGetType,
};

use crate::{
    element_type::{ElementType, TensorElement},
    error::{OgaResult, OgaResultExt},
    impl_drop, impl_from_ptr, impl_into_ptr,
};

/// A tensor managed by ONNX Runtime GenAI.
pub struct Tensor<'data> {
    ptr: NonNull<OgaTensor>,
    _type: PhantomData<&'data mut [u8]>,
    _marker: PhantomData<Rc<()>>,
}

impl<'data> Tensor<'data> {
    /// Creates a [`Tensor`] from an optional caller-owned buffer.
    ///
    /// If a buffer is supplied, the tensor does not own that memory. The caller
    /// must keep the buffer valid for the lifetime of the returned [`Tensor`].
    /// If `data` is `None`, the tensor allocates its own storage.
    ///
    /// # Parameters
    /// * `data` - An optional caller-owned buffer. When supplied, it must remain
    ///   valid for the lifetime of the returned tensor.
    /// * `shape` - The tensor dimensions. For example, `[1, 20, 30]` describes
    ///   a three-dimensional tensor with those dimensions.
    pub fn new_from_buffer<T: TensorElement>(
        data: Option<&'data mut [T]>,
        shape: &[i64],
    ) -> OgaResult<Self> {
        let mut ptr = ptr::null_mut();
        let data_ptr = data.map_or(ptr::null_mut(), |data| data.as_mut_ptr().cast());

        unsafe {
            OgaCreateTensorFromBuffer(data_ptr, shape.as_ptr(), shape.len(), T::OGA_TYPE, &mut ptr)
        }
        .to_result()?;

        Self::try_from(ptr)
    }

    /// Returns the [`ElementType`] of the tensor data.
    pub fn get_type(&self) -> OgaResult<ElementType> {
        let mut tensor_type = MaybeUninit::uninit();
        unsafe { OgaTensorGetType(self.ptr.as_ptr(), tensor_type.as_mut_ptr()) }.to_result()?;

        Ok(unsafe { tensor_type.assume_init() }.into())
    }

    /// Returns the number of dimensions in the tensor shape.
    ///
    /// This is typically used to allocate a buffer of this size before calling
    /// `get_shape`.
    pub fn get_shape_rank(&self) -> OgaResult<usize> {
        let mut rank = MaybeUninit::uninit();

        unsafe { OgaTensorGetShapeRank(self.ptr.as_ptr(), rank.as_mut_ptr()) }.to_result()?;

        Ok(unsafe { rank.assume_init() })
    }

    /// Returns the tensor shape dimensions as a vector.
    ///
    /// The length of the returned vector must match the value returned by
    /// `get_shape_rank`.
    pub fn get_shape(&self) -> OgaResult<Vec<i64>> {
        let rank = self.get_shape_rank()?;
        let mut shape = Vec::with_capacity(rank);

        unsafe { OgaTensorGetShape(self.ptr.as_ptr(), shape.as_mut_ptr(), rank) }.to_result()?;

        Ok(shape)
    }

    /// Returns the tensor data as a slice of the requested element type.
    ///
    /// # Safety
    /// The caller must ensure the tensor and any code using the returned slice
    /// obey the lifetime and aliasing requirements of the underlying native
    /// tensor. The element type is checked against the tensor's runtime type,
    /// but the method remains unsafe because the native buffer is exposed
    /// directly.
    pub unsafe fn get_data<T: TensorElement>(&self) -> OgaResult<&[T]> {
        let element_type = self.get_type()?;
        if element_type != T::OGA_TYPE.into() {
            return Err(crate::error::Error::TensorTypeMismatch);
        }

        let element_count = self
            .get_shape()?
            .into_iter()
            .try_fold(1usize, |count, dimension| {
                usize::try_from(dimension)
                    .ok()
                    .and_then(|dimension| count.checked_mul(dimension))
            })
            .ok_or(crate::error::Error::InvalidTensorShape)?;

        let mut ptr = ptr::null_mut();
        unsafe { OgaTensorGetData(self.ptr.as_ptr(), &mut ptr) }.to_result()?;

        if element_count > 0 {
            let data_ptr = NonNull::new(ptr).ok_or(crate::error::Error::NulError)?;
            return Ok(unsafe {
                std::slice::from_raw_parts(data_ptr.as_ptr().cast(), element_count)
            });
        }

        let data_ptr = NonNull::new(ptr).unwrap_or_else(NonNull::dangling);
        Ok(unsafe { std::slice::from_raw_parts(data_ptr.as_ptr().cast(), 0) })
    }
}

impl_from_ptr! {Tensor<'data>, OgaTensor}
impl_into_ptr! {Tensor<'data>, OgaTensor}
impl_drop! {Tensor<'data>, OgaDestroyTensor}
