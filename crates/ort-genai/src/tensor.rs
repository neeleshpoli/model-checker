use std::ffi::CString;

use ort_genai_sys::{
    OgaCreateNamedTensors, OgaCreateTensorFromBuffer, OgaDestroyNamedTensors, OgaDestroyTensor,
    OgaElementType, OgaNamedTensors, OgaNamedTensorsCount, OgaNamedTensorsDelete,
    OgaNamedTensorsGet, OgaNamedTensorsGetNames, OgaNamedTensorsSet, OgaTensor, OgaTensorGetData,
    OgaTensorGetShape, OgaTensorGetShapeRank, OgaTensorGetType,
};

use crate::error::{check_status, Result};
use crate::string_array::StringArray;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    Undefined,
    Float32,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Int32,
    Int64,
    String,
    Bool,
    Float16,
    Float64,
    Uint32,
    Uint64,
    Complex64,
    Complex128,
    Bfloat16,
}

impl From<OgaElementType> for ElementType {
    fn from(t: OgaElementType) -> Self {
        match t {
            ort_genai_sys::OgaElementType_OgaElementType_undefined => ElementType::Undefined,
            ort_genai_sys::OgaElementType_OgaElementType_float32 => ElementType::Float32,
            ort_genai_sys::OgaElementType_OgaElementType_uint8 => ElementType::Uint8,
            ort_genai_sys::OgaElementType_OgaElementType_int8 => ElementType::Int8,
            ort_genai_sys::OgaElementType_OgaElementType_uint16 => ElementType::Uint16,
            ort_genai_sys::OgaElementType_OgaElementType_int16 => ElementType::Int16,
            ort_genai_sys::OgaElementType_OgaElementType_int32 => ElementType::Int32,
            ort_genai_sys::OgaElementType_OgaElementType_int64 => ElementType::Int64,
            ort_genai_sys::OgaElementType_OgaElementType_string => ElementType::String,
            ort_genai_sys::OgaElementType_OgaElementType_bool => ElementType::Bool,
            ort_genai_sys::OgaElementType_OgaElementType_float16 => ElementType::Float16,
            ort_genai_sys::OgaElementType_OgaElementType_float64 => ElementType::Float64,
            ort_genai_sys::OgaElementType_OgaElementType_uint32 => ElementType::Uint32,
            ort_genai_sys::OgaElementType_OgaElementType_uint64 => ElementType::Uint64,
            ort_genai_sys::OgaElementType_OgaElementType_complex64 => ElementType::Complex64,
            ort_genai_sys::OgaElementType_OgaElementType_complex128 => ElementType::Complex128,
            ort_genai_sys::OgaElementType_OgaElementType_bfloat16 => ElementType::Bfloat16,
            _ => ElementType::Undefined,
        }
    }
}

impl From<ElementType> for OgaElementType {
    fn from(t: ElementType) -> Self {
        match t {
            ElementType::Undefined => ort_genai_sys::OgaElementType_OgaElementType_undefined,
            ElementType::Float32 => ort_genai_sys::OgaElementType_OgaElementType_float32,
            ElementType::Uint8 => ort_genai_sys::OgaElementType_OgaElementType_uint8,
            ElementType::Int8 => ort_genai_sys::OgaElementType_OgaElementType_int8,
            ElementType::Uint16 => ort_genai_sys::OgaElementType_OgaElementType_uint16,
            ElementType::Int16 => ort_genai_sys::OgaElementType_OgaElementType_int16,
            ElementType::Int32 => ort_genai_sys::OgaElementType_OgaElementType_int32,
            ElementType::Int64 => ort_genai_sys::OgaElementType_OgaElementType_int64,
            ElementType::String => ort_genai_sys::OgaElementType_OgaElementType_string,
            ElementType::Bool => ort_genai_sys::OgaElementType_OgaElementType_bool,
            ElementType::Float16 => ort_genai_sys::OgaElementType_OgaElementType_float16,
            ElementType::Float64 => ort_genai_sys::OgaElementType_OgaElementType_float64,
            ElementType::Uint32 => ort_genai_sys::OgaElementType_OgaElementType_uint32,
            ElementType::Uint64 => ort_genai_sys::OgaElementType_OgaElementType_uint64,
            ElementType::Complex64 => ort_genai_sys::OgaElementType_OgaElementType_complex64,
            ElementType::Complex128 => ort_genai_sys::OgaElementType_OgaElementType_complex128,
            ElementType::Bfloat16 => ort_genai_sys::OgaElementType_OgaElementType_bfloat16,
        }
    }
}

use std::marker::PhantomData;

pub struct Tensor<'a> {
    pub(crate) ptr: *mut OgaTensor,
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> Tensor<'a> {
    pub fn from_buffer<T>(
        data: &'a mut [T],
        shape: &[i64],
        element_type: ElementType,
    ) -> Result<Self> {
        let mut ptr: *mut OgaTensor = std::ptr::null_mut();
        let oga_type = element_type.into();
        unsafe {
            check_status(OgaCreateTensorFromBuffer(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                shape.as_ptr(),
                shape.len(),
                oga_type,
                &mut ptr,
            ))?;
        }
        Ok(Self {
            ptr: ptr,
            _marker: PhantomData,
        })
    }

    pub(crate) fn from_ptr(ptr: *mut OgaTensor) -> Self {
        Self {
            ptr: ptr,
            _marker: PhantomData,
        }
    }

    pub fn get_type(&self) -> Result<ElementType> {
        let ptr = self.ptr;
        let mut oga_type = ort_genai_sys::OgaElementType_OgaElementType_undefined;
        unsafe {
            check_status(OgaTensorGetType(ptr, &mut oga_type))?;
        }
        Ok(oga_type.into())
    }

    pub fn get_shape_rank(&self) -> Result<usize> {
        let ptr = self.ptr;
        let mut rank: usize = 0;
        unsafe {
            check_status(OgaTensorGetShapeRank(ptr, &mut rank))?;
        }
        Ok(rank)
    }

    pub fn get_shape(&self) -> Result<Vec<i64>> {
        let ptr = self.ptr;
        let rank = self.get_shape_rank()?;
        let mut shape = vec![0; rank];
        unsafe {
            check_status(OgaTensorGetShape(ptr, shape.as_mut_ptr(), rank))?;
        }
        Ok(shape)
    }

    /// Returns a mutable slice to the tensor's data.
    ///
    /// # Safety
    /// The caller must ensure that the generic type `T` accurately corresponds
    /// to the tensor's underlying `ElementType` (e.g., `T = f32` for `Float32`).
    /// Providing the wrong type will result in Undefined Behavior due to incorrect
    /// alignment and size calculations.
    pub unsafe fn get_data<T>(&mut self) -> Result<&mut [T]> {
        unsafe {
            let ptr = self.ptr;
            let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            check_status(OgaTensorGetData(ptr, &mut data_ptr))?;

            let shape = self.get_shape()?;
            let size: usize = shape.iter().map(|&x| x as usize).product();

            let slice = std::slice::from_raw_parts_mut(data_ptr as *mut T, size);
            Ok(slice)
        }
    }
}

impl<'a> Drop for Tensor<'a> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                OgaDestroyTensor(self.ptr);
            }
        }
    }
}

pub struct NamedTensors<'a> {
    pub(crate) ptr: *mut OgaNamedTensors,
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> NamedTensors<'a> {
    pub fn new() -> Result<Self> {
        let mut ptr: *mut OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaCreateNamedTensors(&mut ptr))?;
        }
        Ok(Self {
            ptr: ptr,
            _marker: PhantomData,
        })
    }

    pub(crate) fn from_ptr(ptr: *mut OgaNamedTensors) -> Self {
        Self {
            ptr: ptr,
            _marker: PhantomData,
        }
    }

    pub fn get(&self, name: &str) -> Result<Tensor<'a>> {
        let ptr = self.ptr;
        let c_name = CString::new(name)?;
        let mut tensor_ptr: *mut OgaTensor = std::ptr::null_mut();
        unsafe {
            check_status(OgaNamedTensorsGet(ptr, c_name.as_ptr(), &mut tensor_ptr))?;
        }
        Ok(Tensor::from_ptr(tensor_ptr))
    }

    pub fn set(&self, name: &str, tensor: &Tensor<'_>) -> Result<()> {
        let ptr = self.ptr;
        let c_name = CString::new(name)?;
        let tensor_ptr = tensor.ptr;
        unsafe {
            check_status(OgaNamedTensorsSet(ptr, c_name.as_ptr(), tensor_ptr))?;
        }
        Ok(())
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        let ptr = self.ptr;
        let c_name = CString::new(name)?;
        unsafe {
            check_status(OgaNamedTensorsDelete(ptr, c_name.as_ptr()))?;
        }
        Ok(())
    }

    pub fn len(&self) -> Result<usize> {
        let ptr = self.ptr;
        let mut count: usize = 0;
        unsafe {
            check_status(OgaNamedTensorsCount(ptr, &mut count))?;
        }
        Ok(count)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn get_names(&self) -> Result<StringArray> {
        let ptr = self.ptr;
        let mut names_ptr: *mut ort_genai_sys::OgaStringArray = std::ptr::null_mut();
        unsafe {
            check_status(OgaNamedTensorsGetNames(ptr, &mut names_ptr))?;
        }
        Ok(StringArray { ptr: names_ptr })
    }
}

impl<'a> Drop for NamedTensors<'a> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                OgaDestroyNamedTensors(self.ptr);
            }
        }
    }
}
