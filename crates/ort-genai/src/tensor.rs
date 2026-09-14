use std::ffi::CString;
use std::sync::{Arc, Mutex};

use ort_genai_sys::{
    OgaCreateNamedTensors, OgaCreateTensorFromBuffer, OgaDestroyNamedTensors, OgaDestroyTensor,
    OgaElementType, OgaNamedTensors, OgaNamedTensorsCount, OgaNamedTensorsDelete,
    OgaNamedTensorsGet, OgaNamedTensorsGetNames, OgaNamedTensorsSet, OgaTensor, OgaTensorGetData,
    OgaTensorGetShape, OgaTensorGetShapeRank, OgaTensorGetType,
};

use crate::error::{Result, check_status};
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

pub struct Tensor {
    pub(crate) ptr: Arc<Mutex<*mut OgaTensor>>,
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    pub fn from_buffer(
        data: *mut std::ffi::c_void,
        shape: &[i64],
        element_type: ElementType,
    ) -> Result<Self> {
        let mut ptr: *mut OgaTensor = std::ptr::null_mut();
        let oga_type = element_type.into();
        unsafe {
            check_status(OgaCreateTensorFromBuffer(
                data,
                shape.as_ptr(),
                shape.len(),
                oga_type,
                &mut ptr,
            ))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub(crate) fn from_ptr(ptr: *mut OgaTensor) -> Self {
        Self {
            ptr: Arc::new(Mutex::new(ptr)),
        }
    }

    pub fn get_type(&self) -> Result<ElementType> {
        let ptr = self.ptr.lock()?;
        let mut oga_type = ort_genai_sys::OgaElementType_OgaElementType_undefined;
        unsafe {
            check_status(OgaTensorGetType(*ptr, &mut oga_type))?;
        }
        Ok(oga_type.into())
    }

    pub fn get_shape_rank(&self) -> Result<usize> {
        let ptr = self.ptr.lock()?;
        let mut rank: usize = 0;
        unsafe {
            check_status(OgaTensorGetShapeRank(*ptr, &mut rank))?;
        }
        Ok(rank)
    }

    pub fn get_shape(&self) -> Result<Vec<i64>> {
        let ptr = self.ptr.lock()?;
        let rank = self.get_shape_rank()?;
        let mut shape = vec![0; rank];
        unsafe {
            check_status(OgaTensorGetShape(*ptr, shape.as_mut_ptr(), rank))?;
        }
        Ok(shape)
    }

    pub fn get_data(&self) -> Result<*mut std::ffi::c_void> {
        let ptr = self.ptr.lock()?;
        let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            check_status(OgaTensorGetData(*ptr, &mut data_ptr))?;
        }
        Ok(data_ptr)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyTensor(*ptr);
                }
            }
        }
    }
}

pub struct NamedTensors {
    pub(crate) ptr: Arc<Mutex<*mut OgaNamedTensors>>,
}

unsafe impl Send for NamedTensors {}
unsafe impl Sync for NamedTensors {}

impl NamedTensors {
    pub fn new() -> Result<Self> {
        let mut ptr: *mut OgaNamedTensors = std::ptr::null_mut();
        unsafe {
            check_status(OgaCreateNamedTensors(&mut ptr))?;
        }
        Ok(Self {
            ptr: Arc::new(Mutex::new(ptr)),
        })
    }

    pub(crate) fn from_ptr(ptr: *mut OgaNamedTensors) -> Self {
        Self {
            ptr: Arc::new(Mutex::new(ptr)),
        }
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        let mut tensor_ptr: *mut OgaTensor = std::ptr::null_mut();
        unsafe {
            check_status(OgaNamedTensorsGet(*ptr, c_name.as_ptr(), &mut tensor_ptr))?;
        }
        Ok(Tensor::from_ptr(tensor_ptr))
    }

    pub fn set(&self, name: &str, tensor: &Tensor) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        let tensor_ptr = tensor.ptr.lock()?;
        unsafe {
            check_status(OgaNamedTensorsSet(*ptr, c_name.as_ptr(), *tensor_ptr))?;
        }
        Ok(())
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        let ptr = self.ptr.lock()?;
        let c_name = CString::new(name)?;
        unsafe {
            check_status(OgaNamedTensorsDelete(*ptr, c_name.as_ptr()))?;
        }
        Ok(())
    }

    pub fn len(&self) -> Result<usize> {
        let ptr = self.ptr.lock()?;
        let mut count: usize = 0;
        unsafe {
            check_status(OgaNamedTensorsCount(*ptr, &mut count))?;
        }
        Ok(count)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn get_names(&self) -> Result<StringArray> {
        let ptr = self.ptr.lock()?;
        let mut names_ptr: *mut ort_genai_sys::OgaStringArray = std::ptr::null_mut();
        unsafe {
            check_status(OgaNamedTensorsGetNames(*ptr, &mut names_ptr))?;
        }
        Ok(StringArray {
            ptr: Arc::new(Mutex::new(names_ptr)),
        })
    }
}

impl Drop for NamedTensors {
    fn drop(&mut self) {
        if let Ok(ptr) = self.ptr.lock() {
            if !ptr.is_null() {
                unsafe {
                    OgaDestroyNamedTensors(*ptr);
                }
            }
        }
    }
}
