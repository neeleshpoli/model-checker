use ort_genai_sys::{
    OgaElementType, OgaElementType_OgaElementType_bfloat16, OgaElementType_OgaElementType_bool,
    OgaElementType_OgaElementType_complex64, OgaElementType_OgaElementType_complex128,
    OgaElementType_OgaElementType_float16, OgaElementType_OgaElementType_float32,
    OgaElementType_OgaElementType_float64, OgaElementType_OgaElementType_int8,
    OgaElementType_OgaElementType_int16, OgaElementType_OgaElementType_int32,
    OgaElementType_OgaElementType_int64, OgaElementType_OgaElementType_string,
    OgaElementType_OgaElementType_uint8, OgaElementType_OgaElementType_uint16,
    OgaElementType_OgaElementType_uint32, OgaElementType_OgaElementType_uint64,
    OgaElementType_OgaElementType_undefined,
};

/// Canonical element/data types used by ONNX Runtime GenAI tensor values.
///
/// These values describe the in-memory representation of a tensor’s elements and are
/// used when converting between native C types and Rust tensor APIs.
#[derive(Debug, PartialEq, Eq)]
pub enum ElementType {
    /// Unspecified or invalid element type.
    Undefined,
    /// 32-bit floating-point value.
    Float32,
    /// Unsigned 8-bit integer.
    UInt8,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 16-bit integer.
    UInt16,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// String element type.
    String,
    /// Boolean value.
    Bool,
    /// 16-bit floating-point value.
    Float16,
    /// 64-bit floating-point value.
    Float64,
    /// Unsigned 32-bit integer.
    UInt32,
    /// Unsigned 64-bit integer.
    UInt64,
    /// Complex value with 32-bit float real and imaginary parts.
    Complex64,
    /// Complex value with 64-bit float real and imaginary parts.
    Complex128,
    /// Brain-float 16-bit value.
    BFloat16,
}

/// Converts the native ONNX Runtime GenAI enum into the Rust-facing representation.
impl From<OgaElementType> for ElementType {
    fn from(value: OgaElementType) -> Self {
        #[allow(non_upper_case_globals)]
        match value {
            OgaElementType_OgaElementType_undefined => Self::Undefined,
            OgaElementType_OgaElementType_float32 => Self::Float32,
            OgaElementType_OgaElementType_uint8 => Self::UInt8,
            OgaElementType_OgaElementType_int8 => Self::Int8,
            OgaElementType_OgaElementType_uint16 => Self::UInt16,
            OgaElementType_OgaElementType_int16 => Self::Int16,
            OgaElementType_OgaElementType_int32 => Self::Int32,
            OgaElementType_OgaElementType_int64 => Self::Int64,
            OgaElementType_OgaElementType_string => Self::String,
            OgaElementType_OgaElementType_bool => Self::Bool,
            OgaElementType_OgaElementType_float16 => Self::Float16,
            OgaElementType_OgaElementType_float64 => Self::Float64,
            OgaElementType_OgaElementType_uint32 => Self::UInt32,
            OgaElementType_OgaElementType_uint64 => Self::UInt64,
            OgaElementType_OgaElementType_complex64 => Self::Complex64,
            OgaElementType_OgaElementType_complex128 => Self::Complex128,
            OgaElementType_OgaElementType_bfloat16 => Self::BFloat16,
            _ => unreachable!("Unknown element type"),
        }
    }
}

/// Converts the Rust element type back into the native GenAI enum representation.
impl From<ElementType> for OgaElementType {
    fn from(value: ElementType) -> Self {
        match value {
            ElementType::Undefined => OgaElementType_OgaElementType_undefined,
            ElementType::Float32 => OgaElementType_OgaElementType_float32,
            ElementType::UInt8 => OgaElementType_OgaElementType_uint8,
            ElementType::Int8 => OgaElementType_OgaElementType_int8,
            ElementType::UInt16 => OgaElementType_OgaElementType_uint16,
            ElementType::Int16 => OgaElementType_OgaElementType_int16,
            ElementType::Int32 => OgaElementType_OgaElementType_int32,
            ElementType::Int64 => OgaElementType_OgaElementType_int64,
            ElementType::String => OgaElementType_OgaElementType_string,
            ElementType::Bool => OgaElementType_OgaElementType_bool,
            ElementType::Float16 => OgaElementType_OgaElementType_float16,
            ElementType::Float64 => OgaElementType_OgaElementType_float64,
            ElementType::UInt32 => OgaElementType_OgaElementType_uint32,
            ElementType::UInt64 => OgaElementType_OgaElementType_uint64,
            ElementType::Complex64 => OgaElementType_OgaElementType_complex64,
            ElementType::Complex128 => OgaElementType_OgaElementType_complex128,
            ElementType::BFloat16 => OgaElementType_OgaElementType_bfloat16,
        }
    }
}

/// Marks a Rust scalar type as compatible with ONNX Runtime GenAI tensor element types.
///
/// This is used to map Rust numeric types to their native equivalent when creating and
/// validating tensor buffers. The associated `OGA_TYPE` is a pure metadata value and does not
/// carry ownership or lifetime information.
pub trait TensorElement {
    const OGA_TYPE: OgaElementType;
}

impl TensorElement for f32 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_float32;
}

impl TensorElement for f64 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_float64;
}

impl TensorElement for i8 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_int8;
}

impl TensorElement for i16 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_int16;
}

impl TensorElement for i32 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_int32;
}

impl TensorElement for i64 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_int64;
}

impl TensorElement for u8 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_uint8;
}

impl TensorElement for u16 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_uint16;
}

impl TensorElement for u32 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_uint32;
}

impl TensorElement for u64 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_uint64;
}

impl TensorElement for bool {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_bool;
}

impl TensorElement for half::f16 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_float16;
}

impl TensorElement for half::bf16 {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_bfloat16;
}

impl TensorElement for num_complex::Complex<f32> {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_complex64;
}

impl TensorElement for num_complex::Complex<f64> {
    const OGA_TYPE: OgaElementType = OgaElementType_OgaElementType_complex128;
}
