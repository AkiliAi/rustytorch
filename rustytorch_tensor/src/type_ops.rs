//! Type operations and conversions for tensors
//!
//! This module implements:
//! - Type conversions between all supported dtypes
//! - Type-specific optimized operations
//! - Automatic type promotion rules
//! - Complex number support

use crate::{storage::StorageType, Tensor};
use num_complex::Complex;
use rustytorch_core::{CoreError, DType, Result};

/// Type conversion operations
pub struct TypeOps;

impl TypeOps {
    /// Convert tensor to specified dtype
    pub fn to_dtype(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
        if tensor.dtype() == dtype {
            return Ok(tensor.clone());
        }

        let data = tensor.storage().to_vec_f64();
        let new_storage = Self::convert_storage(&data, tensor.dtype(), dtype)?;

        let mut options = tensor.options().clone();
        options.dtype = dtype;

        Ok(Tensor {
            storage: std::sync::Arc::new(new_storage),
            shape: tensor.shape().to_vec(),
            strides: tensor.strides().to_vec(),
            offset: tensor.offset(),
            options,
        })
    }

    /// Convert storage from one type to another
    fn convert_storage(data: &[f64], from_dtype: DType, to_dtype: DType) -> Result<StorageType> {
        match to_dtype {
            DType::Float16 => {
                // F16 conversion requires half crate or manual implementation
                // For now, store as f32 internally
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                Ok(StorageType::F32(f32_data))
            }
            DType::Float32 => {
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                Ok(StorageType::F32(f32_data))
            }
            DType::Float64 => Ok(StorageType::F64(data.to_vec())),
            DType::Int8 => {
                let i8_data: Vec<i8> = data
                    .iter()
                    .map(|&x| Self::safe_cast_to_i8(x))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StorageType::I8(i8_data))
            }
            DType::Int16 => {
                let i16_data: Vec<i16> = data
                    .iter()
                    .map(|&x| Self::safe_cast_to_i16(x))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StorageType::I16(i16_data))
            }
            DType::Int32 => {
                let i32_data: Vec<i32> = data
                    .iter()
                    .map(|&x| Self::safe_cast_to_i32(x))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StorageType::I32(i32_data))
            }
            DType::Int64 => {
                let i64_data: Vec<i64> = data
                    .iter()
                    .map(|&x| Self::safe_cast_to_i64(x))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StorageType::I64(i64_data))
            }
            DType::UInt8 => {
                let u8_data: Vec<u8> = data
                    .iter()
                    .map(|&x| Self::safe_cast_to_u8(x))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StorageType::U8(u8_data))
            }
            DType::UInt16 => {
                let u16_data: Vec<u16> = data
                    .iter()
                    .map(|&x| Self::safe_cast_to_u16(x))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StorageType::U16(u16_data))
            }
            DType::UInt32 => {
                let u32_data: Vec<u32> = data
                    .iter()
                    .map(|&x| Self::safe_cast_to_u32(x))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StorageType::U32(u32_data))
            }
            DType::UInt64 => {
                let u64_data: Vec<u64> = data
                    .iter()
                    .map(|&x| Self::safe_cast_to_u64(x))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StorageType::U64(u64_data))
            }
            DType::Bool => {
                let bool_data: Vec<bool> = data.iter().map(|&x| x != 0.0).collect();
                Ok(StorageType::Bool(bool_data))
            }
            DType::Complex64 => {
                // For real to complex conversion, imaginary part is 0
                let complex_data: Vec<Complex<f32>> =
                    data.iter().map(|&x| Complex::new(x as f32, 0.0)).collect();
                Ok(StorageType::Complex64(complex_data))
            }
            DType::Complex128 => {
                let complex_data: Vec<Complex<f64>> =
                    data.iter().map(|&x| Complex::new(x, 0.0)).collect();
                Ok(StorageType::Complex128(complex_data))
            }
        }
    }

    /// Safe cast to i8 with overflow checking
    fn safe_cast_to_i8(x: f64) -> Result<i8> {
        if x >= i8::MIN as f64 && x <= i8::MAX as f64 {
            Ok(x as i8)
        } else {
            Err(CoreError::invalid_op(
                "cast_to_i8",
                &format!("Value {} out of range for i8", x),
            ))
        }
    }

    /// Safe cast to i16 with overflow checking
    fn safe_cast_to_i16(x: f64) -> Result<i16> {
        if x >= i16::MIN as f64 && x <= i16::MAX as f64 {
            Ok(x as i16)
        } else {
            Err(CoreError::invalid_op(
                "cast_to_i16",
                &format!("Value {} out of range for i16", x),
            ))
        }
    }

    /// Safe cast to i32 with overflow checking
    fn safe_cast_to_i32(x: f64) -> Result<i32> {
        if x >= i32::MIN as f64 && x <= i32::MAX as f64 {
            Ok(x as i32)
        } else {
            Err(CoreError::invalid_op(
                "cast_to_i32",
                &format!("Value {} out of range for i32", x),
            ))
        }
    }

    /// Safe cast to i64 with overflow checking
    fn safe_cast_to_i64(x: f64) -> Result<i64> {
        if x >= i64::MIN as f64 && x <= i64::MAX as f64 {
            Ok(x as i64)
        } else {
            Err(CoreError::invalid_op(
                "cast_to_i64",
                &format!("Value {} out of range for i64", x),
            ))
        }
    }

    /// Safe cast to u8 with overflow checking
    fn safe_cast_to_u8(x: f64) -> Result<u8> {
        if x >= 0.0 && x <= u8::MAX as f64 {
            Ok(x as u8)
        } else {
            Err(CoreError::invalid_op(
                "cast_to_u8",
                &format!("Value {} out of range for u8", x),
            ))
        }
    }

    /// Safe cast to u16 with overflow checking
    fn safe_cast_to_u16(x: f64) -> Result<u16> {
        if x >= 0.0 && x <= u16::MAX as f64 {
            Ok(x as u16)
        } else {
            Err(CoreError::invalid_op(
                "cast_to_u16",
                &format!("Value {} out of range for u16", x),
            ))
        }
    }

    /// Safe cast to u32 with overflow checking
    fn safe_cast_to_u32(x: f64) -> Result<u32> {
        if x >= 0.0 && x <= u32::MAX as f64 {
            Ok(x as u32)
        } else {
            Err(CoreError::invalid_op(
                "cast_to_u32",
                &format!("Value {} out of range for u32", x),
            ))
        }
    }

    /// Safe cast to u64 with overflow checking
    fn safe_cast_to_u64(x: f64) -> Result<u64> {
        if x >= 0.0 && x <= u64::MAX as f64 {
            Ok(x as u64)
        } else {
            Err(CoreError::invalid_op(
                "cast_to_u64",
                &format!("Value {} out of range for u64", x),
            ))
        }
    }

    /// Get the promoted dtype for binary operations
    pub fn promote_types(dtype1: DType, dtype2: DType) -> DType {
        // If types are the same, no promotion needed
        if dtype1 == dtype2 {
            return dtype1;
        }

        // Complex types always win
        if matches!(dtype1, DType::Complex128) || matches!(dtype2, DType::Complex128) {
            return DType::Complex128;
        }
        if matches!(dtype1, DType::Complex64) || matches!(dtype2, DType::Complex64) {
            return DType::Complex64;
        }

        // Float types promotion
        match (dtype1, dtype2) {
            (DType::Float64, _) | (_, DType::Float64) => DType::Float64,
            (DType::Float32, _) | (_, DType::Float32) => DType::Float32,
            (DType::Float16, _) | (_, DType::Float16) => DType::Float16,

            // Integer type promotion
            (DType::Int64, _) | (_, DType::Int64) => DType::Int64,
            (DType::UInt64, _) | (_, DType::UInt64) => DType::UInt64,
            (DType::Int32, _) | (_, DType::Int32) => DType::Int32,
            (DType::UInt32, _) | (_, DType::UInt32) => DType::UInt32,
            (DType::Int16, _) | (_, DType::Int16) => DType::Int16,
            (DType::UInt16, _) | (_, DType::UInt16) => DType::UInt16,
            (DType::Int8, _) | (_, DType::Int8) => DType::Int8,
            (DType::UInt8, _) | (_, DType::UInt8) => DType::UInt8,

            // Bool is lowest priority
            (DType::Bool, other) | (other, DType::Bool) => other,

            // Complex types with themselves
            (DType::Complex64, DType::Complex64) => DType::Complex64,
            (DType::Complex128, DType::Complex128) => DType::Complex128,

            // Any other combination defaults to the first type
            // This is a simplified promotion rule
            (first, _) => first,
        }
    }

    /// Check if dtype is floating point
    pub fn is_floating_point(dtype: DType) -> bool {
        matches!(
            dtype,
            DType::Float16 | DType::Float32 | DType::Float64 | DType::Complex64 | DType::Complex128
        )
    }

    /// Check if dtype is integral
    pub fn is_integral(dtype: DType) -> bool {
        matches!(
            dtype,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::UInt8
                | DType::UInt16
                | DType::UInt32
                | DType::UInt64
        )
    }

    /// Check if dtype is complex
    pub fn is_complex(dtype: DType) -> bool {
        matches!(dtype, DType::Complex64 | DType::Complex128)
    }

    /// Get size in bytes for dtype
    pub fn dtype_size(dtype: DType) -> usize {
        match dtype {
            DType::Float16 => 2,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int8 | DType::UInt8 => 1,
            DType::Int16 | DType::UInt16 => 2,
            DType::Int32 | DType::UInt32 => 4,
            DType::Int64 | DType::UInt64 => 8,
            DType::Bool => 1,
            DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }
}

/// Type-specific optimized operations
pub struct TypeSpecificOps;

impl TypeSpecificOps {
    /// Optimized integer operations
    pub fn int_add_i32(a: &[i32], b: &[i32], result: &mut [i32]) {
        for i in 0..a.len() {
            result[i] = a[i].wrapping_add(b[i]);
        }
    }

    /// Optimized unsigned operations
    pub fn uint_add_u32(a: &[u32], b: &[u32], result: &mut [u32]) {
        for i in 0..a.len() {
            result[i] = a[i].wrapping_add(b[i]);
        }
    }

    /// Optimized boolean operations
    pub fn bool_and(a: &[bool], b: &[bool], result: &mut [bool]) {
        for i in 0..a.len() {
            result[i] = a[i] && b[i];
        }
    }

    pub fn bool_or(a: &[bool], b: &[bool], result: &mut [bool]) {
        for i in 0..a.len() {
            result[i] = a[i] || b[i];
        }
    }

    pub fn bool_xor(a: &[bool], b: &[bool], result: &mut [bool]) {
        for i in 0..a.len() {
            result[i] = a[i] ^ b[i];
        }
    }

    /// Complex number operations
    pub fn complex_add_f32(a: &[Complex<f32>], b: &[Complex<f32>], result: &mut [Complex<f32>]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    pub fn complex_mul_f32(a: &[Complex<f32>], b: &[Complex<f32>], result: &mut [Complex<f32>]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    pub fn complex_conj_f32(a: &[Complex<f32>], result: &mut [Complex<f32>]) {
        for i in 0..a.len() {
            result[i] = a[i].conj();
        }
    }
}

/// Extension methods for Tensor to support type operations
impl Tensor {
    /// Convert tensor to specified dtype
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        TypeOps::to_dtype(self, dtype)
    }

    /// Cast to float32
    pub fn to_f32(&self) -> Result<Self> {
        self.to_dtype(DType::Float32)
    }

    /// Cast to float64
    pub fn to_f64(&self) -> Result<Self> {
        self.to_dtype(DType::Float64)
    }

    /// Cast to int32
    pub fn to_i32(&self) -> Result<Self> {
        self.to_dtype(DType::Int32)
    }

    /// Cast to int64
    pub fn to_i64(&self) -> Result<Self> {
        self.to_dtype(DType::Int64)
    }

    /// Cast to bool
    pub fn to_bool(&self) -> Result<Self> {
        self.to_dtype(DType::Bool)
    }

    /// Check if tensor is floating point
    pub fn is_floating_point(&self) -> bool {
        TypeOps::is_floating_point(self.dtype())
    }

    /// Check if tensor is integral
    pub fn is_integral(&self) -> bool {
        TypeOps::is_integral(self.dtype())
    }

    /// Check if tensor is complex
    pub fn is_complex(&self) -> bool {
        TypeOps::is_complex(self.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_conversion() {
        let tensor = Tensor::from_data(&[1.5f32, 2.7, 3.9], vec![3], None);

        // Convert to int32
        let int_tensor = tensor.to_i32().unwrap();
        assert_eq!(int_tensor.dtype(), DType::Int32);

        // Convert to float64
        let f64_tensor = tensor.to_f64().unwrap();
        assert_eq!(f64_tensor.dtype(), DType::Float64);

        // Convert to bool
        let bool_tensor = tensor.to_bool().unwrap();
        assert_eq!(bool_tensor.dtype(), DType::Bool);
    }

    #[test]
    fn test_type_promotion() {
        assert_eq!(
            TypeOps::promote_types(DType::Int32, DType::Float32),
            DType::Float32
        );
        assert_eq!(
            TypeOps::promote_types(DType::Float32, DType::Float64),
            DType::Float64
        );
        assert_eq!(
            TypeOps::promote_types(DType::Int32, DType::Int64),
            DType::Int64
        );
        assert_eq!(
            TypeOps::promote_types(DType::Bool, DType::Int32),
            DType::Int32
        );
        assert_eq!(
            TypeOps::promote_types(DType::Float32, DType::Complex64),
            DType::Complex64
        );
    }

    #[test]
    fn test_dtype_properties() {
        assert!(TypeOps::is_floating_point(DType::Float32));
        assert!(TypeOps::is_floating_point(DType::Complex64));
        assert!(!TypeOps::is_floating_point(DType::Int32));

        assert!(TypeOps::is_integral(DType::Int32));
        assert!(TypeOps::is_integral(DType::UInt8));
        assert!(!TypeOps::is_integral(DType::Float32));

        assert!(TypeOps::is_complex(DType::Complex64));
        assert!(!TypeOps::is_complex(DType::Float32));
    }

    #[test]
    fn test_safe_casting() {
        // Test overflow detection
        assert!(TypeOps::safe_cast_to_u8(256.0).is_err());
        assert!(TypeOps::safe_cast_to_u8(-1.0).is_err());
        assert!(TypeOps::safe_cast_to_u8(100.0).is_ok());

        assert!(TypeOps::safe_cast_to_i8(128.0).is_err());
        assert!(TypeOps::safe_cast_to_i8(-129.0).is_err());
        assert!(TypeOps::safe_cast_to_i8(100.0).is_ok());
    }
}
