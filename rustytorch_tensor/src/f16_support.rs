//! F16 (Half precision) support for tensors
//!
//! This module provides preliminary support for 16-bit floating point operations.
//! F16 is crucial for modern deep learning to reduce memory usage and increase throughput.

use crate::{storage::StorageType, Tensor};
use half::f16;
use rustytorch_core::{CoreError, DType, Result, TensorOptions};

/// F16 operations trait
pub trait F16Ops {
    /// Convert tensor to F16
    fn to_f16(&self) -> Result<Tensor>;

    /// Check if tensor is F16
    fn is_f16(&self) -> bool;

    /// Get F16 data (if tensor is F16)
    fn f16_data(&self) -> Result<Vec<f16>>;
}

/// F16 arithmetic operations
pub struct F16Arithmetic;

impl F16Arithmetic {
    /// Add two F16 tensors
    pub fn add_f16(a: &[f16], b: &[f16]) -> Result<Vec<f16>> {
        if a.len() != b.len() {
            return Err(CoreError::invalid_op(
                "f16_add",
                "Tensors must have same size",
            ));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect())
    }

    /// Subtract two F16 tensors
    pub fn sub_f16(a: &[f16], b: &[f16]) -> Result<Vec<f16>> {
        if a.len() != b.len() {
            return Err(CoreError::invalid_op(
                "f16_sub",
                "Tensors must have same size",
            ));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect())
    }

    /// Multiply two F16 tensors element-wise
    pub fn mul_f16(a: &[f16], b: &[f16]) -> Result<Vec<f16>> {
        if a.len() != b.len() {
            return Err(CoreError::invalid_op(
                "f16_mul",
                "Tensors must have same size",
            ));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect())
    }

    /// Divide two F16 tensors element-wise
    pub fn div_f16(a: &[f16], b: &[f16]) -> Result<Vec<f16>> {
        if a.len() != b.len() {
            return Err(CoreError::invalid_op(
                "f16_div",
                "Tensors must have same size",
            ));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| *x / *y).collect())
    }

    /// Matrix multiplication for F16
    pub fn matmul_f16(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Result<Vec<f16>> {
        let mut result = vec![f16::from_f32(0.0); m * k];

        for i in 0..m {
            for j in 0..k {
                let mut sum = f16::from_f32(0.0);
                for l in 0..n {
                    sum += a[i * n + l] * b[l * k + j];
                }
                result[i * k + j] = sum;
            }
        }

        Ok(result)
    }

    /// Reduction operations
    pub fn sum_f16(data: &[f16]) -> f16 {
        data.iter().fold(f16::from_f32(0.0), |acc, &x| acc + x)
    }

    pub fn mean_f16(data: &[f16]) -> f16 {
        if data.is_empty() {
            return f16::from_f32(0.0);
        }
        let sum = Self::sum_f16(data);
        sum / f16::from_f32(data.len() as f32)
    }

    pub fn max_f16(data: &[f16]) -> Option<f16> {
        data.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    pub fn min_f16(data: &[f16]) -> Option<f16> {
        data.iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }
}

/// F16 conversions
pub struct F16Conversions;

impl F16Conversions {
    /// Convert F32 array to F16
    pub fn f32_to_f16(data: &[f32]) -> Vec<f16> {
        data.iter().map(|&x| f16::from_f32(x)).collect()
    }

    /// Convert F16 array to F32
    pub fn f16_to_f32(data: &[f16]) -> Vec<f32> {
        data.iter().map(|&x| x.to_f32()).collect()
    }

    /// Convert F64 array to F16
    pub fn f64_to_f16(data: &[f64]) -> Vec<f16> {
        data.iter().map(|&x| f16::from_f64(x)).collect()
    }

    /// Convert F16 array to F64
    pub fn f16_to_f64(data: &[f16]) -> Vec<f64> {
        data.iter().map(|&x| x.to_f64()).collect()
    }
}

/// F16 special values and utilities
pub struct F16Utils;

impl F16Utils {
    /// Get F16 epsilon
    pub fn epsilon() -> f16 {
        f16::EPSILON
    }

    /// Get F16 infinity
    pub fn infinity() -> f16 {
        f16::INFINITY
    }

    /// Get F16 negative infinity
    pub fn neg_infinity() -> f16 {
        f16::NEG_INFINITY
    }

    /// Get F16 NaN
    pub fn nan() -> f16 {
        f16::NAN
    }

    /// Check if F16 is finite
    pub fn is_finite(x: f16) -> bool {
        x.is_finite()
    }

    /// Check if F16 is infinite
    pub fn is_infinite(x: f16) -> bool {
        x.is_infinite()
    }

    /// Check if F16 is NaN
    pub fn is_nan(x: f16) -> bool {
        x.is_nan()
    }

    /// Clamp F16 value
    pub fn clamp(x: f16, min: f16, max: f16) -> f16 {
        if x < min {
            min
        } else if x > max {
            max
        } else {
            x
        }
    }
}

/// Mixed precision operations
pub struct MixedPrecisionOps;

impl MixedPrecisionOps {
    /// Perform operation in F32 and convert back to F16
    pub fn mixed_matmul(
        a_f16: &[f16],
        b_f16: &[f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f16>> {
        // Convert to F32
        let a_f32 = F16Conversions::f16_to_f32(a_f16);
        let b_f32 = F16Conversions::f16_to_f32(b_f16);

        // Perform computation in F32
        let mut result_f32 = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0f32;
                for l in 0..n {
                    sum += a_f32[i * n + l] * b_f32[l * k + j];
                }
                result_f32[i * k + j] = sum;
            }
        }

        // Convert back to F16
        Ok(F16Conversions::f32_to_f16(&result_f32))
    }

    /// Automatic mixed precision helper
    pub fn amp_operation<F>(input_f16: &[f16], op: F) -> Vec<f16>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        let input_f32 = F16Conversions::f16_to_f32(input_f16);
        let result_f32 = op(&input_f32);
        F16Conversions::f32_to_f16(&result_f32)
    }
}

/// Extension methods for Tensor to support F16
impl F16Ops for Tensor {
    fn to_f16(&self) -> Result<Tensor> {
        if self.dtype() == DType::Float16 {
            return Ok(self.clone());
        }

        // Convert to F16
        let f16_data = match self.storage() {
            StorageType::F32(data) => F16Conversions::f32_to_f16(data),
            StorageType::F64(data) => F16Conversions::f64_to_f16(data),
            _ => {
                // Convert to F64 first, then to F16
                let f64_data = self.storage().to_vec_f64();
                F16Conversions::f64_to_f16(&f64_data)
            }
        };

        // For now, store F16 as F32 internally (as defined in type_ops.rs)
        let f32_data = F16Conversions::f16_to_f32(&f16_data);
        let storage = StorageType::F32(f32_data);

        let mut options = self.options().clone();
        options.dtype = DType::Float16;

        Ok(Tensor {
            storage: std::sync::Arc::new(storage),
            shape: self.shape().to_vec(),
            strides: self.strides().to_vec(),
            offset: self.offset(),
            options,
        })
    }

    fn is_f16(&self) -> bool {
        self.dtype() == DType::Float16
    }

    fn f16_data(&self) -> Result<Vec<f16>> {
        if !self.is_f16() {
            return Err(CoreError::invalid_op("f16_data", "Tensor is not F16"));
        }

        match self.storage() {
            StorageType::F32(data) => Ok(F16Conversions::f32_to_f16(data)),
            _ => Err(CoreError::invalid_op("f16_data", "Invalid storage for F16")),
        }
    }
}

/// F16-specific tensor creation functions
impl Tensor {
    /// Create F16 tensor from data
    pub fn from_f16(data: &[f16], shape: Vec<usize>) -> Result<Self> {
        let total_size: usize = shape.iter().product();
        if data.len() != total_size {
            return Err(CoreError::invalid_op(
                "from_f16",
                "Data length doesn't match shape",
            ));
        }

        // Convert to F32 for storage (as per current implementation)
        let f32_data = F16Conversions::f16_to_f32(data);
        let storage = StorageType::F32(f32_data);

        let mut options = TensorOptions::default();
        options.dtype = DType::Float16;

        let strides = Self::compute_strides(&shape);

        Ok(Self {
            storage: std::sync::Arc::new(storage),
            shape,
            strides,
            offset: 0,
            options,
        })
    }

    /// Create F16 zeros tensor
    pub fn zeros_f16(shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        let data = vec![f16::from_f32(0.0); total_size];
        Self::from_f16(&data, shape).unwrap()
    }

    /// Create F16 ones tensor
    pub fn ones_f16(shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        let data = vec![f16::from_f32(1.0); total_size];
        Self::from_f16(&data, shape).unwrap()
    }

    /// Create F16 tensor filled with value
    pub fn full_f16(shape: Vec<usize>, value: f16) -> Self {
        let total_size: usize = shape.iter().product();
        let data = vec![value; total_size];
        Self::from_f16(&data, shape).unwrap()
    }

    // Helper to compute strides
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        if shape.len() > 1 {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversions() {
        let f32_data = vec![1.0f32, 2.5, -3.7, 0.0];
        let f16_data = F16Conversions::f32_to_f16(&f32_data);
        let f32_back = F16Conversions::f16_to_f32(&f16_data);

        for (orig, converted) in f32_data.iter().zip(f32_back.iter()) {
            assert!((orig - converted).abs() < 0.01);
        }
    }

    #[test]
    fn test_f16_arithmetic() {
        let a = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let b = vec![f16::from_f32(3.0), f16::from_f32(4.0)];

        let sum = F16Arithmetic::add_f16(&a, &b).unwrap();
        assert_eq!(sum[0].to_f32(), 4.0);
        assert_eq!(sum[1].to_f32(), 6.0);

        let diff = F16Arithmetic::sub_f16(&a, &b).unwrap();
        assert_eq!(diff[0].to_f32(), -2.0);
        assert_eq!(diff[1].to_f32(), -2.0);
    }

    #[test]
    fn test_f16_tensor_creation() {
        let tensor = Tensor::zeros_f16(vec![2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::Float16);

        let ones = Tensor::ones_f16(vec![4]);
        assert_eq!(ones.shape(), &[4]);
        assert_eq!(ones.dtype(), DType::Float16);
    }

    #[test]
    fn test_f16_matmul() {
        let a = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let b = vec![
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
        ];

        let result = F16Arithmetic::matmul_f16(&a, &b, 2, 2, 2).unwrap();

        // [1 2] * [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        assert_eq!(result[0].to_f32(), 19.0);
        assert_eq!(result[1].to_f32(), 22.0);
        assert_eq!(result[2].to_f32(), 43.0);
        assert_eq!(result[3].to_f32(), 50.0);
    }

    #[test]
    fn test_mixed_precision() {
        let a = vec![f16::from_f32(0.1); 100];
        let b = vec![f16::from_f32(0.1); 100];

        // Direct F16 computation might accumulate errors
        let direct = F16Arithmetic::matmul_f16(&a, &b, 10, 10, 10).unwrap();

        // Mixed precision should be more accurate
        let mixed = MixedPrecisionOps::mixed_matmul(&a, &b, 10, 10, 10).unwrap();

        // Both should give reasonable results
        assert!(direct[0].is_finite());
        assert!(mixed[0].is_finite());
    }
}
