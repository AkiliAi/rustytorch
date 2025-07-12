//! Padding and cropping operations for tensors
//!
//! This module implements various padding and cropping operations commonly used in
//! computer vision and deep learning applications.

use crate::{storage::StorageType, Tensor};
use rustytorch_core::{CoreError, Result};

/// Types of padding available
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    /// Fill with constant value (typically 0)
    Constant,
    /// Reflect values at the borders
    Reflect,
    /// Replicate border values
    Replicate,
    /// Circular/wrap-around padding
    Circular,
}

/// Padding specification for each dimension
#[derive(Debug, Clone)]
pub struct PaddingSpec {
    /// (pad_before, pad_after) for each dimension
    pub padding: Vec<(usize, usize)>,
    /// Padding mode
    pub mode: PaddingMode,
    /// Value to use for constant padding
    pub value: f64,
}

impl PaddingSpec {
    /// Create new padding specification
    pub fn new(padding: Vec<(usize, usize)>, mode: PaddingMode, value: f64) -> Self {
        Self {
            padding,
            mode,
            value,
        }
    }

    /// Create constant padding with zero value
    pub fn zeros(padding: Vec<(usize, usize)>) -> Self {
        Self::new(padding, PaddingMode::Constant, 0.0)
    }

    /// Create constant padding with custom value
    pub fn constant(padding: Vec<(usize, usize)>, value: f64) -> Self {
        Self::new(padding, PaddingMode::Constant, value)
    }

    /// Create reflection padding
    pub fn reflect(padding: Vec<(usize, usize)>) -> Self {
        Self::new(padding, PaddingMode::Reflect, 0.0)
    }

    /// Create replication padding
    pub fn replicate(padding: Vec<(usize, usize)>) -> Self {
        Self::new(padding, PaddingMode::Replicate, 0.0)
    }
}

/// Padding and cropping operations
pub struct PaddingOps;

impl PaddingOps {
    /// Apply padding to tensor according to specification
    pub fn pad(tensor: &Tensor, spec: &PaddingSpec) -> Result<Tensor> {
        if spec.padding.len() != tensor.ndim() {
            return Err(CoreError::invalid_op(
                "pad",
                &format!(
                    "Padding dimensions {} != tensor dimensions {}",
                    spec.padding.len(),
                    tensor.ndim()
                ),
            ));
        }

        match spec.mode {
            PaddingMode::Constant => Self::pad_constant(tensor, &spec.padding, spec.value),
            PaddingMode::Reflect => Self::pad_reflect(tensor, &spec.padding),
            PaddingMode::Replicate => Self::pad_replicate(tensor, &spec.padding),
            PaddingMode::Circular => Self::pad_circular(tensor, &spec.padding),
        }
    }

    /// Constant padding (fill with specified value)
    fn pad_constant(tensor: &Tensor, padding: &[(usize, usize)], value: f64) -> Result<Tensor> {
        let old_shape = tensor.shape();
        let mut new_shape = Vec::new();

        // Calculate new shape
        for (i, &(pad_before, pad_after)) in padding.iter().enumerate() {
            new_shape.push(old_shape[i] + pad_before + pad_after);
        }

        match tensor.storage() {
            StorageType::F32(_) => {
                let padded_data = Self::pad_constant_f32(tensor, padding, value as f32)?;
                Ok(Tensor::from_data(
                    &padded_data,
                    new_shape,
                    Some(tensor.options().clone()),
                ))
            }
            StorageType::F64(_) => {
                let padded_data = Self::pad_constant_f64(tensor, padding, value)?;
                Ok(Tensor::from_data(
                    &padded_data,
                    new_shape,
                    Some(tensor.options().clone()),
                ))
            }
            _ => Err(CoreError::invalid_op(
                "pad_constant",
                "Unsupported data type",
            )),
        }
    }

    /// Reflection padding (mirror values at borders)
    fn pad_reflect(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Tensor> {
        // Validate reflection padding constraints
        for (i, &(pad_before, pad_after)) in padding.iter().enumerate() {
            let dim_size = tensor.shape()[i];
            if pad_before >= dim_size || pad_after >= dim_size {
                return Err(CoreError::invalid_op(
                    "pad_reflect",
                    &format!(
                        "Padding size {} exceeds dimension size {} for reflection",
                        pad_before.max(pad_after),
                        dim_size
                    ),
                ));
            }
        }

        let old_shape = tensor.shape();
        let mut new_shape = Vec::new();

        for (i, &(pad_before, pad_after)) in padding.iter().enumerate() {
            new_shape.push(old_shape[i] + pad_before + pad_after);
        }

        match tensor.storage() {
            StorageType::F32(_) => {
                let padded_data = Self::pad_reflect_f32(tensor, padding)?;
                Ok(Tensor::from_data(
                    &padded_data,
                    new_shape,
                    Some(tensor.options().clone()),
                ))
            }
            StorageType::F64(_) => {
                let padded_data = Self::pad_reflect_f64(tensor, padding)?;
                Ok(Tensor::from_data(
                    &padded_data,
                    new_shape,
                    Some(tensor.options().clone()),
                ))
            }
            _ => Err(CoreError::invalid_op(
                "pad_reflect",
                "Unsupported data type",
            )),
        }
    }

    /// Replication padding (extend border values)
    fn pad_replicate(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Tensor> {
        let old_shape = tensor.shape();
        let mut new_shape = Vec::new();

        for (i, &(pad_before, pad_after)) in padding.iter().enumerate() {
            new_shape.push(old_shape[i] + pad_before + pad_after);
        }

        match tensor.storage() {
            StorageType::F32(_) => {
                let padded_data = Self::pad_replicate_f32(tensor, padding)?;
                Ok(Tensor::from_data(
                    &padded_data,
                    new_shape,
                    Some(tensor.options().clone()),
                ))
            }
            StorageType::F64(_) => {
                let padded_data = Self::pad_replicate_f64(tensor, padding)?;
                Ok(Tensor::from_data(
                    &padded_data,
                    new_shape,
                    Some(tensor.options().clone()),
                ))
            }
            _ => Err(CoreError::invalid_op(
                "pad_replicate",
                "Unsupported data type",
            )),
        }
    }

    /// Circular padding (wrap around)
    fn pad_circular(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Tensor> {
        let old_shape = tensor.shape();
        let mut new_shape = Vec::new();

        for (i, &(pad_before, pad_after)) in padding.iter().enumerate() {
            new_shape.push(old_shape[i] + pad_before + pad_after);
        }

        match tensor.storage() {
            StorageType::F32(_) => {
                let padded_data = Self::pad_circular_f32(tensor, padding)?;
                Ok(Tensor::from_data(
                    &padded_data,
                    new_shape,
                    Some(tensor.options().clone()),
                ))
            }
            StorageType::F64(_) => {
                let padded_data = Self::pad_circular_f64(tensor, padding)?;
                Ok(Tensor::from_data(
                    &padded_data,
                    new_shape,
                    Some(tensor.options().clone()),
                ))
            }
            _ => Err(CoreError::invalid_op(
                "pad_circular",
                "Unsupported data type",
            )),
        }
    }

    /// Crop tensor to specified region
    pub fn crop(tensor: &Tensor, start: &[usize], end: &[usize]) -> Result<Tensor> {
        if start.len() != tensor.ndim() || end.len() != tensor.ndim() {
            return Err(CoreError::invalid_op(
                "crop",
                "Start and end coordinates must match tensor dimensions",
            ));
        }

        // Validate crop coordinates
        for (i, (&s, &e)) in start.iter().zip(end.iter()).enumerate() {
            if s >= e {
                return Err(CoreError::invalid_op(
                    "crop",
                    &format!(
                        "Invalid crop range: start {} >= end {} for dimension {}",
                        s, e, i
                    ),
                ));
            }
            if e > tensor.shape()[i] {
                return Err(CoreError::invalid_op(
                    "crop",
                    &format!(
                        "Crop end {} exceeds dimension size {} for dimension {}",
                        e,
                        tensor.shape()[i],
                        i
                    ),
                ));
            }
        }

        // Convert to ranges and use existing slice functionality
        let ranges: Vec<std::ops::Range<usize>> =
            start.iter().zip(end.iter()).map(|(&s, &e)| s..e).collect();

        tensor.slice_ranges(&ranges)
    }

    /// Center crop to specified size
    pub fn center_crop(tensor: &Tensor, target_size: &[usize]) -> Result<Tensor> {
        if target_size.len() != tensor.ndim() {
            return Err(CoreError::invalid_op(
                "center_crop",
                "Target size must match tensor dimensions",
            ));
        }

        let shape = tensor.shape();
        let mut start = Vec::new();
        let mut end = Vec::new();

        for (i, (&current_size, &target)) in shape.iter().zip(target_size.iter()).enumerate() {
            if target > current_size {
                return Err(CoreError::invalid_op(
                    "center_crop",
                    &format!(
                        "Target size {} > current size {} for dimension {}",
                        target, current_size, i
                    ),
                ));
            }

            let margin = current_size - target;
            let start_pos = margin / 2;
            start.push(start_pos);
            end.push(start_pos + target);
        }

        Self::crop(tensor, &start, &end)
    }

    // Helper functions for different data types

    fn pad_constant_f32(
        tensor: &Tensor,
        padding: &[(usize, usize)],
        value: f32,
    ) -> Result<Vec<f32>> {
        let old_shape = tensor.shape();
        let mut new_shape = Vec::new();

        for (i, &(pad_before, pad_after)) in padding.iter().enumerate() {
            new_shape.push(old_shape[i] + pad_before + pad_after);
        }

        let total_size: usize = new_shape.iter().product();
        let mut result = vec![value; total_size];

        // Copy original data to the correct position
        let old_data = tensor.storage().to_vec_f64();
        let old_data_f32: Vec<f32> = old_data.iter().map(|&x| x as f32).collect();

        Self::copy_to_padded_f32(&old_data_f32, &mut result, old_shape, &new_shape, padding)?;

        Ok(result)
    }

    fn pad_constant_f64(
        tensor: &Tensor,
        padding: &[(usize, usize)],
        value: f64,
    ) -> Result<Vec<f64>> {
        let old_shape = tensor.shape();
        let mut new_shape = Vec::new();

        for (i, &(pad_before, pad_after)) in padding.iter().enumerate() {
            new_shape.push(old_shape[i] + pad_before + pad_after);
        }

        let total_size: usize = new_shape.iter().product();
        let mut result = vec![value; total_size];

        let old_data = tensor.storage().to_vec_f64();
        Self::copy_to_padded_f64(&old_data, &mut result, old_shape, &new_shape, padding)?;

        Ok(result)
    }

    fn copy_to_padded_f32(
        src: &[f32],
        dst: &mut [f32],
        old_shape: &[usize],
        new_shape: &[usize],
        padding: &[(usize, usize)],
    ) -> Result<()> {
        // For now, implement a simple version for 1D and 2D tensors
        match old_shape.len() {
            1 => {
                let pad_before = padding[0].0;
                let old_size = old_shape[0];
                for i in 0..old_size {
                    dst[pad_before + i] = src[i];
                }
            }
            2 => {
                let (row_pad_before, _) = padding[0];
                let (col_pad_before, _) = padding[1];
                let old_rows = old_shape[0];
                let old_cols = old_shape[1];
                let new_cols = new_shape[1];

                for r in 0..old_rows {
                    for c in 0..old_cols {
                        let old_idx = r * old_cols + c;
                        let new_idx = (r + row_pad_before) * new_cols + (c + col_pad_before);
                        dst[new_idx] = src[old_idx];
                    }
                }
            }
            _ => {
                return Err(CoreError::invalid_op(
                    "copy_to_padded",
                    "Only 1D and 2D tensors supported for now",
                ));
            }
        }
        Ok(())
    }

    fn copy_to_padded_f64(
        src: &[f64],
        dst: &mut [f64],
        old_shape: &[usize],
        new_shape: &[usize],
        padding: &[(usize, usize)],
    ) -> Result<()> {
        // Similar to f32 version
        match old_shape.len() {
            1 => {
                let pad_before = padding[0].0;
                let old_size = old_shape[0];
                for i in 0..old_size {
                    dst[pad_before + i] = src[i];
                }
            }
            2 => {
                let (row_pad_before, _) = padding[0];
                let (col_pad_before, _) = padding[1];
                let old_rows = old_shape[0];
                let old_cols = old_shape[1];
                let new_cols = new_shape[1];

                for r in 0..old_rows {
                    for c in 0..old_cols {
                        let old_idx = r * old_cols + c;
                        let new_idx = (r + row_pad_before) * new_cols + (c + col_pad_before);
                        dst[new_idx] = src[old_idx];
                    }
                }
            }
            _ => {
                return Err(CoreError::invalid_op(
                    "copy_to_padded",
                    "Only 1D and 2D tensors supported for now",
                ));
            }
        }
        Ok(())
    }

    // Placeholder implementations for other padding modes
    fn pad_reflect_f32(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Vec<f32>> {
        // For now, return error - will implement reflection logic later
        Err(CoreError::invalid_op(
            "pad_reflect_f32",
            "Not yet implemented",
        ))
    }

    fn pad_reflect_f64(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Vec<f64>> {
        Err(CoreError::invalid_op(
            "pad_reflect_f64",
            "Not yet implemented",
        ))
    }

    fn pad_replicate_f32(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Vec<f32>> {
        Err(CoreError::invalid_op(
            "pad_replicate_f32",
            "Not yet implemented",
        ))
    }

    fn pad_replicate_f64(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Vec<f64>> {
        Err(CoreError::invalid_op(
            "pad_replicate_f64",
            "Not yet implemented",
        ))
    }

    fn pad_circular_f32(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Vec<f32>> {
        Err(CoreError::invalid_op(
            "pad_circular_f32",
            "Not yet implemented",
        ))
    }

    fn pad_circular_f64(tensor: &Tensor, padding: &[(usize, usize)]) -> Result<Vec<f64>> {
        Err(CoreError::invalid_op(
            "pad_circular_f64",
            "Not yet implemented",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_padding_1d() {
        let tensor = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], None);
        let spec = PaddingSpec::zeros(vec![(1, 2)]); // Pad 1 before, 2 after

        let result = PaddingOps::pad(&tensor, &spec).unwrap();
        assert_eq!(result.shape(), &[6]); // 3 + 1 + 2 = 6

        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_constant_padding_2d() {
        let tensor = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None);
        let spec = PaddingSpec::zeros(vec![(1, 1), (1, 1)]); // Pad 1 on all sides

        let result = PaddingOps::pad(&tensor, &spec).unwrap();
        assert_eq!(result.shape(), &[4, 4]); // (2+1+1, 2+1+1)

        let data = result.storage().to_vec_f64();
        // Expected: [0,0,0,0, 0,1,2,0, 0,3,4,0, 0,0,0,0]
        assert_eq!(data[0], 0.0); // Top-left corner
        assert_eq!(data[5], 1.0); // Original data at (1,1)
        assert_eq!(data[6], 2.0); // Original data at (1,2)
    }

    #[test]
    fn test_constant_padding_with_value() {
        let tensor = Tensor::from_data(&[1.0f32, 2.0], vec![2], None);
        let spec = PaddingSpec::constant(vec![(1, 1)], 5.0);

        let result = PaddingOps::pad(&tensor, &spec).unwrap();
        assert_eq!(result.shape(), &[4]);

        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![5.0, 1.0, 2.0, 5.0]);
    }

    #[test]
    fn test_crop_basic() {
        let tensor = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], None);

        let result = PaddingOps::crop(&tensor, &[1], &[4]).unwrap();
        assert_eq!(result.shape(), &[3]);

        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_center_crop() {
        let tensor = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], None);

        let result = PaddingOps::center_crop(&tensor, &[3]).unwrap();
        assert_eq!(result.shape(), &[3]);

        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![2.0, 3.0, 4.0]); // Center 3 elements
    }

    #[test]
    fn test_center_crop_2d() {
        let tensor = Tensor::from_data(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
            None,
        );

        let result = PaddingOps::center_crop(&tensor, &[2, 2]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // Should extract the center 2x2 region
        let data = result.storage().to_vec_f64();

        // From 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]], center 2x2 should be [[1,2],[4,5]]
        // because center crop with margin 1/2 = 0 starts at (0,0)
        assert_eq!(data, vec![1.0, 2.0, 4.0, 5.0]);
    }
}
