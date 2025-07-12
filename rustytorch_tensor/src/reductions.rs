//! Advanced reduction operations along specific axes
//!
//! This module implements efficient reductions with support for:
//! - Multiple axes reduction
//! - keepdim parameter
//! - Statistical operations (std, var)
//! - Optimized implementations using SIMD ops

use crate::{
    simd_ops::{F32Ops, F64Ops},
    Tensor,
};
use rustytorch_core::{CoreError, Result, TensorOptions};

/// Advanced reduction operations
pub struct AxisReductions;

impl AxisReductions {
    /// Sum along specified axes with keepdim option
    pub fn sum_dim(tensor: &Tensor, axes: &[usize], keep_dim: bool) -> Result<Tensor> {
        Self::validate_axes(tensor, axes)?;

        if axes.is_empty() {
            // No axes specified, sum all elements
            return Self::sum_all(tensor);
        }

        // Single axis optimization
        if axes.len() == 1 {
            return Self::sum_single_axis(tensor, axes[0], keep_dim);
        }

        // Multiple axes - reduce one by one
        let mut result = tensor.clone();
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_by(|a, b| b.cmp(a)); // Sort in descending order

        for &axis in &sorted_axes {
            let adjusted_axis = if keep_dim {
                axis
            } else {
                // Adjust axis index as dimensions are being reduced
                let count = sorted_axes.iter().filter(|&&a| a > axis).count();
                if axis >= count {
                    axis - count
                } else {
                    0 // Safeguard against underflow
                }
            };
            result = Self::sum_single_axis(&result, adjusted_axis, keep_dim)?;
        }

        Ok(result)
    }

    /// Mean along specified axes
    pub fn mean_dim(tensor: &Tensor, axes: &[usize], keep_dim: bool) -> Result<Tensor> {
        let sum_result = Self::sum_dim(tensor, axes, keep_dim)?;

        // Calculate the number of elements being averaged
        let mut reduced_elements = 1usize;
        for &axis in axes {
            reduced_elements *= tensor.shape()[axis];
        }

        // Divide by number of elements
        Self::divide_scalar(&sum_result, reduced_elements as f64)
    }

    /// Standard deviation along axes
    pub fn std_dim(
        tensor: &Tensor,
        axes: &[usize],
        unbiased: bool,
        keep_dim: bool,
    ) -> Result<Tensor> {
        let variance = Self::var_dim(tensor, axes, unbiased, keep_dim)?;
        Self::sqrt(&variance)
    }

    /// Variance along axes
    pub fn var_dim(
        tensor: &Tensor,
        axes: &[usize],
        unbiased: bool,
        keep_dim: bool,
    ) -> Result<Tensor> {
        // Calculate mean
        let mean = Self::mean_dim(tensor, axes, true)?; // Always keep dims for broadcasting

        // Calculate squared differences
        let squared_diff = Self::squared_diff(tensor, &mean)?;

        // Sum the squared differences
        let sum_sq_diff = Self::sum_dim(&squared_diff, axes, keep_dim)?;

        // Calculate divisor
        let mut n = 1usize;
        for &axis in axes {
            n *= tensor.shape()[axis];
        }

        let divisor = if unbiased && n > 1 {
            (n - 1) as f64
        } else {
            n as f64
        };

        Self::divide_scalar(&sum_sq_diff, divisor)
    }

    /// Min/Max along axes with indices
    pub fn min_dim(tensor: &Tensor, axis: usize, keep_dim: bool) -> Result<(Tensor, Tensor)> {
        Self::validate_axes(tensor, &[axis])?;

        let axis_size = tensor.shape()[axis];
        let mut result_shape = tensor.shape().to_vec();

        if keep_dim {
            result_shape[axis] = 1;
        } else {
            result_shape.remove(axis);
        }

        // Calculate output size
        let output_size: usize = result_shape.iter().product();
        let mut min_values = vec![f64::INFINITY; output_size];
        let mut min_indices = vec![0usize; output_size];

        // Iterate through tensor and find minimums
        Self::reduce_with_indices(
            tensor,
            axis,
            &mut min_values,
            &mut min_indices,
            |current, new_val, new_idx| {
                if new_val < current.0 {
                    (new_val, new_idx)
                } else {
                    *current
                }
            },
        )?;

        let min_tensor = Self::create_tensor_from_f64(
            &min_values,
            result_shape.clone(),
            tensor.options().clone(),
        )?;
        let idx_tensor =
            Self::create_indices_tensor(&min_indices, result_shape, tensor.options().clone())?;

        Ok((min_tensor, idx_tensor))
    }

    /// Max along axes with indices
    pub fn max_dim(tensor: &Tensor, axis: usize, keep_dim: bool) -> Result<(Tensor, Tensor)> {
        Self::validate_axes(tensor, &[axis])?;

        let axis_size = tensor.shape()[axis];
        let mut result_shape = tensor.shape().to_vec();

        if keep_dim {
            result_shape[axis] = 1;
        } else {
            result_shape.remove(axis);
        }

        let output_size: usize = result_shape.iter().product();
        let mut max_values = vec![f64::NEG_INFINITY; output_size];
        let mut max_indices = vec![0usize; output_size];

        Self::reduce_with_indices(
            tensor,
            axis,
            &mut max_values,
            &mut max_indices,
            |current, new_val, new_idx| {
                if new_val > current.0 {
                    (new_val, new_idx)
                } else {
                    *current
                }
            },
        )?;

        let max_tensor = Self::create_tensor_from_f64(
            &max_values,
            result_shape.clone(),
            tensor.options().clone(),
        )?;
        let idx_tensor =
            Self::create_indices_tensor(&max_indices, result_shape, tensor.options().clone())?;

        Ok((max_tensor, idx_tensor))
    }

    /// Argmax - indices of maximum values
    pub fn argmax(tensor: &Tensor, axis: Option<usize>, keep_dim: bool) -> Result<Tensor> {
        match axis {
            Some(ax) => {
                let (_, indices) = Self::max_dim(tensor, ax, keep_dim)?;
                Ok(indices)
            }
            None => {
                // Global argmax
                let flat_data = tensor.storage().to_vec_f64();
                let (max_idx, _) = flat_data
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or_else(|| CoreError::invalid_op("argmax", "Empty tensor"))?;

                let shape = if keep_dim {
                    vec![1; tensor.ndim()]
                } else {
                    vec![]
                };

                Self::create_indices_tensor(&[max_idx], shape, tensor.options().clone())
            }
        }
    }

    /// Argmin - indices of minimum values
    pub fn argmin(tensor: &Tensor, axis: Option<usize>, keep_dim: bool) -> Result<Tensor> {
        match axis {
            Some(ax) => {
                let (_, indices) = Self::min_dim(tensor, ax, keep_dim)?;
                Ok(indices)
            }
            None => {
                let flat_data = tensor.storage().to_vec_f64();
                let (min_idx, _) = flat_data
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or_else(|| CoreError::invalid_op("argmin", "Empty tensor"))?;

                let shape = if keep_dim {
                    vec![1; tensor.ndim()]
                } else {
                    vec![]
                };

                Self::create_indices_tensor(&[min_idx], shape, tensor.options().clone())
            }
        }
    }

    /// Cumulative sum along axis
    pub fn cumsum(tensor: &Tensor, axis: usize) -> Result<Tensor> {
        Self::validate_axes(tensor, &[axis])?;

        match tensor.storage() {
            crate::storage::StorageType::F32(data) => Self::cumsum_f32(tensor, axis, data),
            crate::storage::StorageType::F64(data) => Self::cumsum_f64(tensor, axis, data),
            crate::storage::StorageType::I32(data) => Self::cumsum_i32(tensor, axis, data),
            crate::storage::StorageType::I64(data) => Self::cumsum_i64(tensor, axis, data),
            _ => Err(CoreError::invalid_op("cumsum", "Unsupported data type")),
        }
    }

    /// Cumulative product along axis
    pub fn cumprod(tensor: &Tensor, axis: usize) -> Result<Tensor> {
        Self::validate_axes(tensor, &[axis])?;

        match tensor.storage() {
            crate::storage::StorageType::F32(data) => Self::cumprod_f32(tensor, axis, data),
            crate::storage::StorageType::F64(data) => Self::cumprod_f64(tensor, axis, data),
            crate::storage::StorageType::I32(data) => Self::cumprod_i32(tensor, axis, data),
            crate::storage::StorageType::I64(data) => Self::cumprod_i64(tensor, axis, data),
            _ => Err(CoreError::invalid_op("cumprod", "Unsupported data type")),
        }
    }

    /// Compute various norms
    pub fn norm(
        tensor: &Tensor,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keep_dim: bool,
    ) -> Result<Tensor> {
        let ord = ord.unwrap_or(2.0); // Default to L2 norm

        if let Some(axes) = dim {
            Self::validate_axes(tensor, axes)?;
        }

        match ord {
            f if f == 1.0 => Self::norm_l1(tensor, dim, keep_dim),
            f if f == 2.0 => Self::norm_l2(tensor, dim, keep_dim),
            f if f.is_infinite() && f > 0.0 => Self::norm_inf(tensor, dim, keep_dim),
            f if f.is_infinite() && f < 0.0 => Self::norm_neg_inf(tensor, dim, keep_dim),
            p => Self::norm_p(tensor, p, dim, keep_dim),
        }
    }

    /// Frobenius norm (L2 norm)
    pub fn frobenius_norm(tensor: &Tensor) -> Result<Tensor> {
        Self::norm_l2(tensor, None, false)
    }

    /// Nuclear norm (sum of singular values)
    pub fn nuclear_norm(tensor: &Tensor) -> Result<Tensor> {
        if tensor.ndim() != 2 {
            return Err(CoreError::invalid_op(
                "nuclear_norm",
                "Only 2D tensors supported",
            ));
        }
        // This would require SVD implementation
        Err(CoreError::invalid_op(
            "nuclear_norm",
            "SVD not yet implemented",
        ))
    }

    // Helper functions

    /// Validate that axes are within tensor dimensions
    fn validate_axes(tensor: &Tensor, axes: &[usize]) -> Result<()> {
        for &axis in axes {
            if axis >= tensor.ndim() {
                return Err(CoreError::dim_out_of_bounds(
                    axis,
                    tensor.ndim(),
                    "axis_reduction",
                ));
            }
        }

        // Check for duplicate axes
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort();
        for window in sorted_axes.windows(2) {
            if window[0] == window[1] {
                return Err(CoreError::invalid_op(
                    "axis_reduction",
                    &format!("Duplicate axis: {}", window[0]),
                ));
            }
        }

        Ok(())
    }

    /// Sum all elements
    fn sum_all(tensor: &Tensor) -> Result<Tensor> {
        let data = tensor.storage().to_vec_f64();
        let sum = match tensor.dtype() {
            rustytorch_core::DType::Float32 => {
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                F32Ops::sum(&f32_data) as f64
            }
            rustytorch_core::DType::Float64 => F64Ops::sum(&data),
            _ => data.iter().sum(),
        };

        Self::create_scalar_tensor(sum, tensor.options().clone())
    }

    /// Sum along a single axis
    fn sum_single_axis(tensor: &Tensor, axis: usize, keep_dim: bool) -> Result<Tensor> {
        let mut result_shape = tensor.shape().to_vec();

        if keep_dim {
            result_shape[axis] = 1;
        } else {
            result_shape.remove(axis);
        }

        let output_size: usize = result_shape.iter().product();
        let mut result_data = vec![0.0; output_size];

        // Optimized reduction along axis
        Self::reduce_along_axis(tensor, axis, &mut result_data, |acc, val| acc + val, 0.0)?;

        Self::create_tensor_from_f64(&result_data, result_shape, tensor.options().clone())
    }

    /// Generic reduction along axis
    fn reduce_along_axis<F>(
        tensor: &Tensor,
        axis: usize,
        result: &mut [f64],
        reduce_op: F,
        init_value: f64,
    ) -> Result<()>
    where
        F: Fn(f64, f64) -> f64 + Copy,
    {
        let shape = tensor.shape();
        let strides = tensor.strides();
        let data = tensor.storage().to_vec_f64();

        // Initialize result with init_value
        result.fill(init_value);

        // Calculate strides for output indexing
        let mut output_strides = Vec::new();
        for (i, &size) in shape.iter().enumerate() {
            if i != axis {
                output_strides.push(size);
            }
        }

        // Iterate through all elements
        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            // Convert flat index to multi-dimensional coordinates
            let coords = Self::flat_to_coords(flat_idx, shape);

            // Calculate output index (excluding the reduction axis)
            let mut output_idx = 0;
            let mut output_stride = 1;
            for i in (0..shape.len()).rev() {
                if i != axis {
                    output_idx += coords[i] * output_stride;
                    output_stride *= shape[i];
                }
            }

            // Apply reduction operation
            if let Some(value) = data.get(tensor.offset() + Self::coords_to_flat(&coords, strides))
            {
                result[output_idx] = reduce_op(result[output_idx], *value);
            }
        }

        Ok(())
    }

    /// Reduction with index tracking (for min/max with argmin/argmax)
    fn reduce_with_indices<F>(
        tensor: &Tensor,
        axis: usize,
        values: &mut [f64],
        indices: &mut [usize],
        reduce_op: F,
    ) -> Result<()>
    where
        F: Fn(&mut (f64, usize), f64, usize) -> (f64, usize) + Copy,
    {
        let shape = tensor.shape();
        let data = tensor.storage().to_vec_f64();

        // Initialize with first values along the axis
        for i in 0..values.len() {
            values[i] = f64::INFINITY; // Will be replaced
            indices[i] = 0;
        }

        // Iterate through tensor
        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            let coords = Self::flat_to_coords(flat_idx, shape);
            let axis_index = coords[axis];

            // Calculate output index
            let mut output_idx = 0;
            let mut output_stride = 1;
            for i in (0..shape.len()).rev() {
                if i != axis {
                    output_idx += coords[i] * output_stride;
                    output_stride *= shape[i];
                }
            }

            if let Some(&value) = data.get(flat_idx) {
                let mut current = (values[output_idx], indices[output_idx]);
                let (new_val, new_idx) = reduce_op(&mut current, value, axis_index);
                values[output_idx] = new_val;
                indices[output_idx] = new_idx;
            }
        }

        Ok(())
    }

    /// Helper: squared difference from mean
    fn squared_diff(tensor: &Tensor, mean: &Tensor) -> Result<Tensor> {
        // This would use broadcasting to subtract mean and then square
        // Simplified implementation
        Err(CoreError::invalid_op(
            "squared_diff",
            "broadcasting subtraction not implemented",
        ))
    }

    /// Helper: divide tensor by scalar
    fn divide_scalar(tensor: &Tensor, scalar: f64) -> Result<Tensor> {
        let data = tensor.storage().to_vec_f64();
        let result: Vec<f64> = data.iter().map(|&x| x / scalar).collect();
        Self::create_tensor_from_f64(&result, tensor.shape().to_vec(), tensor.options().clone())
    }

    /// Helper: square root
    fn sqrt(tensor: &Tensor) -> Result<Tensor> {
        let data = tensor.storage().to_vec_f64();
        let result: Vec<f64> = data.iter().map(|&x| x.sqrt()).collect();
        Self::create_tensor_from_f64(&result, tensor.shape().to_vec(), tensor.options().clone())
    }

    /// Create scalar tensor
    pub fn create_scalar_tensor(value: f64, options: TensorOptions) -> Result<Tensor> {
        let f32_value = value as f32;
        Ok(Tensor::from_data(&[f32_value], vec![], Some(options)))
    }

    /// Create tensor from f64 data
    fn create_tensor_from_f64(
        data: &[f64],
        shape: Vec<usize>,
        options: TensorOptions,
    ) -> Result<Tensor> {
        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_data(&f32_data, shape, Some(options)))
    }

    /// Create tensor with indices (as f32 for now)
    fn create_indices_tensor(
        indices: &[usize],
        shape: Vec<usize>,
        options: TensorOptions,
    ) -> Result<Tensor> {
        let f32_indices: Vec<f32> = indices.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_data(&f32_indices, shape, Some(options)))
    }

    /// Convert flat index to coordinates
    fn flat_to_coords(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; shape.len()];
        let mut idx = flat_idx;

        for i in (0..shape.len()).rev() {
            coords[i] = idx % shape[i];
            idx /= shape[i];
        }

        coords
    }

    /// Convert coordinates to flat index using strides
    fn coords_to_flat(coords: &[usize], strides: &[usize]) -> usize {
        coords
            .iter()
            .zip(strides.iter())
            .map(|(&coord, &stride)| coord * stride)
            .sum()
    }

    // Cumulative operations helpers

    /// Cumulative sum for F32 data
    fn cumsum_f32(tensor: &Tensor, axis: usize, data: &[f32]) -> Result<Tensor> {
        let shape = tensor.shape();
        let mut result_data = data.to_vec();

        // Calculate strides for efficient indexing
        let strides = Self::calculate_strides(shape);
        let axis_stride = strides[axis];
        let axis_size = shape[axis];

        // Iterate through all positions and accumulate along axis
        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            let coords = Self::flat_to_coords(flat_idx, shape);
            let axis_pos = coords[axis];

            // Skip the first element along axis (already correct)
            if axis_pos == 0 {
                continue;
            }

            // Calculate indices for current and previous position
            let current_idx = flat_idx;
            let prev_idx = flat_idx - axis_stride;

            result_data[current_idx] = result_data[prev_idx] + data[current_idx];
        }

        Ok(Tensor::from_data(
            &result_data,
            shape.to_vec(),
            Some(tensor.options().clone()),
        ))
    }

    /// Cumulative sum for F64 data
    fn cumsum_f64(tensor: &Tensor, axis: usize, data: &[f64]) -> Result<Tensor> {
        let shape = tensor.shape();
        let mut result_data = data.to_vec();

        let strides = Self::calculate_strides(shape);
        let axis_stride = strides[axis];

        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            let coords = Self::flat_to_coords(flat_idx, shape);
            let axis_pos = coords[axis];

            if axis_pos == 0 {
                continue;
            }

            let current_idx = flat_idx;
            let prev_idx = flat_idx - axis_stride;

            result_data[current_idx] = result_data[prev_idx] + data[current_idx];
        }

        // Convert to f32 for compatibility
        let f32_data: Vec<f32> = result_data.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_data(
            &f32_data,
            shape.to_vec(),
            Some(tensor.options().clone()),
        ))
    }

    /// Cumulative sum for I32 data
    fn cumsum_i32(tensor: &Tensor, axis: usize, data: &[i32]) -> Result<Tensor> {
        let shape = tensor.shape();
        let mut result_data = data.to_vec();

        let strides = Self::calculate_strides(shape);
        let axis_stride = strides[axis];

        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            let coords = Self::flat_to_coords(flat_idx, shape);
            let axis_pos = coords[axis];

            if axis_pos == 0 {
                continue;
            }

            let current_idx = flat_idx;
            let prev_idx = flat_idx - axis_stride;

            result_data[current_idx] = result_data[prev_idx] + data[current_idx];
        }

        // Convert to f32 for compatibility
        let f32_data: Vec<f32> = result_data.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_data(
            &f32_data,
            shape.to_vec(),
            Some(tensor.options().clone()),
        ))
    }

    /// Cumulative sum for I64 data
    fn cumsum_i64(tensor: &Tensor, axis: usize, data: &[i64]) -> Result<Tensor> {
        let shape = tensor.shape();
        let mut result_data = data.to_vec();

        let strides = Self::calculate_strides(shape);
        let axis_stride = strides[axis];

        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            let coords = Self::flat_to_coords(flat_idx, shape);
            let axis_pos = coords[axis];

            if axis_pos == 0 {
                continue;
            }

            let current_idx = flat_idx;
            let prev_idx = flat_idx - axis_stride;

            result_data[current_idx] = result_data[prev_idx] + data[current_idx];
        }

        // Convert to f32 for compatibility
        let f32_data: Vec<f32> = result_data.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_data(
            &f32_data,
            shape.to_vec(),
            Some(tensor.options().clone()),
        ))
    }

    // Cumulative product helpers

    /// Cumulative product for F32 data
    fn cumprod_f32(tensor: &Tensor, axis: usize, data: &[f32]) -> Result<Tensor> {
        let shape = tensor.shape();
        let mut result_data = data.to_vec();

        let strides = Self::calculate_strides(shape);
        let axis_stride = strides[axis];

        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            let coords = Self::flat_to_coords(flat_idx, shape);
            let axis_pos = coords[axis];

            if axis_pos == 0 {
                continue;
            }

            let current_idx = flat_idx;
            let prev_idx = flat_idx - axis_stride;

            result_data[current_idx] = result_data[prev_idx] * data[current_idx];
        }

        Ok(Tensor::from_data(
            &result_data,
            shape.to_vec(),
            Some(tensor.options().clone()),
        ))
    }

    /// Cumulative product for F64 data
    fn cumprod_f64(tensor: &Tensor, axis: usize, data: &[f64]) -> Result<Tensor> {
        let shape = tensor.shape();
        let mut result_data = data.to_vec();

        let strides = Self::calculate_strides(shape);
        let axis_stride = strides[axis];

        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            let coords = Self::flat_to_coords(flat_idx, shape);
            let axis_pos = coords[axis];

            if axis_pos == 0 {
                continue;
            }

            let current_idx = flat_idx;
            let prev_idx = flat_idx - axis_stride;

            result_data[current_idx] = result_data[prev_idx] * data[current_idx];
        }

        // Convert to f32 for compatibility
        let f32_data: Vec<f32> = result_data.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_data(
            &f32_data,
            shape.to_vec(),
            Some(tensor.options().clone()),
        ))
    }

    /// Cumulative product for I32 data
    fn cumprod_i32(tensor: &Tensor, axis: usize, data: &[i32]) -> Result<Tensor> {
        let shape = tensor.shape();
        let mut result_data = data.to_vec();

        let strides = Self::calculate_strides(shape);
        let axis_stride = strides[axis];

        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            let coords = Self::flat_to_coords(flat_idx, shape);
            let axis_pos = coords[axis];

            if axis_pos == 0 {
                continue;
            }

            let current_idx = flat_idx;
            let prev_idx = flat_idx - axis_stride;

            result_data[current_idx] = result_data[prev_idx] * data[current_idx];
        }

        // Convert to f32 for compatibility
        let f32_data: Vec<f32> = result_data.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_data(
            &f32_data,
            shape.to_vec(),
            Some(tensor.options().clone()),
        ))
    }

    /// Cumulative product for I64 data
    fn cumprod_i64(tensor: &Tensor, axis: usize, data: &[i64]) -> Result<Tensor> {
        let shape = tensor.shape();
        let mut result_data = data.to_vec();

        let strides = Self::calculate_strides(shape);
        let axis_stride = strides[axis];

        let total_elements = tensor.numel();
        for flat_idx in 0..total_elements {
            let coords = Self::flat_to_coords(flat_idx, shape);
            let axis_pos = coords[axis];

            if axis_pos == 0 {
                continue;
            }

            let current_idx = flat_idx;
            let prev_idx = flat_idx - axis_stride;

            result_data[current_idx] = result_data[prev_idx] * data[current_idx];
        }

        // Convert to f32 for compatibility
        let f32_data: Vec<f32> = result_data.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_data(
            &f32_data,
            shape.to_vec(),
            Some(tensor.options().clone()),
        ))
    }

    // Norm helpers

    /// L1 norm (Manhattan norm)
    fn norm_l1(tensor: &Tensor, dim: Option<&[usize]>, keep_dim: bool) -> Result<Tensor> {
        // Compute |x|, then sum
        let abs_tensor = Self::abs_tensor(tensor)?;

        match dim {
            Some(axes) => Self::sum_dim(&abs_tensor, axes, keep_dim),
            None => Self::sum_all(&abs_tensor),
        }
    }

    /// L2 norm (Euclidean norm)
    fn norm_l2(tensor: &Tensor, dim: Option<&[usize]>, keep_dim: bool) -> Result<Tensor> {
        // Compute x^2, then sum, then sqrt
        let squared_tensor = Self::square_tensor(tensor)?;

        let sum_result = match dim {
            Some(axes) => Self::sum_dim(&squared_tensor, axes, keep_dim)?,
            None => Self::sum_all(&squared_tensor)?,
        };

        Self::sqrt(&sum_result)
    }

    /// L-infinity norm (max norm)
    fn norm_inf(tensor: &Tensor, dim: Option<&[usize]>, keep_dim: bool) -> Result<Tensor> {
        let abs_tensor = Self::abs_tensor(tensor)?;

        match dim {
            Some(axes) => {
                if axes.len() == 1 {
                    let (values, _) = Self::max_dim(&abs_tensor, axes[0], keep_dim)?;
                    Ok(values)
                } else {
                    // Multiple axes - need to reduce iteratively
                    let mut result = abs_tensor;
                    let mut sorted_axes = axes.to_vec();
                    sorted_axes.sort_by(|a, b| b.cmp(a)); // Descending order

                    for &axis in &sorted_axes {
                        let adjusted_axis = if keep_dim {
                            axis
                        } else {
                            let count = sorted_axes.iter().filter(|&&a| a > axis).count();
                            if axis >= count {
                                axis - count
                            } else {
                                0
                            }
                        };
                        let (values, _) = Self::max_dim(&result, adjusted_axis, keep_dim)?;
                        result = values;
                    }
                    Ok(result)
                }
            }
            None => {
                // Global max: flatten and find maximum
                let data = abs_tensor.storage().to_vec_f64();
                let max_val = data.iter().fold(0.0f64, |acc, &x| acc.max(x));
                Self::create_scalar_tensor(max_val, tensor.options().clone())
            }
        }
    }

    /// L-negative-infinity norm (min norm)
    fn norm_neg_inf(tensor: &Tensor, dim: Option<&[usize]>, keep_dim: bool) -> Result<Tensor> {
        let abs_tensor = Self::abs_tensor(tensor)?;

        match dim {
            Some(axes) => {
                if axes.len() == 1 {
                    let (values, _) = Self::min_dim(&abs_tensor, axes[0], keep_dim)?;
                    Ok(values)
                } else {
                    // Multiple axes - need to reduce iteratively
                    let mut result = abs_tensor;
                    let mut sorted_axes = axes.to_vec();
                    sorted_axes.sort_by(|a, b| b.cmp(a)); // Descending order

                    for &axis in &sorted_axes {
                        let adjusted_axis = if keep_dim {
                            axis
                        } else {
                            let count = sorted_axes.iter().filter(|&&a| a > axis).count();
                            if axis >= count {
                                axis - count
                            } else {
                                0
                            }
                        };
                        let (values, _) = Self::min_dim(&result, adjusted_axis, keep_dim)?;
                        result = values;
                    }
                    Ok(result)
                }
            }
            None => {
                // Global min: flatten and find minimum
                let data = abs_tensor.storage().to_vec_f64();
                let min_val = data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
                Self::create_scalar_tensor(min_val, tensor.options().clone())
            }
        }
    }

    /// General p-norm
    fn norm_p(tensor: &Tensor, p: f64, dim: Option<&[usize]>, keep_dim: bool) -> Result<Tensor> {
        // Compute |x|^p, then sum, then take p-th root
        let abs_tensor = Self::abs_tensor(tensor)?;
        let powered_tensor = Self::pow_tensor(&abs_tensor, p)?;

        let sum_result = match dim {
            Some(axes) => Self::sum_dim(&powered_tensor, axes, keep_dim)?,
            None => Self::sum_all(&powered_tensor)?,
        };

        Self::pow_tensor(&sum_result, 1.0 / p)
    }

    // Helper tensor operations

    /// Compute absolute value of tensor
    fn abs_tensor(tensor: &Tensor) -> Result<Tensor> {
        let data = tensor.storage().to_vec_f64();
        let abs_data: Vec<f64> = data.iter().map(|&x| x.abs()).collect();
        Self::create_tensor_from_f64(&abs_data, tensor.shape().to_vec(), tensor.options().clone())
    }

    /// Square all elements of tensor
    fn square_tensor(tensor: &Tensor) -> Result<Tensor> {
        let data = tensor.storage().to_vec_f64();
        let squared_data: Vec<f64> = data.iter().map(|&x| x * x).collect();
        Self::create_tensor_from_f64(
            &squared_data,
            tensor.shape().to_vec(),
            tensor.options().clone(),
        )
    }

    /// Raise tensor to power p
    fn pow_tensor(tensor: &Tensor, p: f64) -> Result<Tensor> {
        let data = tensor.storage().to_vec_f64();
        let powered_data: Vec<f64> = data.iter().map(|&x| x.powf(p)).collect();
        Self::create_tensor_from_f64(
            &powered_data,
            tensor.shape().to_vec(),
            tensor.options().clone(),
        )
    }

    /// Calculate strides for a given shape
    fn calculate_strides(shape: &[usize]) -> Vec<usize> {
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

    fn create_test_tensor_3d() -> Tensor {
        // 2x3x4 tensor
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        Tensor::from_data(&data, vec![2, 3, 4], None)
    }

    fn create_test_tensor_2d() -> Tensor {
        Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None)
    }

    #[test]
    fn test_sum_all() {
        let tensor = create_test_tensor_2d();
        let result = AxisReductions::sum_dim(&tensor, &[], false).unwrap();

        assert_eq!(result.shape(), &[]);
        let sum_value = result.storage().get_f64(0).unwrap();
        assert!((sum_value - 21.0).abs() < 1e-6); // 1+2+3+4+5+6 = 21
    }

    #[test]
    fn test_sum_single_axis() {
        let tensor = create_test_tensor_2d(); // [[1,2,3], [4,5,6]]

        // Sum along axis 0 (rows)
        let result = AxisReductions::sum_dim(&tensor, &[0], false).unwrap();
        assert_eq!(result.shape(), &[3]);
        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]

        // Sum along axis 1 (columns) with keepdim
        let result = AxisReductions::sum_dim(&tensor, &[1], true).unwrap();
        assert_eq!(result.shape(), &[2, 1]);
        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_mean_axis() {
        let tensor = create_test_tensor_2d();

        let result = AxisReductions::mean_dim(&tensor, &[1], false).unwrap();
        assert_eq!(result.shape(), &[2]);
        let data = result.storage().to_vec_f64();
        assert!((data[0] - 2.0).abs() < 1e-6); // (1+2+3)/3 = 2
        assert!((data[1] - 5.0).abs() < 1e-6); // (4+5+6)/3 = 5
    }

    #[test]
    fn test_argmax_global() {
        let tensor = create_test_tensor_2d();

        let result = AxisReductions::argmax(&tensor, None, false).unwrap();
        assert_eq!(result.shape(), &[]);
        let idx = result.storage().get_f64(0).unwrap() as usize;
        assert_eq!(idx, 5); // Index of maximum value (6.0)
    }

    #[test]
    fn test_cumsum() {
        // Test 1D cumsum
        let tensor_1d = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], None);
        let result = AxisReductions::cumsum(&tensor_1d, 0).unwrap();
        assert_eq!(result.shape(), &[4]);
        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![1.0, 3.0, 6.0, 10.0]); // [1, 1+2, 1+2+3, 1+2+3+4]

        // Test 2D cumsum along axis 0
        let tensor_2d = create_test_tensor_2d(); // [[1,2,3], [4,5,6]]
        let result = AxisReductions::cumsum(&tensor_2d, 0).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]); // [[1,2,3], [1+4,2+5,3+6]]

        // Test 2D cumsum along axis 1
        let result = AxisReductions::cumsum(&tensor_2d, 1).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]); // [[1,1+2,1+2+3], [4,4+5,4+5+6]]
    }

    #[test]
    fn test_cumprod() {
        // Test 1D cumprod
        let tensor_1d = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], None);
        let result = AxisReductions::cumprod(&tensor_1d, 0).unwrap();
        assert_eq!(result.shape(), &[4]);
        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![1.0, 2.0, 6.0, 24.0]); // [1, 1*2, 1*2*3, 1*2*3*4]

        // Test 2D cumprod along axis 0
        let tensor_2d = create_test_tensor_2d(); // [[1,2,3], [4,5,6]]
        let result = AxisReductions::cumprod(&tensor_2d, 0).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 10.0, 18.0]); // [[1,2,3], [1*4,2*5,3*6]]

        // Test 2D cumprod along axis 1
        let result = AxisReductions::cumprod(&tensor_2d, 1).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]); // [[1,1*2,1*2*3], [4,4*5,4*5*6]]
    }

    #[test]
    fn test_norm_l1() {
        let tensor = Tensor::from_data(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5], None);

        // Global L1 norm
        let result = AxisReductions::norm(&tensor, Some(1.0), None, false).unwrap();
        assert_eq!(result.shape(), &[]);
        let norm_value = result.storage().get_f64(0).unwrap();
        assert!((norm_value - 6.0).abs() < 1e-6); // |-2|+|-1|+|0|+|1|+|2| = 6
    }

    #[test]
    fn test_norm_l2() {
        let tensor = Tensor::from_data(&[3.0f32, 4.0], vec![2], None);

        // Global L2 norm (Euclidean norm)
        let result = AxisReductions::norm(&tensor, Some(2.0), None, false).unwrap();
        assert_eq!(result.shape(), &[]);
        let norm_value = result.storage().get_f64(0).unwrap();
        assert!((norm_value - 5.0).abs() < 1e-6); // sqrt(3²+4²) = sqrt(9+16) = 5
    }

    #[test]
    fn test_norm_inf() {
        let tensor = Tensor::from_data(&[-5.0f32, 3.0, -1.0, 4.0], vec![4], None);

        // Global L-infinity norm (max absolute value)
        let result = AxisReductions::norm(&tensor, Some(f64::INFINITY), None, false).unwrap();
        assert_eq!(result.shape(), &[]);
        let norm_value = result.storage().get_f64(0).unwrap();
        assert!((norm_value - 5.0).abs() < 1e-6); // max(|-5|,|3|,|-1|,|4|) = 5
    }

    #[test]
    fn test_norm_p() {
        let tensor = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], None);

        // L3 norm (p=3)
        let result = AxisReductions::norm(&tensor, Some(3.0), None, false).unwrap();
        assert_eq!(result.shape(), &[]);
        let norm_value = result.storage().get_f64(0).unwrap();
        let expected = (1.0_f64.powf(3.0) + 2.0_f64.powf(3.0) + 3.0_f64.powf(3.0)).powf(1.0 / 3.0);
        assert!((norm_value - expected).abs() < 1e-5); // (1³+2³+3³)^(1/3) = (1+8+27)^(1/3) = 36^(1/3)
    }

    #[test]
    fn test_norm_with_dims() {
        let tensor = create_test_tensor_2d(); // [[1,2,3], [4,5,6]]

        // L2 norm along axis 1
        let result = AxisReductions::norm(&tensor, Some(2.0), Some(&[1]), false).unwrap();
        assert_eq!(result.shape(), &[2]);
        let data = result.storage().to_vec_f64();
        let expected_0 = (1.0f64 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt(); // sqrt(14)
        let expected_1 = (4.0f64 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0).sqrt(); // sqrt(77)
        assert!((data[0] - expected_0).abs() < 1e-5);
        assert!((data[1] - expected_1).abs() < 1e-5);
    }

    #[test]
    fn test_frobenius_norm() {
        let tensor = create_test_tensor_2d(); // [[1,2,3], [4,5,6]]

        let result = AxisReductions::frobenius_norm(&tensor).unwrap();
        assert_eq!(result.shape(), &[]);
        let norm_value = result.storage().get_f64(0).unwrap();
        let expected =
            (1.0f64 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0).sqrt();
        assert!((norm_value - expected).abs() < 1e-5); // sqrt(1+4+9+16+25+36) = sqrt(91)
    }

    #[test]
    fn test_cumsum_cumprod_integer_types() {
        use rustytorch_core::{DType, TensorOptions};

        // Test with I32
        let tensor_i32 = Tensor::from_data(
            &[1, 2, 3, 4],
            vec![4],
            Some(TensorOptions::new().dtype(DType::Int32)),
        );
        let cumsum_result = AxisReductions::cumsum(&tensor_i32, 0).unwrap();
        let cumsum_data = cumsum_result.storage().to_vec_f64();
        assert_eq!(cumsum_data, vec![1.0, 3.0, 6.0, 10.0]);

        let cumprod_result = AxisReductions::cumprod(&tensor_i32, 0).unwrap();
        let cumprod_data = cumprod_result.storage().to_vec_f64();
        assert_eq!(cumprod_data, vec![1.0, 2.0, 6.0, 24.0]);

        // Test with I64 (convert to f64 first)
        let tensor_i64 = Tensor::from_data(
            &[2.0f64, 3.0, 4.0],
            vec![3],
            Some(TensorOptions::new().dtype(DType::Int64)),
        );
        let cumsum_result = AxisReductions::cumsum(&tensor_i64, 0).unwrap();
        let cumsum_data = cumsum_result.storage().to_vec_f64();
        assert_eq!(cumsum_data, vec![2.0, 5.0, 9.0]);
    }

    #[test]
    fn test_argmin_axis() {
        let tensor = create_test_tensor_2d();

        let result = AxisReductions::argmin(&tensor, Some(1), false).unwrap();
        assert_eq!(result.shape(), &[2]);
        let indices = result.storage().to_vec_f64();
        assert_eq!(indices[0] as usize, 0); // First row min at index 0
        assert_eq!(indices[1] as usize, 0); // Second row min at index 0
    }

    #[test]
    fn test_min_max_with_indices() {
        let tensor = create_test_tensor_2d();

        let (min_vals, min_indices) = AxisReductions::min_dim(&tensor, 1, false).unwrap();
        assert_eq!(min_vals.shape(), &[2]);
        assert_eq!(min_indices.shape(), &[2]);

        let min_data = min_vals.storage().to_vec_f64();
        let idx_data = min_indices.storage().to_vec_f64();

        assert_eq!(min_data, vec![1.0, 4.0]); // Min values per row
        assert_eq!(idx_data, vec![0.0, 0.0]); // Indices of min values
    }

    #[test]
    fn test_multiple_axes() {
        let tensor = create_test_tensor_3d();

        // Sum along axes 0 and 2
        let result = AxisReductions::sum_dim(&tensor, &[0, 2], false).unwrap();
        assert_eq!(result.shape(), &[3]); // Only axis 1 remains
    }

    #[test]
    fn test_validation() {
        let tensor = create_test_tensor_2d();

        // Test invalid axis
        assert!(AxisReductions::sum_dim(&tensor, &[5], false).is_err());

        // Test duplicate axes
        assert!(AxisReductions::sum_dim(&tensor, &[0, 0], false).is_err());
    }
}
