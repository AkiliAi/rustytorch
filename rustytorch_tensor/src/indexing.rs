//! Advanced indexing and slicing operations for tensors
//!
//! This module implements fancy indexing, masked selection, gather/scatter operations
//! and other advanced indexing patterns compatible with PyTorch.

use crate::{storage::StorageType, Tensor};
use rustytorch_core::{CoreError, Indexable, Result};
use std::ops::Range;

/// Represents different types of indices for advanced indexing
#[derive(Debug, Clone)]
pub enum IndexType {
    /// Single integer index
    Single(usize),

    /// Range of indices (start..end)
    Range(Range<usize>),

    /// Full slice (..)
    FullSlice,

    /// Array of indices for fancy indexing
    Array(Vec<usize>),

    /// Boolean mask for masked selection
    Mask(Vec<bool>),

    /// Tensor indices for advanced indexing
    TensorIndices(Tensor),
}

/// Multi-dimensional index specification
#[derive(Debug, Clone)]
pub struct MultiIndex {
    indices: Vec<IndexType>,
}

impl MultiIndex {
    /// Create a new multi-index
    pub fn new(indices: Vec<IndexType>) -> Self {
        Self { indices }
    }

    /// Create from simple ranges
    pub fn from_ranges(ranges: Vec<Range<usize>>) -> Self {
        let indices = ranges.into_iter().map(IndexType::Range).collect();
        Self { indices }
    }

    /// Get the indices
    pub fn indices(&self) -> &[IndexType] {
        &self.indices
    }
}

/// Advanced indexing operations
pub struct AdvancedIndexing;

impl AdvancedIndexing {
    /// Fancy indexing - select elements using arrays of indices
    pub fn fancy_index(tensor: &Tensor, indices: &[Vec<usize>]) -> Result<Tensor> {
        if indices.len() != tensor.ndim() {
            return Err(CoreError::invalid_op(
                "fancy_index",
                &format!(
                    "Index arrays length {} != tensor dimensions {}",
                    indices.len(),
                    tensor.ndim()
                ),
            ));
        }

        // Validate indices are same length
        if indices.len() > 1 {
            let first_len = indices[0].len();
            for (i, idx_array) in indices.iter().enumerate() {
                if idx_array.len() != first_len {
                    return Err(CoreError::invalid_op(
                        "fancy_index",
                        &format!(
                            "Index array {} has length {} != {}",
                            i,
                            idx_array.len(),
                            first_len
                        ),
                    ));
                }
            }
        }

        // Validate all indices are in bounds
        for (dim, idx_array) in indices.iter().enumerate() {
            for &idx in idx_array {
                if idx >= tensor.shape()[dim] {
                    return Err(CoreError::index_out_of_bounds(
                        vec![idx],
                        tensor.shape().to_vec(),
                    ));
                }
            }
        }

        let result_len = if indices.is_empty() {
            0
        } else {
            indices[0].len()
        };
        let mut result_data = Vec::new();

        // Extract elements based on fancy indices
        for i in 0..result_len {
            let mut linear_idx = tensor.offset();
            for (dim, idx_array) in indices.iter().enumerate() {
                linear_idx += idx_array[i] * tensor.strides()[dim];
            }

            // Extract value from storage at linear_idx
            if let Some(value) = Self::get_storage_value(&tensor.storage(), linear_idx) {
                result_data.push(value);
            } else {
                return Err(CoreError::invalid_op(
                    "fancy_index",
                    &format!("Storage access failed at index {}", linear_idx),
                ));
            }
        }

        // Create result tensor
        Self::create_tensor_from_data(&result_data, vec![result_len], tensor.options().clone())
    }

    /// Masked selection - select elements where mask is true
    pub fn masked_select(tensor: &Tensor, mask: &[bool]) -> Result<Tensor> {
        if mask.len() != tensor.numel() {
            return Err(CoreError::shape_mismatch(
                vec![tensor.numel()],
                vec![mask.len()],
                "masked_select",
            ));
        }

        let mut result_data = Vec::new();
        let flat_data = Self::flatten_tensor_data(tensor)?;

        for (i, &should_select) in mask.iter().enumerate() {
            if should_select {
                result_data.push(flat_data[i]);
            }
        }

        // Result is always 1D
        Self::create_tensor_from_data(
            &result_data,
            vec![result_data.len()],
            tensor.options().clone(),
        )
    }

    /// Masked selection with boolean tensor
    pub fn masked_select_tensor(tensor: &Tensor, mask_tensor: &Tensor) -> Result<Tensor> {
        // Convert mask tensor to boolean array
        let mask_data = Self::tensor_to_bool_array(mask_tensor)?;
        Self::masked_select(tensor, &mask_data)
    }

    /// Gather operation - collect values along an axis using indices
    pub fn gather(tensor: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        if dim >= tensor.ndim() {
            return Err(CoreError::dim_out_of_bounds(dim, tensor.ndim(), "gather"));
        }

        // Indices must be integer tensor
        let index_data = Self::tensor_to_index_array(indices)?;

        // Result shape is same as indices shape
        let result_shape = indices.shape().to_vec();
        let mut result_data = Vec::new();

        // Iterate through indices tensor
        for (pos, &idx) in index_data.iter().enumerate() {
            // Convert flat position to multi-dimensional coordinates for indices tensor
            let indices_coords = Self::flat_to_coords(pos, indices.shape());

            // Create coordinates for source tensor by replacing dim with gathered index
            let mut source_coords = indices_coords.clone();
            if source_coords.len() <= dim {
                source_coords.resize(tensor.ndim(), 0);
            }
            source_coords[dim] = idx;

            // Validate source coordinates
            for (d, &coord) in source_coords.iter().enumerate() {
                if coord >= tensor.shape()[d] {
                    return Err(CoreError::index_out_of_bounds(
                        source_coords.clone(),
                        tensor.shape().to_vec(),
                    ));
                }
            }

            // Calculate linear index in source tensor
            let linear_idx = Self::coords_to_linear(
                &source_coords,
                tensor.shape(),
                tensor.strides(),
                tensor.offset(),
            );

            // Extract value
            if let Some(value) = Self::get_storage_value(&tensor.storage(), linear_idx) {
                result_data.push(value);
            } else {
                return Err(CoreError::invalid_op(
                    "gather",
                    &format!("Storage access failed at index {}", linear_idx),
                ));
            }
        }

        Self::create_tensor_from_data(&result_data, result_shape, tensor.options().clone())
    }

    /// Scatter operation - place values at specified indices
    pub fn scatter(
        tensor: &mut Tensor,
        dim: usize,
        indices: &Tensor,
        values: &Tensor,
    ) -> Result<()> {
        if dim >= tensor.ndim() {
            return Err(CoreError::dim_out_of_bounds(dim, tensor.ndim(), "scatter"));
        }

        if indices.shape() != values.shape() {
            return Err(CoreError::shape_mismatch(
                indices.shape().to_vec(),
                values.shape().to_vec(),
                "scatter",
            ));
        }

        let index_data = Self::tensor_to_index_array(indices)?;
        let value_data = Self::flatten_tensor_data(values)?;

        // This is a simplified implementation
        // In practice, you'd need mutable access to tensor storage
        Err(CoreError::invalid_op(
            "scatter",
            "mutable storage access not yet implemented",
        ))
    }

    /// Advanced slicing with step support
    pub fn slice_with_step(
        tensor: &Tensor,
        ranges: &[(usize, usize, usize)], // (start, end, step)
    ) -> Result<Tensor> {
        if ranges.len() > tensor.ndim() {
            return Err(CoreError::invalid_op(
                "slice_with_step",
                &format!(
                    "Too many slice dimensions: {} > {}",
                    ranges.len(),
                    tensor.ndim()
                ),
            ));
        }

        let mut new_shape = tensor.shape().to_vec();
        let mut new_strides = tensor.strides().to_vec();
        let mut new_offset = tensor.offset();

        // Apply slicing with step to each dimension
        for (dim, &(start, end, step)) in ranges.iter().enumerate() {
            if end > tensor.shape()[dim] {
                return Err(CoreError::invalid_op(
                    "slice_with_step",
                    &format!("Slice end {} > dimension size {}", end, tensor.shape()[dim]),
                ));
            }

            if step == 0 {
                return Err(CoreError::invalid_op(
                    "slice_with_step",
                    "Step cannot be zero",
                ));
            }

            // Update offset for this dimension
            new_offset += start * tensor.strides()[dim];

            // Update shape and stride for this dimension
            new_shape[dim] = (end - start + step - 1) / step; // Ceiling division
            new_strides[dim] = tensor.strides()[dim] * step;
        }

        // Create result tensor with new layout
        // This is simplified - in practice you'd create a view or copy data
        Err(CoreError::invalid_op(
            "slice_with_step",
            "tensor creation from layout not yet implemented",
        ))
    }

    /// Nonzero operation - find indices of non-zero elements
    pub fn nonzero(tensor: &Tensor) -> Result<Vec<Vec<usize>>> {
        let flat_data = Self::flatten_tensor_data(tensor)?;
        let mut nonzero_indices = Vec::new();

        for (flat_idx, &value) in flat_data.iter().enumerate() {
            if value != 0.0 {
                // Assuming f64 representation
                let coords = Self::flat_to_coords(flat_idx, tensor.shape());
                nonzero_indices.push(coords);
            }
        }

        Ok(nonzero_indices)
    }

    /// Where operation - select elements based on condition
    pub fn where_condition(condition: &[bool], x: &Tensor, y: &Tensor) -> Result<Tensor> {
        if condition.len() != x.numel() || x.numel() != y.numel() {
            return Err(CoreError::invalid_op(
                "where",
                "Condition, x, and y must have same number of elements",
            ));
        }

        if x.shape() != y.shape() {
            return Err(CoreError::shape_mismatch(
                x.shape().to_vec(),
                y.shape().to_vec(),
                "where",
            ));
        }

        let x_data = Self::flatten_tensor_data(x)?;
        let y_data = Self::flatten_tensor_data(y)?;
        let mut result_data = Vec::new();

        for (i, &cond) in condition.iter().enumerate() {
            if cond {
                result_data.push(x_data[i]);
            } else {
                result_data.push(y_data[i]);
            }
        }

        Self::create_tensor_from_data(&result_data, x.shape().to_vec(), x.options().clone())
    }

    // Helper functions

    /// Get value from storage at linear index
    fn get_storage_value(storage: &StorageType, index: usize) -> Option<f64> {
        storage.get_f64(index)
    }

    /// Flatten tensor data to f64 vector
    fn flatten_tensor_data(tensor: &Tensor) -> Result<Vec<f64>> {
        Ok(tensor.storage().to_vec_f64())
    }

    /// Convert tensor to boolean array
    fn tensor_to_bool_array(tensor: &Tensor) -> Result<Vec<bool>> {
        let data = Self::flatten_tensor_data(tensor)?;
        Ok(data.iter().map(|&x| x != 0.0).collect())
    }

    /// Convert tensor to index array
    fn tensor_to_index_array(tensor: &Tensor) -> Result<Vec<usize>> {
        let data = Self::flatten_tensor_data(tensor)?;
        let indices: std::result::Result<Vec<usize>, CoreError> = data
            .iter()
            .map(|&x| {
                if x >= 0.0 && x.fract() == 0.0 {
                    Ok(x as usize)
                } else {
                    Err(CoreError::invalid_op(
                        "tensor_to_indices",
                        &format!("Invalid index value: {}", x),
                    ))
                }
            })
            .collect();
        indices
    }

    /// Convert flat index to multi-dimensional coordinates
    fn flat_to_coords(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; shape.len()];
        let mut idx = flat_idx;

        for i in (0..shape.len()).rev() {
            coords[i] = idx % shape[i];
            idx /= shape[i];
        }

        coords
    }

    /// Convert multi-dimensional coordinates to linear index
    fn coords_to_linear(
        coords: &[usize],
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> usize {
        let mut linear_idx = offset;
        for (i, &coord) in coords.iter().enumerate() {
            if i < strides.len() {
                linear_idx += coord * strides[i];
            }
        }
        linear_idx
    }

    /// Create tensor from data (simplified)
    fn create_tensor_from_data(
        data: &[f64],
        shape: Vec<usize>,
        options: rustytorch_core::TensorOptions,
    ) -> Result<Tensor> {
        // Convert f64 data to appropriate type based on options.dtype
        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_data(&f32_data, shape, Some(options)))
    }
}

/// Implement Indexable trait for Tensor
impl Indexable for Tensor {
    type Output = f64;
    type Index = Tensor;

    fn get(&self, indices: &[usize]) -> Result<Self::Output> {
        if indices.len() != self.ndim() {
            return Err(CoreError::invalid_op(
                "get",
                &format!(
                    "Index length {} != tensor dimensions {}",
                    indices.len(),
                    self.ndim()
                ),
            ));
        }

        // Validate indices
        for (dim, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[dim] {
                return Err(CoreError::index_out_of_bounds(
                    indices.to_vec(),
                    self.shape().to_vec(),
                ));
            }
        }

        // Calculate linear index
        let mut linear_idx = self.offset();
        for (dim, &idx) in indices.iter().enumerate() {
            linear_idx += idx * self.strides()[dim];
        }

        // Get value from storage
        self.storage().get_f64(linear_idx).ok_or_else(|| {
            CoreError::invalid_op(
                "get",
                &format!("Storage access failed at index {}", linear_idx),
            )
        })
    }

    fn set(&mut self, indices: &[usize], value: Self::Output) -> Result<()> {
        if indices.len() != self.ndim() {
            return Err(CoreError::invalid_op(
                "set",
                &format!(
                    "Index length {} != tensor dimensions {}",
                    indices.len(),
                    self.ndim()
                ),
            ));
        }

        // Validate indices
        for (dim, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[dim] {
                return Err(CoreError::index_out_of_bounds(
                    indices.to_vec(),
                    self.shape().to_vec(),
                ));
            }
        }

        // Calculate linear index
        let mut linear_idx = self.offset();
        for (dim, &idx) in indices.iter().enumerate() {
            linear_idx += idx * self.strides()[dim];
        }

        // For now, we can't mutate Arc<StorageType> directly
        // This would require implementing a mutable storage system
        // This is a fundamental design limitation that would need architectural changes
        Err(CoreError::invalid_op(
            "set",
            "In-place modification requires mutable storage design - use tensor operations instead",
        ))
    }

    fn slice(&self, ranges: &[Range<usize>]) -> Result<Self> {
        // Use existing slice_ranges functionality from tensor_ops
        self.slice_ranges(ranges)
            .map_err(|e| CoreError::invalid_op("slice", &e.to_string()))
    }

    fn index(&self, indices: &Self::Index) -> Result<Self> {
        // Advanced indexing using tensor indices
        let index_data = AdvancedIndexing::tensor_to_index_array(indices)?;

        // For now, assume 1D indexing
        if self.ndim() != 1 {
            return Err(CoreError::invalid_op(
                "index",
                "Multi-dimensional tensor indexing not yet fully implemented",
            ));
        }

        let mut result_data = Vec::new();
        for &idx in &index_data {
            if idx >= self.numel() {
                return Err(CoreError::index_out_of_bounds(
                    vec![idx],
                    self.shape().to_vec(),
                ));
            }

            if let Some(value) = self
                .storage()
                .get_f64(self.offset() + idx * self.strides()[0])
            {
                result_data.push(value as f32);
            }
        }

        Ok(Tensor::from_data(
            &result_data,
            vec![result_data.len()],
            Some(self.options().clone()),
        ))
    }

    fn masked_select(&self, mask: &Self) -> Result<Self> {
        AdvancedIndexing::masked_select_tensor(self, mask)
    }

    fn gather(&self, dim: usize, indices: &Self::Index) -> Result<Self> {
        AdvancedIndexing::gather(self, dim, indices)
    }

    fn scatter(&mut self, _dim: usize, _indices: &Self::Index, _values: &Self) -> Result<()> {
        Err(CoreError::invalid_op(
            "scatter",
            "mutable operations not yet implemented",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustytorch_core::Indexable;

    fn create_test_tensor_2d() -> Tensor {
        Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None)
    }

    fn create_test_tensor_1d() -> Tensor {
        Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], None)
    }

    #[test]
    fn test_indexable_get() {
        let tensor = create_test_tensor_2d();

        // Test valid indices
        assert!((tensor.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((tensor.get(&[0, 1]).unwrap() - 2.0).abs() < 1e-6);
        assert!((tensor.get(&[1, 2]).unwrap() - 6.0).abs() < 1e-6);

        // Test invalid indices
        assert!(tensor.get(&[2, 0]).is_err()); // Out of bounds
        assert!(tensor.get(&[0]).is_err()); // Wrong number of indices
    }

    #[test]
    fn test_fancy_indexing() {
        let tensor = create_test_tensor_2d();

        // Select elements (0,1) and (1,0)
        let row_indices = vec![0, 1];
        let col_indices = vec![1, 0];
        let indices = vec![row_indices, col_indices];

        let result = AdvancedIndexing::fancy_index(&tensor, &indices).unwrap();
        assert_eq!(result.shape(), &[2]);

        // Should get elements at (0,1)=2.0 and (1,0)=4.0
        let result_data = result.storage().to_vec_f64();
        assert!((result_data[0] - 2.0).abs() < 1e-6);
        assert!((result_data[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_masked_select() {
        let tensor = create_test_tensor_1d();
        let mask = vec![true, false, true, false, true]; // Select 1st, 3rd, 5th elements

        let result = AdvancedIndexing::masked_select(&tensor, &mask).unwrap();
        assert_eq!(result.shape(), &[3]);

        let result_data = result.storage().to_vec_f64();
        assert_eq!(result_data, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_nonzero() {
        let tensor = Tensor::from_data(&[0.0f32, 1.0, 0.0, 3.0], vec![2, 2], None);

        let nonzero_indices = AdvancedIndexing::nonzero(&tensor).unwrap();
        assert_eq!(nonzero_indices.len(), 2); // Two non-zero elements
        assert_eq!(nonzero_indices[0], vec![0, 1]); // Position of 1.0
        assert_eq!(nonzero_indices[1], vec![1, 1]); // Position of 3.0
    }

    #[test]
    fn test_where_condition() {
        let x = create_test_tensor_1d();
        let y = Tensor::from_data(&[10.0f32, 20.0, 30.0, 40.0, 50.0], vec![5], None);
        let condition = vec![true, false, true, false, true];

        let result = AdvancedIndexing::where_condition(&condition, &x, &y).unwrap();
        let result_data = result.storage().to_vec_f64();

        // Should select from x where true, y where false
        assert_eq!(result_data, vec![1.0, 20.0, 3.0, 40.0, 5.0]);
    }

    #[test]
    fn test_tensor_indexing() {
        let tensor = create_test_tensor_1d();
        let indices = Tensor::from_data(&[0.0f32, 2.0, 4.0], vec![3], None);

        let result = tensor.index(&indices).unwrap();
        assert_eq!(result.shape(), &[3]);

        let result_data = result.storage().to_vec_f64();
        assert_eq!(result_data, vec![1.0, 3.0, 5.0]); // Elements at indices 0, 2, 4
    }

    #[test]
    fn test_indexable_trait() {
        let tensor = create_test_tensor_2d();

        // Test trait get method
        let value = Indexable::get(&tensor, &[0, 1]).unwrap();
        assert!((value - 2.0).abs() < 1e-6);

        let value2 = Indexable::get(&tensor, &[1, 2]).unwrap();
        assert!((value2 - 6.0).abs() < 1e-6);

        // Test slice functionality
        let ranges = vec![0..2, 1..3];
        let sliced = Indexable::slice(&tensor, &ranges).unwrap();
        assert_eq!(sliced.shape(), &[2, 2]);

        // Should contain elements [2,3,5,6] from original [1,2,3,4,5,6]
        let sliced_data = sliced.storage().to_vec_f64();
        assert!((sliced_data[0] - 2.0).abs() < 1e-6); // tensor[0,1]
        assert!((sliced_data[1] - 3.0).abs() < 1e-6); // tensor[0,2]
        assert!((sliced_data[2] - 5.0).abs() < 1e-6); // tensor[1,1]
        assert!((sliced_data[3] - 6.0).abs() < 1e-6); // tensor[1,2]
    }
}
