//! Tensor view system for zero-copy tensor operations
//!
//! This module implements efficient tensor views that allow slicing, indexing,
//! and reshaping without copying the underlying data.

use crate::{storage::StorageType, Tensor};
use rustytorch_core::{CoreError, Result, TensorOptions};
use std::ops::Range;
use std::sync::Arc;

/// A view into a tensor that shares the underlying storage
/// but can have different shape, strides, and offset
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    /// Reference to the underlying storage
    storage: &'a Arc<StorageType>,

    /// Shape of this view
    shape: Vec<usize>,

    /// Strides for memory layout
    strides: Vec<usize>,

    /// Offset into the storage
    offset: usize,

    /// Tensor options (dtype, device, etc.)
    options: TensorOptions,

    /// Whether this view is contiguous in memory
    is_contiguous: bool,
}

impl<'a> TensorView<'a> {
    /// Create a new view from a tensor
    pub fn new(tensor: &'a Tensor) -> Self {
        Self {
            storage: tensor.storage_ref(),
            shape: tensor.shape().to_vec(),
            strides: tensor.strides().to_vec(),
            offset: tensor.offset(),
            options: tensor.options().clone(),
            is_contiguous: tensor.is_contiguous(),
        }
    }

    /// Create a view with custom parameters
    pub fn from_parts(
        storage: &'a Arc<StorageType>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
        options: TensorOptions,
    ) -> Result<Self> {
        // Validate that the view parameters are valid
        Self::validate_view_params(&shape, &strides, offset, storage.numel())?;

        let is_contiguous = Self::check_contiguous(&shape, &strides);

        Ok(Self {
            storage,
            shape,
            strides,
            offset,
            options,
            is_contiguous,
        })
    }

    /// Get the shape of this view
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of this view
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the offset of this view
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if this view is contiguous
    pub fn is_contiguous(&self) -> bool {
        self.is_contiguous
    }

    /// Get tensor options
    pub fn options(&self) -> &TensorOptions {
        &self.options
    }

    /// Get storage reference
    pub fn storage(&self) -> &Arc<StorageType> {
        self.storage
    }

    /// Slice the view along specified dimensions
    pub fn slice(&self, ranges: &[Range<usize>]) -> Result<TensorView<'a>> {
        if ranges.len() > self.ndim() {
            return Err(CoreError::invalid_op(
                "slice",
                &format!(
                    "Too many slice dimensions: {} > {}",
                    ranges.len(),
                    self.ndim()
                ),
            ));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        let mut new_offset = self.offset;

        // Apply slicing to each dimension
        for (dim, range) in ranges.iter().enumerate() {
            if range.end > self.shape[dim] {
                return Err(CoreError::invalid_op(
                    "slice",
                    &format!(
                        "Slice end {} > dimension size {}",
                        range.end, self.shape[dim]
                    ),
                ));
            }

            if range.start >= range.end {
                return Err(CoreError::invalid_op(
                    "slice",
                    &format!("Invalid slice range: {}..{}", range.start, range.end),
                ));
            }

            // Update offset for this dimension
            new_offset += range.start * self.strides[dim];

            // Update shape for this dimension
            new_shape[dim] = range.end - range.start;
        }

        let is_contiguous = Self::check_contiguous(&new_shape, &new_strides);

        Ok(TensorView {
            storage: self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            options: self.options.clone(),
            is_contiguous,
        })
    }

    /// Select a single index along a dimension, reducing dimensionality
    pub fn select(&self, dim: usize, index: usize) -> Result<TensorView<'a>> {
        if dim >= self.ndim() {
            return Err(CoreError::dim_out_of_bounds(dim, self.ndim(), "select"));
        }

        if index >= self.shape[dim] {
            return Err(CoreError::invalid_op(
                "select",
                &format!("Index {} >= dimension size {}", index, self.shape[dim]),
            ));
        }

        // Update offset
        let new_offset = self.offset + index * self.strides[dim];

        // Remove the selected dimension
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(dim);
        new_strides.remove(dim);

        let is_contiguous = Self::check_contiguous(&new_shape, &new_strides);

        Ok(TensorView {
            storage: self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            options: self.options.clone(),
            is_contiguous,
        })
    }

    /// Narrow the view along a dimension
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<TensorView<'a>> {
        if dim >= self.ndim() {
            return Err(CoreError::dim_out_of_bounds(dim, self.ndim(), "narrow"));
        }

        if start + length > self.shape[dim] {
            return Err(CoreError::invalid_op(
                "narrow",
                &format!(
                    "Narrow range {}..{} exceeds dimension size {}",
                    start,
                    start + length,
                    self.shape[dim]
                ),
            ));
        }

        let mut new_shape = self.shape.clone();
        let new_offset = self.offset + start * self.strides[dim];
        new_shape[dim] = length;

        let is_contiguous = Self::check_contiguous(&new_shape, &self.strides);

        Ok(TensorView {
            storage: self.storage,
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
            options: self.options.clone(),
            is_contiguous,
        })
    }

    /// Reshape the view (only works if contiguous or compatible)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<'a>> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(CoreError::shape_mismatch(
                vec![self.numel()],
                vec![new_numel],
                "view_reshape",
            ));
        }

        // For now, only allow reshape if contiguous
        if !self.is_contiguous {
            return Err(CoreError::invalid_op(
                "view_reshape",
                "Reshape requires contiguous tensor view",
            ));
        }

        // Compute new strides for row-major layout
        let new_strides = Self::compute_contiguous_strides(new_shape);

        Ok(TensorView {
            storage: self.storage,
            shape: new_shape.to_vec(),
            strides: new_strides,
            offset: self.offset,
            options: self.options.clone(),
            is_contiguous: true,
        })
    }

    /// Transpose two dimensions
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<TensorView<'a>> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(CoreError::dim_out_of_bounds(
                dim0.max(dim1),
                self.ndim(),
                "view_transpose",
            ));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        let is_contiguous = Self::check_contiguous(&new_shape, &new_strides);

        Ok(TensorView {
            storage: self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            options: self.options.clone(),
            is_contiguous,
        })
    }

    /// Permute dimensions according to given order
    pub fn permute(&self, dims: &[usize]) -> Result<TensorView<'a>> {
        if dims.len() != self.ndim() {
            return Err(CoreError::invalid_op(
                "view_permute",
                &format!(
                    "Permutation length {} != tensor dimensions {}",
                    dims.len(),
                    self.ndim()
                ),
            ));
        }

        // Check that dims is a valid permutation
        let mut seen = vec![false; self.ndim()];
        for &dim in dims {
            if dim >= self.ndim() {
                return Err(CoreError::dim_out_of_bounds(
                    dim,
                    self.ndim(),
                    "view_permute",
                ));
            }
            if seen[dim] {
                return Err(CoreError::invalid_op(
                    "view_permute",
                    &format!("Duplicate dimension {} in permutation", dim),
                ));
            }
            seen[dim] = true;
        }

        let mut new_shape = vec![0; self.ndim()];
        let mut new_strides = vec![0; self.ndim()];

        for (new_dim, &old_dim) in dims.iter().enumerate() {
            new_shape[new_dim] = self.shape[old_dim];
            new_strides[new_dim] = self.strides[old_dim];
        }

        let is_contiguous = Self::check_contiguous(&new_shape, &new_strides);

        Ok(TensorView {
            storage: self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            options: self.options.clone(),
            is_contiguous,
        })
    }

    /// Squeeze dimensions of size 1
    pub fn squeeze(&self, dim: Option<usize>) -> Result<TensorView<'a>> {
        match dim {
            Some(d) => {
                if d >= self.ndim() {
                    return Err(CoreError::dim_out_of_bounds(d, self.ndim(), "view_squeeze"));
                }
                if self.shape[d] != 1 {
                    return Err(CoreError::invalid_op(
                        "view_squeeze",
                        &format!("Dimension {} has size {}, cannot squeeze", d, self.shape[d]),
                    ));
                }

                let mut new_shape = self.shape.clone();
                let mut new_strides = self.strides.clone();
                new_shape.remove(d);
                new_strides.remove(d);

                let is_contiguous = Self::check_contiguous(&new_shape, &new_strides);

                Ok(TensorView {
                    storage: self.storage,
                    shape: new_shape,
                    strides: new_strides,
                    offset: self.offset,
                    options: self.options.clone(),
                    is_contiguous,
                })
            }
            None => {
                // Squeeze all dimensions of size 1
                let mut new_shape = Vec::new();
                let mut new_strides = Vec::new();

                for (i, &size) in self.shape.iter().enumerate() {
                    if size != 1 {
                        new_shape.push(size);
                        new_strides.push(self.strides[i]);
                    }
                }

                let is_contiguous = Self::check_contiguous(&new_shape, &new_strides);

                Ok(TensorView {
                    storage: self.storage,
                    shape: new_shape,
                    strides: new_strides,
                    offset: self.offset,
                    options: self.options.clone(),
                    is_contiguous,
                })
            }
        }
    }

    /// Unsqueeze (add dimension of size 1)
    pub fn unsqueeze(&self, dim: usize) -> Result<TensorView<'a>> {
        if dim > self.ndim() {
            return Err(CoreError::invalid_op(
                "view_unsqueeze",
                &format!(
                    "Unsqueeze dimension {} > tensor dimensions {}",
                    dim,
                    self.ndim()
                ),
            ));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        // Insert dimension of size 1 at the specified position
        new_shape.insert(dim, 1);
        // For unsqueeze, the stride can be anything since the dimension has size 1
        // We'll use the next dimension's stride or 1 if it's the last dimension
        let stride = if dim < self.strides.len() {
            self.strides[dim]
        } else if !self.strides.is_empty() {
            self.strides[self.strides.len() - 1]
        } else {
            1
        };
        new_strides.insert(dim, stride);

        let is_contiguous = Self::check_contiguous(&new_shape, &new_strides);

        Ok(TensorView {
            storage: self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            options: self.options.clone(),
            is_contiguous,
        })
    }

    /// Convert this view to an owned tensor (copies data)
    pub fn to_tensor(&self) -> Result<Tensor> {
        // For now, we'll implement a simple copy-based conversion
        // In a full implementation, this would handle non-contiguous views properly

        if self.is_contiguous {
            // If contiguous, we can create a tensor that shares or copies the relevant slice
            self.contiguous_to_tensor()
        } else {
            // If not contiguous, we need to copy and reorder the data
            self.non_contiguous_to_tensor()
        }
    }

    /// Helper function to validate view parameters
    fn validate_view_params(
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        storage_size: usize,
    ) -> Result<()> {
        if shape.len() != strides.len() {
            return Err(CoreError::invalid_op(
                "view_validation",
                &format!(
                    "Shape length {} != strides length {}",
                    shape.len(),
                    strides.len()
                ),
            ));
        }

        // Check that the view doesn't exceed storage bounds
        if !shape.is_empty() {
            let mut max_index = offset;
            for (_i, (&size, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
                if size > 0 {
                    max_index = max_index.max(offset + (size - 1) * stride);
                }
            }

            if max_index >= storage_size {
                return Err(CoreError::invalid_op(
                    "view_validation",
                    &format!(
                        "View extends beyond storage: max_index {} >= storage_size {}",
                        max_index, storage_size
                    ),
                ));
            }
        }

        Ok(())
    }

    /// Check if a shape and strides represent a contiguous layout
    fn check_contiguous(shape: &[usize], strides: &[usize]) -> bool {
        if shape.is_empty() {
            return true;
        }

        let expected_strides = Self::compute_contiguous_strides(shape);
        strides == expected_strides
    }

    /// Compute strides for a contiguous row-major layout
    fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Convert contiguous view to tensor
    fn contiguous_to_tensor(&self) -> Result<Tensor> {
        // This is a simplified implementation
        // In practice, you'd want to slice the storage and create a new tensor
        Err(CoreError::invalid_op(
            "contiguous_to_tensor",
            "not implemented yet",
        ))
    }

    /// Convert non-contiguous view to tensor by copying data
    fn non_contiguous_to_tensor(&self) -> Result<Tensor> {
        // This would involve iterating through the view's logical indices
        // and copying data to create a contiguous tensor
        Err(CoreError::invalid_op(
            "non_contiguous_to_tensor",
            "not implemented yet",
        ))
    }
}

/// Iterator over the elements of a tensor view
pub struct TensorViewIterator<'a> {
    view: &'a TensorView<'a>,
    current_indices: Vec<usize>,
    finished: bool,
}

impl<'a> TensorViewIterator<'a> {
    pub fn new(view: &'a TensorView<'a>) -> Self {
        let finished = view.numel() == 0;
        Self {
            view,
            current_indices: vec![0; view.ndim()],
            finished,
        }
    }

    /// Get the current linear index in storage
    fn linear_index(&self) -> usize {
        let mut index = self.view.offset;
        for (i, &idx) in self.current_indices.iter().enumerate() {
            index += idx * self.view.strides[i];
        }
        index
    }

    /// Advance to next index
    fn advance(&mut self) {
        if self.finished {
            return;
        }

        // Increment indices in row-major order
        let mut carry = 1;
        for i in (0..self.current_indices.len()).rev() {
            self.current_indices[i] += carry;
            if self.current_indices[i] < self.view.shape[i] {
                carry = 0;
                break;
            } else {
                self.current_indices[i] = 0;
                carry = 1;
            }
        }

        if carry == 1 {
            self.finished = true;
        }
    }
}

impl<'a> Iterator for TensorViewIterator<'a> {
    type Item = (Vec<usize>, usize); // (indices, linear_storage_index)

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let result = (self.current_indices.clone(), self.linear_index());
        self.advance();
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustytorch_core::TensorOptions;

    // Helper function to create a simple tensor for testing
    fn create_test_tensor() -> Tensor {
        Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None)
    }

    #[test]
    fn test_basic_view_creation() {
        let tensor = create_test_tensor();
        let view = TensorView::new(&tensor);

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.numel(), 6);
        assert_eq!(view.ndim(), 2);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_view_slice() {
        let tensor = create_test_tensor();
        let view = TensorView::new(&tensor);

        // Slice first row
        let sliced = view.slice(&[0..1, 0..3]).unwrap();
        assert_eq!(sliced.shape(), &[1, 3]);
        assert_eq!(sliced.numel(), 3);

        // Slice subset
        let subset = view.slice(&[0..2, 1..3]).unwrap();
        assert_eq!(subset.shape(), &[2, 2]);
        assert_eq!(subset.numel(), 4);
    }

    #[test]
    fn test_view_select() {
        let tensor = create_test_tensor();
        let view = TensorView::new(&tensor);

        // Select first row
        let selected = view.select(0, 0).unwrap();
        assert_eq!(selected.shape(), &[3]);
        assert_eq!(selected.numel(), 3);
        assert_eq!(selected.ndim(), 1);
    }

    #[test]
    fn test_view_transpose() {
        let tensor = create_test_tensor();
        let view = TensorView::new(&tensor);

        let transposed = view.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.numel(), 6);
        // After transpose, it's typically not contiguous
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_view_squeeze_unsqueeze() {
        let tensor = create_test_tensor();
        let view = TensorView::new(&tensor);

        // Add dimension
        let unsqueezed = view.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 1, 3]);

        // Remove it back
        let squeezed = unsqueezed.squeeze(Some(1)).unwrap();
        assert_eq!(squeezed.shape(), &[2, 3]);
    }

    #[test]
    fn test_view_iterator() {
        // Create a small 2x2 tensor for easy testing
        let tensor = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None);
        let view = TensorView::new(&tensor);

        let indices: Vec<_> = TensorViewIterator::new(&view)
            .map(|(indices, _)| indices)
            .collect();

        assert_eq!(
            indices,
            vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
        );
    }
}
