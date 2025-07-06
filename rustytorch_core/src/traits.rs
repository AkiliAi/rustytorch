//! Core traits defining tensor operations and behaviors

use crate::errors::Result;
use std::ops::Range;

/// Trait for types supporting basic numeric operations
///
/// All operations are fallible to handle shape mismatches, type errors, etc.
pub trait NumericOps<Rhs = Self> {
    type Output;

    /// Element-wise addition
    fn add(self, rhs: Rhs) -> Result<Self::Output>;

    /// Element-wise subtraction
    fn sub(self, rhs: Rhs) -> Result<Self::Output>;

    /// Element-wise multiplication
    fn mul(self, rhs: Rhs) -> Result<Self::Output>;

    /// Element-wise division
    fn div(self, rhs: Rhs) -> Result<Self::Output>;

    /// Negation (unary minus)
    fn neg(self) -> Result<Self::Output>
    where
        Self: Sized;

    /// Absolute value
    fn abs(self) -> Result<Self::Output>
    where
        Self: Sized;

    /// Power operation
    fn pow(self, exponent: Rhs) -> Result<Self::Output>;

    /// Element-wise remainder
    fn rem(self, rhs: Rhs) -> Result<Self::Output>;
}

/// Trait for reduction operations
pub trait Reduction {
    type Output;
    type Axes;

    /// Sum of all elements
    fn sum(&self) -> Result<Self::Output>;

    /// Mean of all elements
    fn mean(&self) -> Result<Self::Output>;

    /// Maximum element
    fn max(&self) -> Result<Self::Output>;

    /// Minimum element  
    fn min(&self) -> Result<Self::Output>;

    /// Sum along specified axes
    fn sum_dim(&self, dim: Self::Axes, keep_dim: bool) -> Result<Self::Output>;

    /// Mean along specified axes
    fn mean_dim(&self, dim: Self::Axes, keep_dim: bool) -> Result<Self::Output>;

    /// Max along specified axes, returns (values, indices)
    fn max_dim(&self, dim: Self::Axes, keep_dim: bool) -> Result<(Self::Output, Self::Output)>;

    /// Min along specified axes, returns (values, indices)
    fn min_dim(&self, dim: Self::Axes, keep_dim: bool) -> Result<(Self::Output, Self::Output)>;

    /// Standard deviation
    fn std(&self, unbiased: bool) -> Result<Self::Output>;

    /// Variance
    fn var(&self, unbiased: bool) -> Result<Self::Output>;

    /// Standard deviation along axes
    fn std_dim(&self, dim: Self::Axes, unbiased: bool, keep_dim: bool) -> Result<Self::Output>;

    /// Variance along axes
    fn var_dim(&self, dim: Self::Axes, unbiased: bool, keep_dim: bool) -> Result<Self::Output>;

    /// Argmax - indices of maximum values
    fn argmax(&self, dim: Option<Self::Axes>, keep_dim: bool) -> Result<Self::Output>;

    /// Argmin - indices of minimum values
    fn argmin(&self, dim: Option<Self::Axes>, keep_dim: bool) -> Result<Self::Output>;
}

/// Trait for shape manipulation operations
pub trait Reshapable {
    /// Reshape tensor to new shape
    fn reshape(&self, shape: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Flatten tensor to 1D
    fn flatten(&self) -> Result<Self>
    where
        Self: Sized;

    /// Transpose two dimensions
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self>
    where
        Self: Sized;

    /// Permute dimensions according to the given order
    fn permute(&self, dims: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Remove dimensions of size 1
    fn squeeze(&self, dim: Option<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Add a dimension of size 1
    fn unsqueeze(&self, dim: usize) -> Result<Self>
    where
        Self: Sized;

    /// View tensor with new shape (without copying data)
    fn view(&self, shape: &[isize]) -> Result<Self>
    where
        Self: Sized;

    /// Broadcast to a specific shape
    fn broadcast_to(&self, shape: &[usize]) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for indexing and slicing operations
pub trait Indexable {
    type Output;
    type Index;

    /// Get element at specific indices
    fn get(&self, indices: &[usize]) -> Result<Self::Output>;

    /// Set element at specific indices
    fn set(&mut self, indices: &[usize], value: Self::Output) -> Result<()>;

    /// Slice tensor with ranges
    fn slice(&self, ranges: &[Range<usize>]) -> Result<Self>
    where
        Self: Sized;

    /// Advanced indexing with tensor indices
    fn index(&self, indices: &Self::Index) -> Result<Self>
    where
        Self: Sized;

    /// Masked selection
    fn masked_select(&self, mask: &Self) -> Result<Self>
    where
        Self: Sized;

    /// Gather values along an axis
    fn gather(&self, dim: usize, indices: &Self::Index) -> Result<Self>
    where
        Self: Sized;

    /// Scatter values along an axis
    fn scatter(&mut self, dim: usize, indices: &Self::Index, values: &Self) -> Result<()>
    where
        Self: Sized;
}

/// Trait for comparison operations
pub trait Comparable<Rhs = Self> {
    type Output;

    /// Element-wise equality
    fn eq(&self, other: &Rhs) -> Result<Self::Output>;

    /// Element-wise inequality
    fn ne(&self, other: &Rhs) -> Result<Self::Output>;

    /// Element-wise less than
    fn lt(&self, other: &Rhs) -> Result<Self::Output>;

    /// Element-wise less than or equal
    fn le(&self, other: &Rhs) -> Result<Self::Output>;

    /// Element-wise greater than
    fn gt(&self, other: &Rhs) -> Result<Self::Output>;

    /// Element-wise greater than or equal
    fn ge(&self, other: &Rhs) -> Result<Self::Output>;

    /// Check if all elements are true (for boolean tensors)
    fn all(&self) -> Result<bool>;

    /// Check if any element is true (for boolean tensors)
    fn any(&self) -> Result<bool>;
}

/// Trait for broadcasting behavior
pub trait Broadcasting {
    /// Check if shapes are broadcastable
    fn broadcastable_with(&self, other: &Self) -> bool;

    /// Get the broadcasted shape of two tensors
    fn broadcast_shape(&self, other: &Self) -> Result<Vec<usize>>;

    /// Apply broadcasting rules to align shapes
    fn broadcast_tensors(tensors: &[&Self]) -> Result<Vec<Self>>
    where
        Self: Sized;
}

/// Trait for automatic differentiation support
pub trait Differentiable {
    type Gradient;

    /// Compute gradients via backpropagation
    fn backward(
        &self,
        gradient: Option<Self::Gradient>,
        retain_graph: bool,
        create_graph: bool,
    ) -> Result<()>;

    /// Get accumulated gradient
    fn grad(&self) -> Option<&Self::Gradient>;

    /// Get mutable reference to gradient
    fn grad_mut(&mut self) -> Option<&mut Self::Gradient>;

    /// Check if gradient computation is enabled
    fn requires_grad(&self) -> bool;

    /// Enable or disable gradient computation
    fn set_requires_grad(&mut self, requires_grad: bool);

    /// Detach from computation graph
    fn detach(&self) -> Self
    where
        Self: Sized;

    /// Zero out gradients
    fn zero_grad(&mut self);

    /// Register a backward hook
    fn register_hook<F>(&mut self, hook: F)
    where
        F: Fn(&Self::Gradient) -> Self::Gradient + 'static;
}

/// Trait for serialization and deserialization
pub trait Serializable {
    /// Save to file
    fn save(&self, path: &str) -> Result<()>;

    /// Load from file
    fn load(path: &str) -> Result<Self>
    where
        Self: Sized;

    /// Save to a writer
    fn save_to<W: std::io::Write>(&self, writer: &mut W) -> Result<()>;

    /// Load from a reader
    fn load_from<R: std::io::Read>(reader: &mut R) -> Result<Self>
    where
        Self: Sized;

    /// Export to numpy-compatible format
    fn to_numpy(&self) -> Result<Vec<u8>>;

    /// Import from numpy-compatible format
    fn from_numpy(data: &[u8]) -> Result<Self>
    where
        Self: Sized;
}
