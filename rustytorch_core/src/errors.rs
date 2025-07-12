//! Core error types for RustyTorch

use std::error::Error;
use std::fmt;

/// Core error type for RustyTorch operations
#[derive(Debug, Clone)]
pub enum CoreError {
    /// Shape mismatch between tensors
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
        operation: String,
    },

    /// Invalid dimension index
    DimensionOutOfBounds {
        dim: usize,
        ndim: usize,
        operation: String,
    },

    /// Index out of bounds
    IndexOutOfBounds {
        indices: Vec<usize>,
        shape: Vec<usize>,
    },

    /// Type mismatch between tensors
    TypeMismatch {
        expected: String,
        got: String,
        operation: String,
    },

    /// Invalid operation for the given input
    InvalidOperation { operation: String, reason: String },

    /// Operation not supported
    UnsupportedOperation {
        operation: String,
        dtype: Option<String>,
        device: Option<String>,
    },

    /// Broadcasting error
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
        reason: String,
    },

    /// Device mismatch between tensors
    DeviceMismatch {
        expected: String,
        got: String,
        operation: String,
    },

    /// Memory allocation error
    AllocationError { size: usize, reason: String },

    /// Numerical computation error
    NumericalError { operation: String, reason: String },

    /// IO error for serialization
    IoError { operation: String, source: String },

    /// Generic error with custom message
    Other(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::ShapeMismatch {
                expected,
                got,
                operation,
            } => {
                write!(
                    f,
                    "Shape mismatch in {}: expected {:?}, got {:?}",
                    operation, expected, got
                )
            }
            CoreError::DimensionOutOfBounds {
                dim,
                ndim,
                operation,
            } => {
                write!(
                    f,
                    "Dimension {} out of bounds for {}-dimensional tensor in {}",
                    dim, ndim, operation
                )
            }
            CoreError::IndexOutOfBounds { indices, shape } => {
                write!(
                    f,
                    "Index {:?} out of bounds for tensor with shape {:?}",
                    indices, shape
                )
            }
            CoreError::TypeMismatch {
                expected,
                got,
                operation,
            } => {
                write!(
                    f,
                    "Type mismatch in {}: expected {}, got {}",
                    operation, expected, got
                )
            }
            CoreError::InvalidOperation { operation, reason } => {
                write!(f, "Invalid operation {}: {}", operation, reason)
            }
            CoreError::UnsupportedOperation {
                operation,
                dtype,
                device,
            } => {
                let mut msg = format!("Unsupported operation: {}", operation);
                if let Some(dt) = dtype {
                    msg.push_str(&format!(" for dtype {}", dt));
                }
                if let Some(dev) = device {
                    msg.push_str(&format!(" on device {}", dev));
                }
                write!(f, "{}", msg)
            }
            CoreError::BroadcastError {
                shape1,
                shape2,
                reason,
            } => {
                write!(
                    f,
                    "Cannot broadcast tensors with shapes {:?} and {:?}: {}",
                    shape1, shape2, reason
                )
            }
            CoreError::DeviceMismatch {
                expected,
                got,
                operation,
            } => {
                write!(
                    f,
                    "Device mismatch in {}: expected {}, got {}",
                    operation, expected, got
                )
            }
            CoreError::AllocationError { size, reason } => {
                write!(f, "Failed to allocate {} bytes: {}", size, reason)
            }
            CoreError::NumericalError { operation, reason } => {
                write!(f, "Numerical error in {}: {}", operation, reason)
            }
            CoreError::IoError { operation, source } => {
                write!(f, "IO error during {}: {}", operation, source)
            }
            CoreError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for CoreError {}

/// Type alias for Results with CoreError
pub type Result<T> = std::result::Result<T, CoreError>;

/// Helper functions for creating common errors
impl CoreError {
    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: Vec<usize>, got: Vec<usize>, operation: &str) -> Self {
        CoreError::ShapeMismatch {
            expected,
            got,
            operation: operation.to_string(),
        }
    }

    /// Create a dimension out of bounds error
    pub fn dim_out_of_bounds(dim: usize, ndim: usize, operation: &str) -> Self {
        CoreError::DimensionOutOfBounds {
            dim,
            ndim,
            operation: operation.to_string(),
        }
    }

    /// Create an invalid operation error
    pub fn invalid_op(operation: &str, reason: &str) -> Self {
        CoreError::InvalidOperation {
            operation: operation.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Create a broadcasting error
    pub fn broadcast_error(shape1: Vec<usize>, shape2: Vec<usize>, reason: &str) -> Self {
        CoreError::BroadcastError {
            shape1,
            shape2,
            reason: reason.to_string(),
        }
    }

    /// Create an index out of bounds error
    pub fn index_out_of_bounds(indices: Vec<usize>, shape: Vec<usize>) -> Self {
        CoreError::IndexOutOfBounds { indices, shape }
    }

    /// Create a memory error
    pub fn memory_error(reason: &str) -> Self {
        CoreError::AllocationError {
            size: 0,
            reason: reason.to_string(),
        }
    }
}

/// Trait for converting other error types to CoreError
pub trait IntoCoreError {
    fn into_core_error(self, context: &str) -> CoreError;
}

impl IntoCoreError for std::io::Error {
    fn into_core_error(self, context: &str) -> CoreError {
        CoreError::IoError {
            operation: context.to_string(),
            source: self.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoreError::shape_mismatch(vec![2, 3], vec![3, 2], "matmul");
        assert_eq!(
            err.to_string(),
            "Shape mismatch in matmul: expected [2, 3], got [3, 2]"
        );
    }

    #[test]
    fn test_error_creation_helpers() {
        let err = CoreError::dim_out_of_bounds(3, 2, "transpose");
        assert!(matches!(err, CoreError::DimensionOutOfBounds { .. }));
    }
}
