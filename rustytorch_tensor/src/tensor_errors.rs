// rustytorch_tensor/src/tensor_errors.rs


use std::fmt;
use std::fmt::{Display, Formatter};



#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorError{
    pub error : TensorErrorType,
    pub message: String,
}

#[derive(Debug,Clone,PartialEq,Eq)]
pub enum TensorErrorType{
    ShapeMismatch,
    IndexOutOfBounds,
    InvalidOperation,
    InvalidType,
    DeviceMismatch,
    MemoryAllocationError,
    UnsupportedOperation,
    StorageError,
    DeviceError,
    TypeError,
    BroadcastingError,
    Other,

}


impl Display for TensorError{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "TensorError: {:?} - {}", self.error, self.message)
    }
}


impl Display for TensorErrorType{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            TensorErrorType::ShapeMismatch => write!(f, "Shape Mismatch"),
            TensorErrorType::IndexOutOfBounds => write!(f, "Index Out Of Bounds"),
            TensorErrorType::InvalidOperation => write!(f, "Invalid Operation"),
            TensorErrorType::InvalidType => write!(f, "Invalid Type"),
            TensorErrorType::DeviceMismatch => write!(f, "Device Mismatch"),
            TensorErrorType::MemoryAllocationError => write!(f, "Memory Allocation Error"),
            TensorErrorType::UnsupportedOperation => write!(f, "Unsupported Operation"),
            TensorErrorType::StorageError => write!(f, "Storage Error"),
            TensorErrorType::DeviceError => write!(f, "Device Error"),
            TensorErrorType::TypeError => write!(f, "Type Error"),
            TensorErrorType::BroadcastingError => write!(f, "Broadcasting Error"),
            TensorErrorType::Other => write!(f, "Other Error"),
        }
    }
}


impl TensorError{
    /// Crée une nouvelle erreur de tenseur
    pub fn new(error:TensorErrorType,message:&str) -> Self{
        let message = match &error{
            TensorErrorType::ShapeMismatch => format!("Shape mismatch: {}", message),
            TensorErrorType::IndexOutOfBounds => format!("Index out of bounds: {}", message),
            TensorErrorType::InvalidOperation => format!("Invalid operation: {}", message),
            TensorErrorType::InvalidType => format!("Invalid type: {}", message),
            TensorErrorType::DeviceMismatch => format!("Device mismatch: {}", message),
            TensorErrorType::MemoryAllocationError => format!("Memory allocation error: {}", message),
            TensorErrorType::UnsupportedOperation => format!("Unsupported operation: {}", message),
            TensorErrorType::StorageError => format!("Storage error: {}", message),
            TensorErrorType::DeviceError => format!("Device error: {}", message),
            TensorErrorType::TypeError => format!("Type error: {}", message),
            TensorErrorType::BroadcastingError => format!("Broadcasting error: {}", message),
            TensorErrorType::Other => format!("Other error: {}", message),
        };
        TensorError{
            error,
            message,
        }
    }
}