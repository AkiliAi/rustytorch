//rustytorch_nn/src/nn_errors.rs

use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

// Erreur du module de réseau de neurones
#[derive(Debug)]
pub enum NNError {
    ShapeMismatch(String),
    ParameterError(String),
    ForwardError(String),
    BackwardError(String),
    InitializationError(String),
    DeviceError(String),
}

impl Display for NNError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            NNError::ShapeMismatch(msg) => write!(f, "Shape Mismatch: {}", msg),
            NNError::ParameterError(msg) => write!(f, "Parameter Error: {}", msg),
            NNError::ForwardError(msg) => write!(f, "Forward Error: {}", msg),
            NNError::BackwardError(msg) => write!(f, "Backward Error: {}", msg),
            NNError::InitializationError(msg) => write!(f, "Initialization Error: {}", msg),
            NNError::DeviceError(msg) => write!(f, "Device Error: {}", msg),
        }
    }
}

impl Error for NNError {}
