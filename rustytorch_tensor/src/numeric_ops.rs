// rustytorch_tensor/src/numeric_ops.rs

use crate::tensor_errors::TensorError;
use crate::Tensor;
use rustytorch_core::{CoreError, NumericOps, Result};

impl NumericOps for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Result<Self::Output> {
        // Convert TensorError to CoreError
        self.add_broadcast(&rhs)
            .map_err(|e| CoreError::invalid_op("add", &e.to_string()))
    }

    fn sub(self, rhs: Self) -> Result<Self::Output> {
        self.sub_broadcast(&rhs)
            .map_err(|e| CoreError::invalid_op("sub", &e.to_string()))
    }

    fn mul(self, rhs: Self) -> Result<Self::Output> {
        self.mul_broadcast(&rhs)
            .map_err(|e| CoreError::invalid_op("mul", &e.to_string()))
    }

    fn div(self, rhs: Self) -> Result<Self::Output> {
        self.div_broadcast(&rhs)
            .map_err(|e| CoreError::invalid_op("div", &e.to_string()))
    }

    fn neg(self) -> Result<Self::Output> {
        // Stub implementation
        Err(CoreError::invalid_op("neg", "not implemented yet"))
    }

    fn abs(self) -> Result<Self::Output> {
        // Stub implementation
        Err(CoreError::invalid_op("abs", "not implemented yet"))
    }

    fn pow(self, exponent: Self) -> Result<Self::Output> {
        self.pow_broadcast(&exponent)
            .map_err(|e| CoreError::invalid_op("pow", &e.to_string()))
    }

    fn rem(self, rhs: Self) -> Result<Self::Output> {
        // Stub implementation
        Err(CoreError::invalid_op("rem", "not implemented yet"))
    }
}

impl Tensor {
    pub fn add_ref(&self, rhs: &Self) -> std::result::Result<Self, TensorError> {
        self.add_broadcast(rhs)
    }

    pub fn sub_ref(&self, rhs: &Self) -> std::result::Result<Self, TensorError> {
        self.sub_broadcast(rhs)
    }

    pub fn mul_ref(&self, rhs: &Self) -> std::result::Result<Self, TensorError> {
        self.mul_broadcast(rhs)
    }

    pub fn div_ref(&self, rhs: &Self) -> std::result::Result<Self, TensorError> {
        self.div_broadcast(rhs)
    }

    pub fn pow_ref(&self, exponent: &Self) -> std::result::Result<Self, TensorError> {
        self.pow_broadcast(exponent)
    }
}
