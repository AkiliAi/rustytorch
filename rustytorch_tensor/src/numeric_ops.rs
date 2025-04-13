// rustytorch_tensor/src/numeric_ops.rs

use rustytorch_core::NumericOps;
use crate::Tensor;
use crate::tensor_errors::TensorError;

impl NumericOps for Tensor {
    type Output = Result<Tensor, TensorError>;
    fn add(self, rhs: Self) -> Self::Output {
        // Utiliser add_broadcast mais avec la valeur, pas la référence
        self.add_broadcast(&rhs)
    }

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_broadcast(&rhs)
    }

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_broadcast(&rhs)
    }

    fn div(self, rhs: Self) -> Self::Output {
        self.div_broadcast(&rhs)
    }

}


impl Tensor {

    pub fn add_ref(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.add_broadcast(rhs)
    }

    pub fn sub_ref(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.sub_broadcast(rhs)
    }

    pub fn mul_ref(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.mul_broadcast(rhs)
    }

    pub fn div_ref(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.div_broadcast(rhs)
    }
}