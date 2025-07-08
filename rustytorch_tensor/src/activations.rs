// rustytorch_tensor/src/activations.rs

use crate::Tensor;

// use rustytorch_core::NumericOps;

impl Tensor {
    // /// Applique la fonction d'activation ReLU (Rectified Linear Unit)
    // /// ReLU(x) = max(0, x)
    // pub fn relu(&self) -> Result<Self, TensorError> {
    //     self.apply_unary_op(
    //         |x| if x > 0.0 { x } else { 0.0 },
    //         |x| if x > 0.0 { x } else { 0.0 }
    //     )
    // }
    //
    // /// Calcule le gradient de ReLU par rapport à l'entrée
    // /// d(ReLU(x))/dx = 1 si x > 0, 0 sinon
    // pub fn relu_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
    //     let zeros = Self::zeros(self.shape().to_vec(), Some(self.options.clone()));
    //     let mask = self.gt(&zeros)?;
    //     let mask_f64 = mask.to_f64()?;
    //     mask_f64.mul(grad_output)
    // }
    //
    // /// Applique la fonction d'activation Sigmoid
    // /// Sigmoid(x) = 1 / (1 + exp(-x))
    // pub fn sigmoid(&self) -> Result<Self, TensorError> {
    //     self.apply_unary_op(
    //         |x| 1.0 / (1.0 + (-x).exp()),
    //         |x| 1.0 / (1.0 + (-x).exp())
    //     )
    // }
    //
    // /// Calcule le gradient de la fonction Sigmoid
    // /// d(Sigmoid(x))/dx = Sigmoid(x) * (1 - Sigmoid(x))
    // pub fn sigmoid_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
    //     let sigmoid_x = self.sigmoid()?;
    //     let one = Self::ones(sigmoid_x.shape().to_vec(), Some(self.options.clone()));
    //     let one_minus_sigmoid = one.sub(&sigmoid_x)?;
    //     let grad = sigmoid_x.mul(&one_minus_sigmoid)?;
    //     grad.mul(grad_output)
    // }
    //
    // /// Applique la fonction d'activation Tanh
    // /// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    // pub fn tanh(&self) -> Result<Self, TensorError> {
    //     self.apply_unary_op(
    //         |x| x.tanh(),
    //         |x| x.tanh()
    //     )
    // }
    //
    // /// Calcule le gradient de la fonction Tanh
    // /// d(tanh(x))/dx = 1 - tanh(x)^2
    // pub fn tanh_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
    //     let tanh_x = self.tanh()?;
    //     let tanh_squared = tanh_x.mul(&tanh_x)?;
    //     let one = Self::ones(tanh_x.shape().to_vec(), Some(self.options.clone()));
    //     let grad = one.sub(&tanh_squared)?;
    //     grad.mul(grad_output.clone())
    // }

    // /// Applique une opération unaire optimisée élément par élément sur le tenseur
    // pub fn apply_unary_op<F32Op, F64Op>(&self, f32_op: F32Op, f64_op: F64Op) -> Result<Self, TensorError>
    // where
    //     F32Op: Fn(f32) -> f32 + Sync + Send,
    //     F64Op: Fn(f64) -> f64 + Sync + Send,
    // {
    //     let mut result = self.clone();
    //
    //     match self.storage.as_ref() {
    //         StorageType::F32(data) => {
    //             let mut result_data = vec![0.0; data.len()];
    //
    //             // Utiliser une boucle séquentielle
    //             for (res, &val) in result_data.iter_mut().zip(data.iter()) {
    //                 *res = f32_op(val);
    //             }
    //
    //             result.storage = Arc::new(StorageType::from_f32(&result_data));
    //         },
    //         StorageType::F64(data) => {
    //             let mut result_data = vec![0.0; data.len()];
    //
    //             for (res, &val) in result_data.iter_mut().zip(data.iter()) {
    //                 *res = f64_op(val);
    //             }
    //
    //             result.storage = Arc::new(StorageType::from_f64(&result_data));
    //         },
    //         _ => return Err(TensorError::new(
    //             TensorErrorType::UnsupportedOperation,
    //             "Unsupported storage type for unary operation",
    //         )),
    //     }
    //
    //     Ok(result)
    // }
}
