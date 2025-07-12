//! Module F - API fonctionnelle simplifiée et fonctionnelle

use crate::{Variable, Operation, GRAD_ENABLED};
use rustytorch_tensor::Tensor;
use rustytorch_core::{DType, TensorOptions, NumericOps};

/// Module contenant les fonctions de l'API fonctionnelle
pub mod F {
    use super::*;
    use rustytorch_core::NumericOps;
    
    // ====== ACTIVATIONS ======
    
    /// Fonction ReLU (Rectified Linear Unit)
    pub fn relu(x: &Variable) -> Variable {
        let tensor = x.tensor();
        let result_tensor = tensor.relu().expect("ReLU operation failed");
        
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) || !x.requires_grad() {
            return Variable::from_tensor(result_tensor, false);
        }
        
        let x_clone = x.clone();
        let grad_fn = Some(Box::new(move |grad_output: &Tensor| {
            // Gradient: d/dx ReLU(x) = 1 if x > 0, 0 otherwise
            let x_tensor = x_clone.tensor();
            let zero_options = Some(TensorOptions::new().dtype(x_tensor.dtype()));
            let zero = Tensor::zeros(x_tensor.shape().to_vec(), zero_options);
            let mask = x_tensor.gt(&zero).unwrap();
            
            // Convert boolean mask to float for multiplication
            let mask_float = mask.to_dtype(grad_output.dtype()).unwrap();
            let grad = grad_output.mul(mask_float).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>);
        
        Variable::from_operation(
            result_tensor,
            Operation::Relu,
            vec![x.clone()],
            grad_fn,
        )
    }
    
    /// Fonction Sigmoid
    pub fn sigmoid(x: &Variable) -> Variable {
        let tensor = x.tensor();
        let result_tensor = tensor.sigmoid().expect("Sigmoid operation failed");
        
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) || !x.requires_grad() {
            return Variable::from_tensor(result_tensor, false);
        }
        
        let result_clone = result_tensor.clone();
        let grad_fn = Some(Box::new(move |grad_output: &Tensor| {
            // Gradient: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            let one_options = Some(TensorOptions::new().dtype(result_clone.dtype()));
            let one = Tensor::ones(result_clone.shape().to_vec(), one_options);
            let one_minus_sigmoid = one.sub(result_clone.clone()).unwrap();
            let grad = grad_output.mul(result_clone.clone()).unwrap()
                .mul(one_minus_sigmoid).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>);
        
        Variable::from_operation(
            result_tensor,
            Operation::Sigmoid,
            vec![x.clone()],
            grad_fn,
        )
    }
    
    /// Fonction Tanh
    pub fn tanh(x: &Variable) -> Variable {
        let tensor = x.tensor();
        let result_tensor = tensor.tanh().expect("Tanh operation failed");
        
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) || !x.requires_grad() {
            return Variable::from_tensor(result_tensor, false);
        }
        
        let result_clone = result_tensor.clone();
        let grad_fn = Some(Box::new(move |grad_output: &Tensor| {
            // Gradient: d/dx tanh(x) = 1 - tanh(x)^2
            let one_options = Some(TensorOptions::new().dtype(result_clone.dtype()));
            let one = Tensor::ones(result_clone.shape().to_vec(), one_options);
            let tanh_squared = result_clone.mul(result_clone.clone()).unwrap();
            let grad = grad_output.mul(one.sub(tanh_squared).unwrap()).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>);
        
        Variable::from_operation(
            result_tensor,
            Operation::Tanh,
            vec![x.clone()],
            grad_fn,
        )
    }
    
    /// Fonction Softmax le long d'une dimension
    pub fn softmax(x: &Variable, dim: i32) -> Variable {
        let tensor = x.tensor();
        let dim_usize = if dim < 0 {
            (tensor.shape().len() as i32 + dim) as usize
        } else {
            dim as usize
        };
        
        let result_tensor = tensor.softmax(Some(dim_usize)).expect("Softmax operation failed");
        
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) || !x.requires_grad() {
            return Variable::from_tensor(result_tensor, false);
        }
        
        let result_clone = result_tensor.clone();
        let grad_fn = Some(Box::new(move |grad_output: &Tensor| {
            // Gradient: d/dx softmax(x) = softmax(x) * (grad_output - sum(grad_output * softmax(x)))
            let sum_grad_softmax = grad_output.mul(result_clone.clone()).unwrap()
                .sum_dim(Some(dim_usize)).unwrap();
            let grad = result_clone.mul(grad_output.sub(sum_grad_softmax).unwrap()).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>);
        
        Variable::from_operation(
            result_tensor,
            Operation::Softmax,
            vec![x.clone()],
            grad_fn,
        )
    }
    
    /// Fonction Leaky ReLU
    pub fn leaky_relu(x: &Variable, negative_slope: f64) -> Variable {
        let tensor = x.tensor();
        let zero_options = Some(TensorOptions::new().dtype(tensor.dtype()));
        let zero = Tensor::zeros(tensor.shape().to_vec(), zero_options);
        let mask = tensor.gt(&zero).unwrap();
        
        // Compute leaky_relu: x if x > 0, negative_slope * x otherwise
        let neg_slope_tensor = Tensor::full(tensor.shape().to_vec(), negative_slope, tensor.dtype()).unwrap();
        let negative_part = tensor.mul(neg_slope_tensor).unwrap();
        let positive_part = tensor.clone();
        
        // result = mask * positive_part + (1 - mask) * negative_part
        let one_options = Some(TensorOptions::new().dtype(mask.dtype()));
        let one = Tensor::ones(mask.shape().to_vec(), one_options);
        let inv_mask = one.sub(mask.clone()).unwrap();
        
        // Convert masks to tensor dtype for arithmetic
        let mask_float = mask.to_dtype(tensor.dtype()).unwrap();
        let inv_mask_float = inv_mask.to_dtype(tensor.dtype()).unwrap();
        
        let result_tensor = mask_float.mul(positive_part).unwrap()
            .add(inv_mask_float.mul(negative_part).unwrap()).unwrap();
        
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) || !x.requires_grad() {
            return Variable::from_tensor(result_tensor, false);
        }
        
        let x_clone = x.clone();
        let grad_fn = Some(Box::new(move |grad_output: &Tensor| {
            // Gradient: 1 if x > 0, negative_slope otherwise
            let x_tensor = x_clone.tensor();
            let zero_options = Some(TensorOptions::new().dtype(x_tensor.dtype()));
            let zero = Tensor::zeros(x_tensor.shape().to_vec(), zero_options);
            let mask = x_tensor.gt(&zero).unwrap();
            
            let one_options = Some(TensorOptions::new().dtype(mask.dtype()));
            let one = Tensor::ones(mask.shape().to_vec(), one_options);
            let neg_slope_tensor = Tensor::full(mask.shape().to_vec(), negative_slope, mask.dtype()).unwrap();
            let inv_mask = one.sub(mask.clone()).unwrap();
            
            let grad_mask = mask.to_dtype(x_tensor.dtype()).unwrap()
                .add(inv_mask.to_dtype(x_tensor.dtype()).unwrap().mul(neg_slope_tensor.to_dtype(x_tensor.dtype()).unwrap()).unwrap()).unwrap();
            let grad = grad_output.mul(grad_mask).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>);
        
        Variable::from_operation(
            result_tensor,
            Operation::Relu, // Using Relu operation tag
            vec![x.clone()],
            grad_fn,
        )
    }
    
    // ====== LOSS FUNCTIONS ======
    
    /// Mean Squared Error Loss
    pub fn mse_loss(input: &Variable, target: &Variable) -> Variable {
        let diff = input.sub(target);
        let squared = diff.mul(&diff);
        squared.mean()
    }
    
    /// L1 Loss (Mean Absolute Error)  
    pub fn l1_loss(input: &Variable, target: &Variable) -> Variable {
        let diff = input.sub(target);
        let abs_diff = diff.abs();
        abs_diff.mean()
    }
    
    /// Binary Cross Entropy Loss (simplified version)
    pub fn binary_cross_entropy(input: &Variable, target: &Variable, eps: f64) -> Variable {
        // Clamp input to avoid log(0)
        let clamped_input = input.clamp(eps, 1.0 - eps);
        
        // target * log(input)
        let pos_term = target.mul(&clamped_input.log());
        
        // (1 - target) * log(1 - input)
        let one_minus_target = target.neg().add_scalar(1.0);
        let one_minus_input = clamped_input.neg().add_scalar(1.0);
        let neg_term = one_minus_target.mul(&one_minus_input.log());
        
        // -(target * log(input) + (1 - target) * log(1 - input))
        let sum = pos_term.add(&neg_term);
        sum.neg().mean()
    }
    
    // ====== NORMALIZATION (Basic) ======
    
    /// Simple Layer Normalization (basic implementation)
    pub fn layer_norm(
        x: &Variable,
        normalized_shape: &[usize],
        weight: Option<&Variable>,
        bias: Option<&Variable>,
        eps: f64
    ) -> Variable {
        // Calculate mean and variance over the last dimensions
        let mean = x.mean();
        let x_centered = x.sub(&mean);
        let var = x_centered.mul(&x_centered).mean();
        
        // Normalize
        let eps_var = Variable::from_tensor(
            Tensor::full(vec![1], eps, x.tensor().dtype()).unwrap(),
            false
        );
        let std = var.add(&eps_var).sqrt();
        let x_normalized = x_centered.div(&std);
        
        // Scale and shift
        let output = match (weight, bias) {
            (Some(w), Some(b)) => x_normalized.mul(w).add(b),
            (Some(w), None) => x_normalized.mul(w),
            (None, Some(b)) => x_normalized.add(b),
            (None, None) => x_normalized,
        };
        
        output
    }
    
    // ====== REGULARIZATION ======
    
    /// Dropout (basic implementation)
    pub fn dropout(x: &Variable, p: f64, training: bool) -> Variable {
        if !training || p == 0.0 {
            return x.clone();
        }
        
        if p == 1.0 {
            return Variable::from_tensor(
                Tensor::zeros(x.shape(), Some(TensorOptions::new().dtype(x.tensor().dtype()))),
                x.requires_grad()
            );
        }
        
        // For simplicity, just return the input scaled by (1-p) in training mode
        // A full implementation would use random masks
        x.mul_scalar(1.0 - p)
    }
}

// Helper methods for Variable
impl Variable {
    /// Helper method for scalar multiplication
    pub fn mul_scalar(&self, scalar: f64) -> Self {
        let result_tensor = self.tensor().mul_scalar(scalar).expect("Failed to multiply by scalar");
        
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) || !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }
        
        let grad_fn = Some(Box::new(move |grad_output: &Tensor| {
            let grad = grad_output.mul_scalar(scalar).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>);
        
        Self::from_operation(
            result_tensor,
            Operation::Mul,
            vec![self.clone()],
            grad_fn,
        )
    }
    
    /// Helper method for scalar addition
    pub fn add_scalar(&self, scalar: f64) -> Self {
        let result_tensor = self.tensor().add_scalar(scalar).expect("Failed to add scalar");
        
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) || !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }
        
        let grad_fn = Some(Box::new(move |grad_output: &Tensor| {
            // Gradient of addition is just the input gradient
            vec![grad_output.clone()]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>);
        
        Self::from_operation(
            result_tensor,
            Operation::Add,
            vec![self.clone()],
            grad_fn,
        )
    }
    
    /// Helper method for clamp
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        let tensor = self.tensor();
        let min_tensor = Tensor::full(tensor.shape().to_vec(), min, tensor.dtype()).unwrap();
        let max_tensor = Tensor::full(tensor.shape().to_vec(), max, tensor.dtype()).unwrap();
        
        // clamp(x, min, max) = max(min(x, max), min)
        let clamped_max = tensor.minimum(&max_tensor).unwrap();
        let result_tensor = clamped_max.maximum(&min_tensor).unwrap();
        
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) || !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }
        
        let x_clone = self.clone();
        let grad_fn = Some(Box::new(move |grad_output: &Tensor| {
            // Gradient passes through only where min < x < max
            let x_tensor = x_clone.tensor();
            let min_tensor = Tensor::full(x_tensor.shape().to_vec(), min, x_tensor.dtype()).unwrap();
            let max_tensor = Tensor::full(x_tensor.shape().to_vec(), max, x_tensor.dtype()).unwrap();
            
            let min_mask = x_tensor.gt(&min_tensor).unwrap();
            let max_mask = x_tensor.lt(&max_tensor).unwrap();
            
            // Both conditions must be true
            let mask_float = min_mask.to_dtype(x_tensor.dtype()).unwrap()
                .mul(max_mask.to_dtype(x_tensor.dtype()).unwrap()).unwrap();
            let grad = grad_output.mul(mask_float).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>);
        
        Self::from_operation(
            result_tensor,
            Operation::None,
            vec![self.clone()],
            grad_fn,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::F::*;
    
    #[test]
    fn test_relu_basic() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let tensor = Tensor::from_data(&data, vec![5], None);
        let x = Variable::from_tensor(tensor, true);
        
        let y = relu(&x);
        let y_data = y.tensor().to_vec::<f64>().unwrap();
        
        assert_eq!(y_data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_sigmoid_basic() {
        let data = vec![0.0];
        let tensor = Tensor::from_data(&data, vec![1], None);
        let x = Variable::from_tensor(tensor, true);
        
        let y = sigmoid(&x);
        let y_data = y.tensor().to_vec::<f64>().unwrap();
        
        // sigmoid(0) = 0.5
        assert!((y_data[0] - 0.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_mse_loss_basic() {
        let pred_data = vec![1.0, 2.0, 3.0, 4.0];
        let target_data = vec![1.1, 2.1, 2.9, 4.2];
        
        let pred = Variable::from_tensor(Tensor::from_data(&pred_data, vec![4], None), true);
        let target = Variable::from_tensor(Tensor::from_data(&target_data, vec![4], None), false);
        
        let loss = mse_loss(&pred, &target);
        let loss_value = loss.tensor().to_vec::<f64>().unwrap()[0];
        
        // MSE = ((0.1)^2 + (0.1)^2 + (0.1)^2 + (0.2)^2) / 4 = 0.0175
        assert!((loss_value - 0.0175).abs() < 1e-6);
    }
    
    #[test]
    fn test_scalar_operations() {
        let x = Variable::from_tensor(
            Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None),
            true
        );
        
        // Test scalar operations
        let y = x.mul_scalar(2.0);
        let y_values = y.tensor().to_vec::<f64>().unwrap();
        assert_eq!(y_values, vec![2.0, 4.0, 6.0]);
        
        let z = x.add_scalar(1.0);
        let z_values = z.tensor().to_vec::<f64>().unwrap();
        assert_eq!(z_values, vec![2.0, 3.0, 4.0]);
    }
}