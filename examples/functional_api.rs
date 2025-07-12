//! Example demonstrating the functional API (module F)

use rustytorch_autograd::{Variable, functional::F::*};
use rustytorch_tensor::Tensor;

fn main() {
    println!("=== RustyTorch Functional API Demo ===\n");
    
    // 1. Activation Functions
    println!("1. Activation Functions:");
    let x = Variable::from_tensor(
        Tensor::from_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], None),
        true
    );
    
    println!("   Input: {:?}", x.tensor().to_vec::<f64>().unwrap());
    
    let relu_out = relu(&x);
    println!("   ReLU: {:?}", relu_out.tensor().to_vec::<f64>().unwrap());
    
    let sigmoid_out = sigmoid(&x);
    println!("   Sigmoid: {:?}", sigmoid_out.tensor().to_vec::<f64>().unwrap());
    
    let tanh_out = tanh(&x);
    println!("   Tanh: {:?}", tanh_out.tensor().to_vec::<f64>().unwrap());
    
    let leaky_relu_out = leaky_relu(&x, 0.1);
    println!("   LeakyReLU(0.1): {:?}", leaky_relu_out.tensor().to_vec::<f64>().unwrap());
    
    println!();
    
    // 2. Softmax
    println!("2. Softmax:");
    let logits = Variable::from_tensor(
        Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], None),
        true
    );
    
    let probs = softmax(&logits, -1);
    let probs_values = probs.tensor().to_vec::<f64>().unwrap();
    println!("   Logits: {:?}", logits.tensor().to_vec::<f64>().unwrap());
    println!("   Softmax: {:?}", probs_values);
    println!("   Sum: {:.6}", probs_values.iter().sum::<f64>());
    
    println!();
    
    // 3. Loss Functions
    println!("3. Loss Functions:");
    let predictions = Variable::from_tensor(
        Tensor::from_data(&[0.9, 0.8, 0.7, 0.6], vec![4], None),
        true
    );
    let targets = Variable::from_tensor(
        Tensor::from_data(&[1.0, 1.0, 0.0, 0.0], vec![4], None),
        false
    );
    
    let mse = mse_loss(&predictions, &targets);
    println!("   MSE Loss: {:.6}", mse.tensor().to_vec::<f64>().unwrap()[0]);
    
    let l1 = l1_loss(&predictions, &targets);
    println!("   L1 Loss: {:.6}", l1.tensor().to_vec::<f64>().unwrap()[0]);
    
    let bce = binary_cross_entropy(&predictions, &targets, 1e-7);
    println!("   BCE Loss: {:.6}", bce.tensor().to_vec::<f64>().unwrap()[0]);
    
    println!();
    
    // 4. Advanced Activations
    println!("4. Advanced Activations:");
    let x_adv = Variable::from_tensor(
        Tensor::from_data(&[-1.0, 0.0, 1.0], vec![3], None),
        true
    );
    
    let gelu_out = gelu(&x_adv);
    println!("   GELU: {:?}", gelu_out.tensor().to_vec::<f64>().unwrap());
    
    let swish_out = swish(&x_adv);
    println!("   Swish: {:?}", swish_out.tensor().to_vec::<f64>().unwrap());
    
    let mish_out = mish(&x_adv);
    println!("   Mish: {:?}", mish_out.tensor().to_vec::<f64>().unwrap());
    
    println!();
    
    // 5. Gradient Flow
    println!("5. Gradient Flow Through Functional API:");
    let x_grad = Variable::from_tensor(
        Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None),
        true
    );
    
    // Chain of operations
    let y = relu(&x_grad);
    let z = sigmoid(&y);
    let loss = z.mean();
    
    println!("   Forward: x -> ReLU -> Sigmoid -> mean");
    println!("   x: {:?}", x_grad.tensor().to_vec::<f64>().unwrap());
    println!("   After ReLU: {:?}", y.tensor().to_vec::<f64>().unwrap());
    println!("   After Sigmoid: {:?}", z.tensor().to_vec::<f64>().unwrap());
    println!("   Loss: {:.6}", loss.tensor().to_vec::<f64>().unwrap()[0]);
    
    // Backward pass
    loss.backward(None, false, false).unwrap();
    
    if let Some(grad) = x_grad.grad() {
        println!("   Gradient at x: {:?}", grad.to_vec::<f64>().unwrap());
    }
    
    println!();
    
    // 6. Normalization (basic example)
    println!("6. Layer Normalization:");
    let x_norm = Variable::from_tensor(
        Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None),
        true
    );
    
    let normalized = layer_norm(&x_norm, &[3], None, None, 1e-5);
    println!("   Input shape: {:?}", x_norm.shape());
    println!("   Output shape: {:?}", normalized.shape());
    println!("   Normalized values: {:?}", normalized.tensor().to_vec::<f64>().unwrap());
    
    println!();
    
    // 7. Dropout (demonstration)
    println!("7. Dropout:");
    let x_dropout = Variable::from_tensor(
        Tensor::ones(vec![10], None),
        true
    );
    
    let dropout_train = dropout(&x_dropout, 0.5, true);
    let dropout_eval = dropout(&x_dropout, 0.5, false);
    
    println!("   Training mode (p=0.5): {:?}", dropout_train.tensor().to_vec::<f64>().unwrap());
    println!("   Eval mode (p=0.5): {:?}", dropout_eval.tensor().to_vec::<f64>().unwrap());
    
    println!("\n=== Demo Complete ===");
}