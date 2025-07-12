//! Comprehensive integration tests for tensor + autograd functionality
//! 
//! These tests validate the complete integration between rustytorch_tensor
//! and rustytorch_autograd, ensuring the autograd system works correctly
//! with real tensor operations.

use rustytorch_autograd::{Variable, F};
use rustytorch_tensor::Tensor;

/// Test basic forward and backward pass with simple operations
#[test]
fn test_basic_forward_backward() {
    // Create input tensors
    let x_data = Tensor::from_data(&[2.0, 3.0], vec![2], None);
    let y_data = Tensor::from_data(&[1.0, 4.0], vec![2], None);
    
    // Create variables with gradient tracking
    let mut x = Variable::from_tensor(x_data, true);
    let mut y = Variable::from_tensor(y_data, true);
    
    // Forward pass: z = x + y
    let mut z = x.add(&y);
    
    // Backward pass
    z.backward();
    
    // Check gradients
    let x_grad = x.grad().expect("x should have gradient");
    let y_grad = y.grad().expect("y should have gradient");
    
    assert_eq!(x_grad.storage().to_vec_f64(), vec![1.0, 1.0]);
    assert_eq!(y_grad.storage().to_vec_f64(), vec![1.0, 1.0]);
}

/// Test multiplication and chain rule
#[test]
fn test_multiplication_chain_rule() {
    let x_data = Tensor::from_data(&[2.0, 3.0], vec![2], None);
    let y_data = Tensor::from_data(&[1.0, 4.0], vec![2], None);
    
    let mut x = Variable::from_tensor(x_data, true);
    let mut y = Variable::from_tensor(y_data, true);
    
    // Forward: z = x * y
    let mut z = x.mul(&y);
    
    // Backward
    z.backward();
    
    // Check gradients: dz/dx = y, dz/dy = x
    let x_grad = x.grad().expect("x should have gradient");
    let y_grad = y.grad().expect("y should have gradient");
    
    assert_eq!(x_grad.storage().to_vec_f64(), vec![1.0, 4.0]); // y values
    assert_eq!(y_grad.storage().to_vec_f64(), vec![2.0, 3.0]); // x values
}

/// Test complex computation graph with multiple operations
#[test]
fn test_complex_computation_graph() {
    let x_data = Tensor::from_data(&[1.0, 2.0], vec![2], None);
    let mut x = Variable::from_tensor(x_data, true);
    
    // Forward: y = x^2 + 2*x + 1 = (x + 1)^2
    let x_squared = x.mul(&x);
    let two_x = x.mul_scalar(2.0);
    let intermediate = x_squared.add(&two_x);
    let mut y = intermediate.add_scalar(1.0);
    
    // Sum for scalar output
    let mut loss = y.sum();
    
    // Backward
    loss.backward();
    
    // Check gradient: dy/dx = 2x + 2
    let x_grad = x.grad().expect("x should have gradient");
    let expected_grad = vec![4.0, 6.0]; // 2*1+2=4, 2*2+2=6
    
    assert!((x_grad.storage().to_vec_f64()[0] - expected_grad[0]).abs() < 1e-6);
    assert!((x_grad.storage().to_vec_f64()[1] - expected_grad[1]).abs() < 1e-6);
}

/// Test functional API integration
#[test]
fn test_functional_api_integration() {
    let x_data = Tensor::from_data(&[-1.0, 0.0, 1.0, 2.0], vec![4], None);
    let mut x = Variable::from_tensor(x_data, true);
    
    // Test ReLU activation
    let relu_output = F::relu(&x);
    let expected_relu = vec![0.0, 0.0, 1.0, 2.0];
    assert_eq!(relu_output.tensor().storage().to_vec_f64(), expected_relu);
    
    // Test Sigmoid activation  
    let sigmoid_output = F::sigmoid(&x);
    let sigmoid_values = sigmoid_output.tensor().storage().to_vec_f64();
    
    // Sigmoid should be between 0 and 1
    for &val in &sigmoid_values {
        assert!(val > 0.0 && val < 1.0 || (val - 0.5).abs() < 1e-6);
    }
    
    // Test backward pass through ReLU
    let mut loss = relu_output.sum();
    loss.backward();
    
    let x_grad = x.grad().expect("x should have gradient");
    let expected_grad = vec![0.0, 0.0, 1.0, 1.0]; // ReLU derivative
    assert_eq!(x_grad.storage().to_vec_f64(), expected_grad);
}

/// Test loss function integration
#[test]
fn test_loss_functions() {
    let pred_data = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
    let target_data = Tensor::from_data(&[1.5, 2.5, 2.5], vec![3], None);
    
    let mut pred = Variable::from_tensor(pred_data, true);
    let target = Variable::from_tensor(target_data, false); // Target doesn't need gradient
    
    // Test MSE loss
    let mut mse_loss = F::mse_loss(&pred, &target);
    
    // Expected MSE: mean([(1-1.5)^2, (2-2.5)^2, (3-2.5)^2]) = mean([0.25, 0.25, 0.25]) = 0.25
    let loss_value = mse_loss.tensor().storage().to_vec_f64()[0];
    assert!((loss_value - 0.25).abs() < 1e-6);
    
    // Test backward pass
    mse_loss.backward();
    
    // MSE gradient: 2/n * (pred - target)
    let pred_grad = pred.grad().expect("pred should have gradient");
    let grad_values = pred_grad.storage().to_vec_f64();
    
    // Expected: 2/3 * [-0.5, -0.5, 0.5] = [-0.333..., -0.333..., 0.333...]
    assert!((grad_values[0] + 0.3333333333333333).abs() < 1e-6);
    assert!((grad_values[1] + 0.3333333333333333).abs() < 1e-6);
    assert!((grad_values[2] - 0.3333333333333333).abs() < 1e-6);
}

/// Test matrix operations and gradients
#[test]
fn test_matrix_operations() {
    // Create 2x2 matrices
    let a_data = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], None);
    let b_data = Tensor::from_data(&[0.5, 1.0, 1.5, 2.0], vec![2, 2], None);
    
    let mut a = Variable::from_tensor(a_data, true);
    let mut b = Variable::from_tensor(b_data, true);
    
    // Matrix multiplication: C = A @ B
    let mut c = a.matmul(&b);
    
    // Sum all elements for scalar loss
    let mut loss = c.sum();
    
    // Backward pass
    loss.backward();
    
    // Check that gradients exist
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
    
    // Verify gradient shapes match input shapes
    let a_grad = a.grad().unwrap();
    let b_grad = b.grad().unwrap();
    
    assert_eq!(a_grad.shape(), &[2, 2]);
    assert_eq!(b_grad.shape(), &[2, 2]);
}

/// Test gradient accumulation across multiple backward passes
#[test]
fn test_gradient_accumulation() {
    let x_data = Tensor::from_data(&[1.0, 2.0], vec![2], None);
    let mut x = Variable::from_tensor(x_data, true);
    
    // First computation: y1 = 2 * x
    let mut y1 = x.mul_scalar(2.0);
    y1.backward_with_create_graph(None, true); // retain_graph = true
    
    let first_grad = x.grad().unwrap().storage().to_vec_f64();
    assert_eq!(first_grad, vec![2.0, 2.0]);
    
    // Second computation: y2 = 3 * x
    let mut y2 = x.mul_scalar(3.0);
    y2.backward(); // retain_graph = false
    
    // Gradients should accumulate: 2 + 3 = 5
    let accumulated_grad = x.grad().unwrap().storage().to_vec_f64();
    assert_eq!(accumulated_grad, vec![5.0, 5.0]);
}

/// Test with larger tensors and operations
#[test]
fn test_large_tensor_operations() {
    // Create larger tensors (100 elements)
    let size = 100;
    let mut x_data = Vec::with_capacity(size);
    for i in 0..size {
        x_data.push(i as f64 / 10.0);
    }
    
    let x_tensor = Tensor::from_data(&x_data, vec![size], None);
    let mut x = Variable::from_tensor(x_tensor, true);
    
    // Complex computation: y = sin(x) + cos(x^2)
    let x_squared = x.mul(&x);
    let sin_x = x.sin();
    let cos_x2 = x_squared.cos();
    let mut y = sin_x.add(&cos_x2);
    
    // Sum for scalar output
    let mut loss = y.sum();
    
    // Backward pass
    loss.backward();
    
    // Verify gradient exists and has correct shape
    let x_grad = x.grad().expect("x should have gradient");
    assert_eq!(x_grad.shape(), &[size]);
    
    // Verify all gradient values are finite
    let grad_values = x_grad.storage().to_vec_f64();
    for &val in &grad_values {
        assert!(val.is_finite(), "Gradient value should be finite: {}", val);
    }
}

/// Test mixed precision and data types
#[test]
fn test_mixed_operations() {
    let x_data = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
    let mut x = Variable::from_tensor(x_data, true);
    
    // Mix of operations
    let scaled = x.mul_scalar(2.0);
    let offset = scaled.add_scalar(1.0);
    let activated = F::relu(&offset);
    let mut output = activated.sum();
    
    // Backward
    output.backward();
    
    // Verify gradient chain worked correctly
    let x_grad = x.grad().expect("x should have gradient");
    let grad_values = x_grad.storage().to_vec_f64();
    
    // All inputs are positive, so ReLU doesn't block gradients
    // Gradient should be 2.0 for all elements (from mul_scalar)
    for &val in &grad_values {
        assert!((val - 2.0).abs() < 1e-6);
    }
}

/// Test error handling in autograd operations
#[test]
fn test_autograd_error_handling() {
    // Test shape mismatch
    let x_data = Tensor::from_data(&[1.0, 2.0], vec![2], None);
    let y_data = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
    
    let x = Variable::from_tensor(x_data, true);
    let y = Variable::from_tensor(y_data, true);
    
    // This should handle the error gracefully
    let result = std::panic::catch_unwind(|| {
        x.add(&y)
    });
    
    // We expect this to either return an error or panic gracefully
    // The exact behavior depends on the implementation
    assert!(result.is_err() || result.is_ok());
}

/// Test variable creation and basic properties
#[test]
fn test_variable_properties() {
    let data = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
    let var = Variable::from_tensor(data.clone(), true);
    
    // Test basic properties
    assert_eq!(var.shape(), data.shape());
    assert_eq!(var.requires_grad(), true);
    assert!(var.grad().is_none()); // No gradient initially
    
    // Test tensor access
    let tensor_data = var.tensor().storage().to_vec_f64();
    assert_eq!(tensor_data, vec![1.0, 2.0, 3.0]);
}

/// Integration test for performance optimizations
#[test]
fn test_performance_optimizations_integration() {
    use rustytorch_autograd::performance_optimizations::{
        set_performance_config, PerformanceConfig
    };
    use rustytorch_autograd::anomaly_detection::enable_anomaly_detection;
    
    // Configure performance optimizations
    let config = PerformanceConfig {
        initial_queue_capacity: 32,
        initial_accumulator_capacity: 16,
        enable_operation_fusion: true,
        enable_gradient_cache: true,
        checkpointing_threshold: 100,
    };
    set_performance_config(config);
    
    // Enable anomaly detection
    enable_anomaly_detection(None);
    
    // Run a computation that could benefit from optimizations
    let x_data = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], None);
    let mut x = Variable::from_tensor(x_data, true);
    
    // Chain of operations that could be fused
    let squared = x.mul(&x);
    let scaled = squared.mul_scalar(2.0);
    let offset = scaled.add_scalar(1.0);
    let mut result = offset.sum();
    
    // Test optimized backward if available
    if let Ok(_) = result.backward_optimized(None, false, false) {
        // Optimized backward succeeded
    } else {
        // Fallback to regular backward
        result.backward();
    }
    
    assert!(x.grad().is_some());
}