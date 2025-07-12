//! Integration tests for rustytorch_tensor
//!
//! This module contains comprehensive integration tests that verify the correct
//! behavior of tensor operations in realistic scenarios and edge cases.

use rustytorch_core::{DType, NumericOps, Reduction, Reshapable};
use rustytorch_tensor::Tensor;
use std::f64;

#[test]
fn test_tensor_creation_and_basic_properties() {
    // Test different data types
    let f32_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor_f32 = Tensor::from_data(&f32_data, vec![2, 2], None);
    assert_eq!(tensor_f32.shape(), &[2, 2]);
    assert_eq!(tensor_f32.ndim(), 2);
    assert_eq!(tensor_f32.numel(), 4);
    assert_eq!(tensor_f32.dtype(), DType::Float32);

    // Test zeros and ones
    let zeros = Tensor::zeros(vec![3, 3], None);
    assert_eq!(zeros.shape(), &[3, 3]);
    assert_eq!(zeros.numel(), 9);

    let ones = Tensor::ones(vec![2, 3, 4], None);
    assert_eq!(ones.shape(), &[2, 3, 4]);
    assert_eq!(ones.numel(), 24);

    // Test random tensor
    let rand_tensor = Tensor::rand(vec![10, 10], None);
    assert_eq!(rand_tensor.shape(), &[10, 10]);
    assert_eq!(rand_tensor.numel(), 100);
}

#[test]
fn test_comprehensive_arithmetic_operations() {
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![2.0f32, 3.0, 4.0, 5.0];

    let a = Tensor::from_data(&a_data, vec![2, 2], None);
    let b = Tensor::from_data(&b_data, vec![2, 2], None);

    // Addition
    let add_result = a.clone().add(b.clone()).unwrap();
    let add_data = add_result.storage().to_vec_f64();
    assert_eq!(add_data, vec![3.0, 5.0, 7.0, 9.0]);

    // Subtraction
    let sub_result = a.clone().sub(b.clone()).unwrap();
    let sub_data = sub_result.storage().to_vec_f64();
    assert_eq!(sub_data, vec![-1.0, -1.0, -1.0, -1.0]);

    // Multiplication
    let mul_result = a.clone().mul(b.clone()).unwrap();
    let mul_data = mul_result.storage().to_vec_f64();
    assert_eq!(mul_data, vec![2.0, 6.0, 12.0, 20.0]);

    // Division
    let div_result = a.clone().div(b.clone()).unwrap();
    let div_data = div_result.storage().to_vec_f64();
    assert!((div_data[0] - 0.5).abs() < 1e-6);
    assert!((div_data[1] - (2.0 / 3.0)).abs() < 1e-6);
    assert!((div_data[2] - 0.75).abs() < 1e-6);
    assert!((div_data[3] - 0.8).abs() < 1e-6);
}

#[test]
fn test_matrix_operations_comprehensive() {
    // Test matrix multiplication with different sizes

    // 2x3 * 3x2 = 2x2
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let a = Tensor::from_data(&a_data, vec![2, 3], None);
    let b = Tensor::from_data(&b_data, vec![3, 2], None);

    let result = a.matmul(&b).unwrap();
    assert_eq!(result.shape(), &[2, 2]);

    let result_data = result.storage().to_vec_f64();
    // [1,2,3] * [7,8; 9,10; 11,12] = [58,64; 139,154]
    assert_eq!(result_data[0], 58.0);
    assert_eq!(result_data[1], 64.0);
    assert_eq!(result_data[2], 139.0);
    assert_eq!(result_data[3], 154.0);

    // Test square matrix multiplication
    let square_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let square = Tensor::from_data(&square_data, vec![2, 2], None);
    let square_result = square.matmul(&square).unwrap();

    let square_result_data = square_result.storage().to_vec_f64();
    // [1,2; 3,4] * [1,2; 3,4] = [7,10; 15,22]
    assert_eq!(square_result_data, vec![7.0, 10.0, 15.0, 22.0]);
}

#[test]
fn test_reduction_operations_comprehensive() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_data(&data, vec![2, 3], None);

    // Global reductions
    let sum = tensor.sum().unwrap();
    let sum_value = sum.storage().get_f64(0).unwrap();
    assert_eq!(sum_value, 21.0);

    let mean = tensor.mean().unwrap();
    let mean_value = mean.storage().get_f64(0).unwrap();
    assert_eq!(mean_value, 3.5);

    let max = tensor.max().unwrap();
    let max_value = max.storage().get_f64(0).unwrap();
    assert_eq!(max_value, 6.0);

    let min = tensor.min().unwrap();
    let min_value = min.storage().get_f64(0).unwrap();
    assert_eq!(min_value, 1.0);

    // Axis-specific reductions
    let sum_axis0 = tensor.sum_dim(Some(0)).unwrap();
    assert_eq!(sum_axis0.shape(), &[3]);
    let sum_axis0_data = sum_axis0.storage().to_vec_f64();
    assert_eq!(sum_axis0_data, vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]

    let sum_axis1 = tensor.sum_dim(Some(1)).unwrap();
    assert_eq!(sum_axis1.shape(), &[2]);
    let sum_axis1_data = sum_axis1.storage().to_vec_f64();
    assert_eq!(sum_axis1_data, vec![6.0, 15.0]); // [1+2+3, 4+5+6]

    // Test with keepdim (note: current API doesn't support keepdim parameter)
    let sum_keepdim = tensor.sum_dim(Some(1)).unwrap();
    assert_eq!(sum_keepdim.shape(), &[2]);

    // Test argmax and argmin
    let argmax = tensor.argmax(None, false).unwrap();
    let argmax_value = argmax.storage().get_f64(0).unwrap() as usize;
    assert_eq!(argmax_value, 5); // Index of value 6.0

    let argmin = tensor.argmin(None, false).unwrap();
    let argmin_value = argmin.storage().get_f64(0).unwrap() as usize;
    assert_eq!(argmin_value, 0); // Index of value 1.0
}

#[test]
fn test_tensor_reshaping_comprehensive() {
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let tensor = Tensor::from_data(&data, vec![12], None);

    // Test various reshape operations
    let reshaped_2d = tensor.reshape(&[3, 4]).unwrap();
    assert_eq!(reshaped_2d.shape(), &[3, 4]);
    assert_eq!(reshaped_2d.numel(), 12);

    let reshaped_3d = tensor.reshape(&[2, 2, 3]).unwrap();
    assert_eq!(reshaped_3d.shape(), &[2, 2, 3]);
    assert_eq!(reshaped_3d.numel(), 12);

    // Test transpose
    let matrix = tensor.reshape(&[3, 4]).unwrap();
    let transposed = matrix.transpose(0, 1).unwrap();
    assert_eq!(transposed.shape(), &[4, 3]);

    // Test flatten
    let flattened = reshaped_3d.flatten().unwrap();
    assert_eq!(flattened.shape(), &[12]);
    assert_eq!(flattened.numel(), 12);

    // Verify data integrity after reshaping
    let original_data = tensor.storage().to_vec_f64();
    let flattened_data = flattened.storage().to_vec_f64();
    assert_eq!(original_data, flattened_data);
}

#[test]
fn test_linear_algebra_comprehensive() {
    // Test with a well-conditioned 3x3 matrix
    let data = vec![2.0f64, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
    let matrix = Tensor::from_data(&data, vec![3, 3], None);

    // Test determinant
    let det = matrix.det().unwrap();
    assert!((det - 4.0).abs() < 1e-10); // Expected determinant is 4

    // Test LU decomposition
    let (l, u, p) = matrix.lu().unwrap();
    assert_eq!(l.shape(), &[3, 3]);
    assert_eq!(u.shape(), &[3, 3]);
    assert_eq!(p.shape(), &[3, 3]);

    // Verify P*A = L*U
    let pa = p.matmul(&matrix).unwrap();
    let lu = l.matmul(&u).unwrap();
    let pa_data = pa.storage().to_vec_f64();
    let lu_data = lu.storage().to_vec_f64();
    for i in 0..9 {
        assert!((pa_data[i] - lu_data[i]).abs() < 1e-10);
    }

    // Test QR decomposition
    let (q, r) = matrix.qr().unwrap();
    assert_eq!(q.shape(), &[3, 3]);
    assert_eq!(r.shape(), &[3, 3]);

    // Verify A = Q*R
    let qr = q.matmul(&r).unwrap();
    let matrix_data = matrix.storage().to_vec_f64();
    let qr_data = qr.storage().to_vec_f64();
    for i in 0..9 {
        assert!((matrix_data[i] - qr_data[i]).abs() < 1e-10);
    }

    // Test linear system solving
    let b_data = vec![1.0f64, 2.0, 3.0];
    let b = Tensor::from_data(&b_data, vec![3], None);
    let x = matrix.solve(&b).unwrap();
    assert_eq!(x.shape(), &[3]);

    // Verify A*x = b
    let ax = matrix.matmul(&x.reshape(&[3, 1]).unwrap()).unwrap();
    let ax_data = ax.storage().to_vec_f64();
    for i in 0..3 {
        assert!((ax_data[i] - b_data[i]).abs() < 1e-10);
    }
}

#[test]
fn test_type_conversions_comprehensive() {
    let f32_data = vec![1.5f32, 2.7, -3.2, 4.0];
    let tensor_f32 = Tensor::from_data(&f32_data, vec![4], None);

    // Test conversion to f64
    let tensor_f64 = tensor_f32.to_f64().unwrap();
    assert_eq!(tensor_f64.dtype(), DType::Float64);
    let f64_data = tensor_f64.storage().to_vec_f64();
    assert!((f64_data[0] - 1.5).abs() < 1e-6);
    assert!((f64_data[1] - 2.7).abs() < 1e-6);

    // Test conversion to i32
    let tensor_i32 = tensor_f32.to_i32().unwrap();
    assert_eq!(tensor_i32.dtype(), DType::Int32);

    // Test conversion to bool
    let tensor_bool = tensor_f32.to_bool().unwrap();
    assert_eq!(tensor_bool.dtype(), DType::Bool);

    // Test type properties
    assert!(tensor_f32.is_floating_point());
    assert!(!tensor_f32.is_integral());
    assert!(!tensor_f32.is_complex());

    assert!(!tensor_i32.is_floating_point());
    assert!(tensor_i32.is_integral());
    assert!(!tensor_i32.is_complex());
}

#[test]
fn test_broadcasting_comprehensive() {
    // Scalar + Vector
    let scalar = Tensor::from_data(&[5.0f32], vec![1], None);
    let vector = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], None);

    let result = scalar.add_broadcast(&vector).unwrap();
    assert_eq!(result.shape(), &[3]);
    let result_data = result.storage().to_vec_f64();
    assert_eq!(result_data, vec![6.0, 7.0, 8.0]);

    // Matrix + Vector (row broadcasting)
    let matrix_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = Tensor::from_data(&matrix_data, vec![2, 3], None);
    let row_vector = Tensor::from_data(&[10.0f32, 20.0, 30.0], vec![1, 3], None);

    let broadcast_result = matrix.add_broadcast(&row_vector).unwrap();
    assert_eq!(broadcast_result.shape(), &[2, 3]);
    let broadcast_data = broadcast_result.storage().to_vec_f64();
    assert_eq!(broadcast_data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

    // Test multiplication broadcasting
    let mul_result = matrix.mul_broadcast(&row_vector).unwrap();
    let mul_data = mul_result.storage().to_vec_f64();
    assert_eq!(mul_data, vec![10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
}

#[test]
fn test_edge_cases_and_error_handling() {
    // Test empty tensors
    let empty = Tensor::zeros(vec![0], None);
    assert_eq!(empty.numel(), 0);
    assert_eq!(empty.shape(), &[0]);

    // Test single element tensors
    let single = Tensor::from_data(&[42.0f32], vec![], None);
    assert_eq!(single.numel(), 1);
    assert_eq!(single.ndim(), 0);

    // Test large tensors (memory and performance)
    let large = Tensor::zeros(vec![1000, 1000], None);
    assert_eq!(large.numel(), 1_000_000);

    // Test invalid reshape
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_data(&data, vec![2, 2], None);

    let invalid_reshape = tensor.reshape(&[3, 3]);
    assert!(invalid_reshape.is_err());

    // Test invalid transpose
    let invalid_transpose = tensor.transpose(0, 5);
    assert!(invalid_transpose.is_err());

    // Test incompatible matrix multiplication
    let a = Tensor::from_data(&[1.0f32, 2.0], vec![1, 2], None);
    let b = Tensor::from_data(&[3.0f32, 4.0, 5.0], vec![1, 3], None);

    let invalid_matmul = a.matmul(&b);
    assert!(invalid_matmul.is_err());
}

#[test]
fn test_numerical_stability() {
    // Test operations with very small numbers
    let small_data = vec![1e-10f64, 2e-10, 3e-10, 4e-10];
    let small_tensor = Tensor::from_data(&small_data, vec![2, 2], None);

    let sum = small_tensor.sum().unwrap();
    let sum_value = sum.storage().get_f64(0).unwrap();
    assert!((sum_value - 10e-10).abs() < 1e-15);

    // Test operations with very large numbers
    let large_data = vec![1e10f64, 2e10, 3e10, 4e10];
    let large_tensor = Tensor::from_data(&large_data, vec![2, 2], None);

    let large_sum = large_tensor.sum().unwrap();
    let large_sum_value = large_sum.storage().get_f64(0).unwrap();
    assert!((large_sum_value - 10e10).abs() < 1e5);

    // Test operations with mixed scales
    let mixed_data = vec![1e-5f64, 1e5, 1e-5, 1e5];
    let mixed_tensor = Tensor::from_data(&mixed_data, vec![2, 2], None);

    let mixed_sum = mixed_tensor.sum().unwrap();
    let mixed_sum_value = mixed_sum.storage().get_f64(0).unwrap();
    assert!((mixed_sum_value - 2.00002e5).abs() < 1e-10);
}

#[test]
fn test_memory_efficiency() {
    // Test that views don't copy data
    let large_data: Vec<f32> = (0..100000).map(|i| i as f32).collect();
    let tensor = Tensor::from_data(&large_data, vec![1000, 100], None);

    // Reshaping should be efficient (no data copy)
    let reshaped = tensor.reshape(&[100, 1000]).unwrap();
    assert_eq!(reshaped.shape(), &[100, 1000]);

    // Transpose should be efficient (no data copy)
    let transposed = tensor.transpose(0, 1).unwrap();
    assert_eq!(transposed.shape(), &[100, 1000]);

    // Verify data integrity
    let original_sum = tensor.sum().unwrap().storage().get_f64(0).unwrap();
    let reshaped_sum = reshaped.sum().unwrap().storage().get_f64(0).unwrap();
    let transposed_sum = transposed.sum().unwrap().storage().get_f64(0).unwrap();

    assert!((original_sum - reshaped_sum).abs() < 1e-6);
    assert!((original_sum - transposed_sum).abs() < 1e-6);
}

#[test]
fn test_consistency_across_operations() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_data(&data, vec![2, 3], None);

    // Test that multiple ways of computing the same result are consistent

    // Sum all elements multiple ways
    let sum1 = tensor.sum().unwrap().storage().get_f64(0).unwrap();
    let sum2 = tensor
        .flatten()
        .unwrap()
        .sum()
        .unwrap()
        .storage()
        .get_f64(0)
        .unwrap();
    let sum3_axis0 = tensor
        .sum_dim(Some(0))
        .unwrap()
        .sum()
        .unwrap()
        .storage()
        .get_f64(0)
        .unwrap();
    let sum3_axis1 = tensor
        .sum_dim(Some(1))
        .unwrap()
        .sum()
        .unwrap()
        .storage()
        .get_f64(0)
        .unwrap();

    assert!((sum1 - sum2).abs() < 1e-10);
    assert!((sum1 - sum3_axis0).abs() < 1e-10);
    assert!((sum1 - sum3_axis1).abs() < 1e-10);

    // Test that transpose twice returns to original
    let double_transpose = tensor.transpose(0, 1).unwrap().transpose(0, 1).unwrap();
    let original_data = tensor.storage().to_vec_f64();
    let double_transpose_data = double_transpose.storage().to_vec_f64();

    for (orig, dt) in original_data.iter().zip(double_transpose_data.iter()) {
        assert!((orig - dt).abs() < 1e-10);
    }

    // Test that reshape to original shape preserves data
    let original_shape = tensor.shape().to_vec();
    let reshaped_back = tensor
        .reshape(&[6])
        .unwrap()
        .reshape(&original_shape)
        .unwrap();
    let reshaped_data = reshaped_back.storage().to_vec_f64();

    for (orig, reshaped) in original_data.iter().zip(reshaped_data.iter()) {
        assert!((orig - reshaped).abs() < 1e-10);
    }
}
