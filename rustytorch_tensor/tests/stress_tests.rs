//! Stress tests for rustytorch_tensor
//!
//! This module contains stress tests that verify tensor operations work correctly
//! under demanding conditions such as large matrices, extreme values, and edge cases.

use rustytorch_core::{NumericOps, Reduction, Reshapable};
use rustytorch_tensor::Tensor;

#[test]
fn test_large_matrix_operations() {
    // Test with reasonably large matrices (not too large to avoid CI timeouts)
    let size = 500;

    // Create test matrices with known patterns
    let mut data_a = vec![0.0f32; size * size];
    let mut data_b = vec![0.0f32; size * size];

    for i in 0..size {
        for j in 0..size {
            data_a[i * size + j] = (i + j) as f32 / (size * size) as f32;
            data_b[i * size + j] = (i * j) as f32 / (size * size) as f32;
        }
    }

    let matrix_a = Tensor::from_data(&data_a, vec![size, size], None);
    let matrix_b = Tensor::from_data(&data_b, vec![size, size], None);

    // Test matrix multiplication
    let result = matrix_a.matmul(&matrix_b).unwrap();
    assert_eq!(result.shape(), &[size, size]);
    assert_eq!(result.numel(), size * size);

    // Verify the result is finite and reasonable
    let result_data = result.storage().to_vec_f64();
    for &value in result_data.iter() {
        assert!(value.is_finite());
        assert!(value >= 0.0); // Given our input pattern, result should be non-negative
    }

    // Test element-wise operations
    let add_result = matrix_a.clone().add(matrix_b.clone()).unwrap();
    assert_eq!(add_result.shape(), &[size, size]);

    let mul_result = matrix_a.clone().mul(matrix_b.clone()).unwrap();
    assert_eq!(mul_result.shape(), &[size, size]);
}

#[test]
fn test_large_tensor_reductions() {
    let size = 1_000_000; // 1M elements
    let data: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) / size as f32).collect();
    let tensor = Tensor::from_data(&data, vec![size], None);

    // Test global reductions
    let sum = tensor.sum().unwrap();
    let sum_value = sum.storage().get_f64(0).unwrap();

    // Expected sum is approximately size/2 (average of 1/size to 1)
    let expected_sum = 0.5 + 0.5 / size as f64;
    assert!((sum_value - expected_sum).abs() < 1e-6);

    let mean = tensor.mean().unwrap();
    let mean_value = mean.storage().get_f64(0).unwrap();
    let expected_mean = expected_sum / size as f64;
    assert!((mean_value - expected_mean).abs() < 1e-9);

    // Test min/max
    let min = tensor.min().unwrap();
    let min_value = min.storage().get_f64(0).unwrap();
    assert!((min_value - 1.0 / size as f64).abs() < 1e-9);

    let max = tensor.max().unwrap();
    let max_value = max.storage().get_f64(0).unwrap();
    assert!((max_value - 1.0).abs() < 1e-9);
}

#[test]
fn test_multi_dimensional_large_tensors() {
    // Test with 3D tensor: 100x100x100 = 1M elements
    let dims = [100, 100, 100];
    let size = dims.iter().product::<usize>();

    let data: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
    let tensor = Tensor::from_data(&data, dims.to_vec(), None);

    // Test axis reductions
    for axis in 0..3 {
        let reduced = tensor.sum_dim(Some(axis)).unwrap();
        assert_eq!(reduced.ndim(), 2);
        assert_eq!(reduced.numel(), size / dims[axis]);

        // Verify the sum is preserved
        let total_sum = reduced.sum().unwrap().storage().get_f64(0).unwrap();
        let original_sum = tensor.sum().unwrap().storage().get_f64(0).unwrap();
        assert!((total_sum - original_sum).abs() < 1e-6);
    }

    // Test reshaping
    let reshaped = tensor.reshape(&[1000, 1000]).unwrap();
    assert_eq!(reshaped.shape(), &[1000, 1000]);

    let flattened = tensor.flatten().unwrap();
    assert_eq!(flattened.shape(), &[size]);

    // Verify data integrity
    let original_sum = tensor.sum().unwrap().storage().get_f64(0).unwrap();
    let reshaped_sum = reshaped.sum().unwrap().storage().get_f64(0).unwrap();
    let flattened_sum = flattened.sum().unwrap().storage().get_f64(0).unwrap();

    assert!((original_sum - reshaped_sum).abs() < 1e-10);
    assert!((original_sum - flattened_sum).abs() < 1e-10);
}

#[test]
fn test_extreme_values() {
    // Test with very small values
    let small_data = vec![f32::MIN_POSITIVE; 1000];
    let small_tensor = Tensor::from_data(&small_data, vec![1000], None);

    let small_sum = small_tensor.sum().unwrap();
    let small_sum_value = small_sum.storage().get_f64(0).unwrap();
    assert!(small_sum_value.is_finite());
    assert!(small_sum_value > 0.0);

    // Test with large values (but not MAX to avoid overflow)
    let large_value = 1e30f32;
    let large_data = vec![large_value; 100]; // Small count to avoid overflow
    let large_tensor = Tensor::from_data(&large_data, vec![100], None);

    let large_sum = large_tensor.sum().unwrap();
    let large_sum_value = large_sum.storage().get_f64(0).unwrap();
    assert!(large_sum_value.is_finite());
    assert!(large_sum_value > 0.0);

    // Test with mixed positive and negative values
    let mixed_data: Vec<f32> = (0..10000)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mixed_tensor = Tensor::from_data(&mixed_data, vec![10000], None);

    let mixed_sum = mixed_tensor.sum().unwrap();
    let mixed_sum_value = mixed_sum.storage().get_f64(0).unwrap();
    assert!((mixed_sum_value).abs() < 1e-10); // Should sum to ~0

    let mixed_mean = mixed_tensor.mean().unwrap();
    let mixed_mean_value = mixed_mean.storage().get_f64(0).unwrap();
    assert!((mixed_mean_value).abs() < 1e-10); // Should average to ~0
}

#[test]
fn test_precision_accumulation() {
    // Test that precision is maintained in long accumulation chains
    let size = 100000;

    // Create data where each element is a small increment
    let increment = 1e-6f64;
    let data: Vec<f64> = (0..size).map(|_| increment).collect();
    let tensor = Tensor::from_data(&data, vec![size], None);

    let sum = tensor.sum().unwrap();
    let sum_value = sum.storage().get_f64(0).unwrap();

    // Expected sum should be size * increment
    let expected = size as f64 * increment;
    let relative_error = (sum_value - expected).abs() / expected;

    // Allow for some floating point error, but it should be small
    assert!(
        relative_error < 1e-10,
        "Relative error {} too large. Expected: {}, Got: {}",
        relative_error,
        expected,
        sum_value
    );
}

#[test]
fn test_memory_intensive_operations() {
    // Test operations that might stress memory allocation
    let base_size = 1000;

    // Create a series of tensors of increasing size
    for multiplier in [1, 2, 4, 8] {
        let size = base_size * multiplier;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::from_data(&data, vec![size], None);

        // Test multiple reshape operations
        let sqrt_size = (size as f64).sqrt() as usize;
        if sqrt_size * sqrt_size == size {
            let matrix = tensor.reshape(&[sqrt_size, sqrt_size]).unwrap();
            let transposed = matrix.transpose(0, 1).unwrap();
            let back_to_vector = transposed.flatten().unwrap();

            assert_eq!(back_to_vector.numel(), size);
        }

        // Test reductions
        let sum = tensor.sum().unwrap();
        assert!(sum.storage().get_f64(0).unwrap().is_finite());

        let mean = tensor.mean().unwrap();
        assert!(mean.storage().get_f64(0).unwrap().is_finite());
    }
}

#[test]
fn test_broadcasting_stress() {
    // Test broadcasting with large tensors
    let large_size = 1000;
    let small_size = 100;

    // Large tensor
    let large_data: Vec<f32> = (0..large_size * large_size)
        .map(|i| (i as f32 + 1.0) / (large_size * large_size) as f32)
        .collect();
    let large_tensor = Tensor::from_data(&large_data, vec![large_size, large_size], None);

    // Small tensor for broadcasting
    let small_data: Vec<f32> = (0..large_size)
        .map(|i| (i as f32 + 1.0) / large_size as f32)
        .collect();
    let small_tensor = Tensor::from_data(&small_data, vec![1, large_size], None);

    // Test broadcasting addition
    let broadcast_result = large_tensor.add_broadcast(&small_tensor).unwrap();
    assert_eq!(broadcast_result.shape(), &[large_size, large_size]);

    // Verify the result makes sense
    let result_sum = broadcast_result
        .sum()
        .unwrap()
        .storage()
        .get_f64(0)
        .unwrap();
    let large_sum = large_tensor.sum().unwrap().storage().get_f64(0).unwrap();
    let small_sum = small_tensor.sum().unwrap().storage().get_f64(0).unwrap();

    // The result should be approximately large_sum + small_sum * large_size
    let expected_sum = large_sum + small_sum * large_size as f64;
    let relative_error = (result_sum - expected_sum).abs() / expected_sum;
    assert!(relative_error < 1e-6);
}

#[test]
fn test_linear_algebra_stress() {
    // Test linear algebra operations with moderately large matrices
    let size = 200;

    // Create a well-conditioned matrix
    let mut data = vec![0.0f64; size * size];
    for i in 0..size {
        for j in 0..size {
            if i == j {
                data[i * size + j] = 2.0;
            } else if (i as i32 - j as i32).abs() == 1 {
                data[i * size + j] = -1.0;
            }
        }
    }
    let matrix = Tensor::from_data(&data, vec![size, size], None);

    // Test determinant computation
    let det = matrix.det().unwrap();
    assert!(det.is_finite());
    assert!(det != 0.0); // Matrix should be non-singular

    // Test LU decomposition
    let (l, u, p) = matrix.lu().unwrap();
    assert_eq!(l.shape(), &[size, size]);
    assert_eq!(u.shape(), &[size, size]);
    assert_eq!(p.shape(), &[size, size]);

    // Verify decomposition accuracy on a subset (full verification would be expensive)
    let test_indices = [0, size / 4, size / 2, 3 * size / 4, size - 1];
    for &i in &test_indices {
        for &j in &test_indices {
            let pa_elem = p.matmul(&matrix).unwrap();
            let lu_elem = l.matmul(&u).unwrap();

            let pa_data = pa_elem.storage().to_vec_f64();
            let lu_data = lu_elem.storage().to_vec_f64();

            let idx = i * size + j;
            assert!((pa_data[idx] - lu_data[idx]).abs() < 1e-10);
        }
    }

    // Test solving a linear system
    let b_data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
    let b = Tensor::from_data(&b_data, vec![size], None);

    let x = matrix.solve(&b).unwrap();
    assert_eq!(x.shape(), &[size]);

    // Verify solution by checking residual on a subset
    let ax = matrix.matmul(&x.reshape(&[size, 1]).unwrap()).unwrap();
    let ax_data = ax.storage().to_vec_f64();

    for &i in &test_indices {
        assert!((ax_data[i] - b_data[i]).abs() < 1e-8);
    }
}

#[test]
fn test_type_conversion_stress() {
    let size = 50000;

    // Start with f32 data
    let f32_data: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) / size as f32).collect();
    let f32_tensor = Tensor::from_data(&f32_data, vec![size], None);

    // Chain of conversions
    let f64_tensor = f32_tensor.to_f64().unwrap();
    let i32_tensor = f64_tensor.to_i32().unwrap();
    let bool_tensor = i32_tensor.to_bool().unwrap();
    let back_to_f32 = bool_tensor.to_f32().unwrap();

    // Verify shapes are preserved
    assert_eq!(f64_tensor.shape(), &[size]);
    assert_eq!(i32_tensor.shape(), &[size]);
    assert_eq!(bool_tensor.shape(), &[size]);
    assert_eq!(back_to_f32.shape(), &[size]);

    // Verify data integrity where possible
    let f64_data = f64_tensor.storage().to_vec_f64();
    for i in 0..std::cmp::min(100, size) {
        // Check first 100 elements
        assert!((f64_data[i] - f32_data[i] as f64).abs() < 1e-6);
    }

    // Boolean conversion should be all true (since all values > 0)
    let bool_sum = bool_tensor.sum().unwrap().storage().get_f64(0).unwrap();
    assert_eq!(bool_sum, size as f64);
}

#[test]
fn test_performance_regression() {
    // This test serves as a basic performance regression check
    // Times might vary, but operations should complete in reasonable time

    use std::time::Instant;

    let size = 1000;
    let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
    let matrix = Tensor::from_data(&data, vec![size, size], None);

    // Matrix multiplication should complete quickly
    let start = Instant::now();
    let _result = matrix.matmul(&matrix).unwrap();
    let duration = start.elapsed();

    // Should complete in less than 10 seconds (very generous bound)
    assert!(
        duration.as_secs() < 10,
        "Matrix multiplication took too long: {:?}",
        duration
    );

    // Large reduction should complete quickly
    let large_size = 1_000_000;
    let large_data: Vec<f32> = (0..large_size).map(|i| i as f32).collect();
    let large_tensor = Tensor::from_data(&large_data, vec![large_size], None);

    let start = Instant::now();
    let _sum = large_tensor.sum().unwrap();
    let duration = start.elapsed();

    // Should complete in less than 1 second
    assert!(
        duration.as_millis() < 1000,
        "Large sum took too long: {:?}",
        duration
    );
}
