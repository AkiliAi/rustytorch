//! Benchmarks for tensor operations
//!
//! This module contains comprehensive performance benchmarks for core tensor operations,
//! including arithmetic, linear algebra, reductions, and type conversions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustytorch_core::{NumericOps, Reduction, Reshapable};
use rustytorch_tensor::Tensor;

/// Benchmark basic tensor creation operations
fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    // Test different sizes
    let sizes = [100, 1000, 10000, 100000];

    for size in sizes {
        group.bench_with_input(BenchmarkId::new("zeros", size), &size, |b, &size| {
            b.iter(|| {
                let tensor = Tensor::zeros(vec![size], None);
                black_box(tensor)
            })
        });

        group.bench_with_input(BenchmarkId::new("ones", size), &size, |b, &size| {
            b.iter(|| {
                let tensor = Tensor::ones(vec![size], None);
                black_box(tensor)
            })
        });

        group.bench_with_input(BenchmarkId::new("rand", size), &size, |b, &size| {
            b.iter(|| {
                let tensor = Tensor::rand(vec![size], None);
                black_box(tensor)
            })
        });
    }

    group.finish();
}

/// Benchmark arithmetic operations
fn bench_arithmetic_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic_ops");

    let sizes = [1000, 10000, 100000];

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32 + 1.0).collect();
        let a = Tensor::from_data(&data, vec![size], None);
        let b = Tensor::from_data(&data, vec![size], None);

        group.bench_with_input(BenchmarkId::new("add", size), &size, |bench, _| {
            bench.iter(|| {
                let result = a.clone().add(black_box(b.clone())).unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("mul", size), &size, |bench, _| {
            bench.iter(|| {
                let result = a.clone().mul(black_box(b.clone())).unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &size, |bench, _| {
            bench.iter(|| {
                let result = a.clone().sub(black_box(b.clone())).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark matrix multiplication
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");

    // Test square matrices of different sizes
    let sizes = [32, 64, 128, 256, 512];

    for size in sizes {
        let data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 + 1.0) / (size * size) as f32)
            .collect();
        let a = Tensor::from_data(&data, vec![size, size], None);
        let b = Tensor::from_data(&data, vec![size, size], None);

        group.bench_with_input(
            BenchmarkId::new("matmul", format!("{}x{}", size, size)),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = a.matmul(black_box(&b)).unwrap();
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark reduction operations
fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");

    let sizes = [1000, 10000, 100000];

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32 + 1.0).collect();
        let tensor = Tensor::from_data(&data, vec![size], None);

        group.bench_with_input(BenchmarkId::new("sum", size), &size, |bench, _| {
            bench.iter(|| {
                let result = tensor.sum().unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("mean", size), &size, |bench, _| {
            bench.iter(|| {
                let result = tensor.mean().unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("max", size), &size, |bench, _| {
            bench.iter(|| {
                let result = tensor.max().unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("min", size), &size, |bench, _| {
            bench.iter(|| {
                let result = tensor.min().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark multi-dimensional reductions
fn bench_multi_dim_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_dim_reductions");

    // Test 3D tensors with different axis reductions
    let shape = [100, 100, 100];
    let size = shape.iter().product::<usize>();
    let data: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) / size as f32).collect();
    let tensor = Tensor::from_data(&data, shape.to_vec(), None);

    for axis in 0..3 {
        group.bench_with_input(BenchmarkId::new("sum_dim", axis), &axis, |bench, &axis| {
            bench.iter(|| {
                let result = tensor.sum_dim(Some(black_box(axis))).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark linear algebra operations
fn bench_linear_algebra(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_algebra");

    let sizes = [32, 64, 128, 256];

    for size in sizes {
        // Create a well-conditioned matrix
        let mut data = vec![0.0f64; size * size];
        for i in 0..size {
            for j in 0..size {
                data[i * size + j] = if i == j {
                    2.0
                } else if (i as i32 - j as i32).abs() == 1 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
        let matrix = Tensor::from_data(&data, vec![size, size], None);

        group.bench_with_input(
            BenchmarkId::new("lu_decomposition", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = matrix.lu().unwrap();
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("qr_decomposition", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = matrix.qr().unwrap();
                    black_box(result)
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("determinant", size), &size, |bench, _| {
            bench.iter(|| {
                let result = matrix.det().unwrap();
                black_box(result)
            })
        });

        // Only benchmark smaller sizes for inverse due to computational cost
        if size <= 128 {
            group.bench_with_input(BenchmarkId::new("inverse", size), &size, |bench, _| {
                bench.iter(|| {
                    let result = matrix.inverse().unwrap();
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

/// Benchmark type conversions
fn bench_type_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_conversions");

    let sizes = [1000, 10000, 100000];

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32 + 1.0).collect();
        let tensor_f32 = Tensor::from_data(&data, vec![size], None);

        group.bench_with_input(BenchmarkId::new("f32_to_f64", size), &size, |bench, _| {
            bench.iter(|| {
                let result = tensor_f32.to_f64().unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("f32_to_i32", size), &size, |bench, _| {
            bench.iter(|| {
                let result = tensor_f32.to_i32().unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("f32_to_bool", size), &size, |bench, _| {
            bench.iter(|| {
                let result = tensor_f32.to_bool().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark tensor reshaping operations
fn bench_reshaping(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshaping");

    let size = 100000;
    let data: Vec<f32> = (0..size).map(|i| i as f32 + 1.0).collect();
    let tensor = Tensor::from_data(&data, vec![size], None);

    // Different reshape patterns
    let shapes = vec![
        vec![1000, 100],
        vec![100, 1000],
        vec![10, 10, 1000],
        vec![50, 50, 40],
        vec![25, 25, 160],
    ];

    for (i, shape) in shapes.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("reshape", i), shape, |bench, shape| {
            bench.iter(|| {
                let result = tensor.reshape(black_box(shape)).unwrap();
                black_box(result)
            })
        });
    }

    // Transpose operations
    let matrix_data: Vec<f32> = (0..10000).map(|i| i as f32 + 1.0).collect();
    let matrix = Tensor::from_data(&matrix_data, vec![100, 100], None);

    group.bench_function("transpose", |bench| {
        bench.iter(|| {
            let result = matrix.transpose(black_box(0), black_box(1)).unwrap();
            black_box(result)
        })
    });

    group.bench_function("flatten", |bench| {
        bench.iter(|| {
            let result = matrix.flatten().unwrap();
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark broadcasting operations
fn bench_broadcasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting");

    // Different broadcasting scenarios
    let scenarios = vec![
        // (shape_a, shape_b, description)
        (vec![1000], vec![1], "vector_scalar"),
        (vec![100, 100], vec![100], "matrix_vector"),
        (vec![100, 100], vec![1, 100], "matrix_row"),
        (vec![50, 50, 50], vec![50, 1], "tensor3d_matrix"),
    ];

    for (shape_a, shape_b, desc) in scenarios {
        let size_a = shape_a.iter().product();
        let size_b = shape_b.iter().product();

        let data_a: Vec<f32> = (0..size_a)
            .map(|i| (i as f32 + 1.0) / size_a as f32)
            .collect();
        let data_b: Vec<f32> = (0..size_b)
            .map(|i| (i as f32 + 1.0) / size_b as f32)
            .collect();

        let tensor_a = Tensor::from_data(&data_a, shape_a, None);
        let tensor_b = Tensor::from_data(&data_b, shape_b, None);

        group.bench_with_input(BenchmarkId::new("add_broadcast", desc), desc, |bench, _| {
            bench.iter(|| {
                let result = tensor_a.add_broadcast(black_box(&tensor_b)).unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("mul_broadcast", desc), desc, |bench, _| {
            bench.iter(|| {
                let result = tensor_a.mul_broadcast(black_box(&tensor_b)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_creation,
    bench_arithmetic_ops,
    bench_matrix_multiplication,
    bench_reductions,
    bench_multi_dim_reductions,
    bench_linear_algebra,
    bench_type_conversions,
    bench_reshaping,
    bench_broadcasting
);

criterion_main!(benches);
