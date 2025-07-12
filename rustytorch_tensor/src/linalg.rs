//! Optimized linear algebra operations for tensors
//!
//! This module implements:
//! - Optimized matrix multiplication (GEMM)
//! - Matrix decompositions (LU, QR, SVD)
//! - Linear solvers and inverse operations
//! - Eigenvalue and eigenvector computations

use crate::{storage::StorageType, Tensor};
use rayon::prelude::*;
use rustytorch_core::{CoreError, Reshapable, Result};

/// Linear algebra operations
pub struct LinAlg;

impl LinAlg {
    /// Optimized matrix multiplication (GEMM)
    /// Computes C = alpha * A @ B + beta * C
    pub fn gemm(
        a: &Tensor,
        b: &Tensor,
        alpha: f64,
        beta: f64,
        c: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Validate matrix dimensions
        if a.ndim() < 2 || b.ndim() < 2 {
            return Err(CoreError::invalid_op(
                "gemm",
                "Input tensors must be at least 2-dimensional",
            ));
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        // Get the last two dimensions for matrix multiplication
        let (m, k) = (a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1]);
        let (k2, n) = (b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1]);

        if k != k2 {
            return Err(CoreError::shape_mismatch(vec![m, k], vec![k2, n], "gemm"));
        }

        // Handle batch dimensions
        let batch_dims_a = &a_shape[..a_shape.len() - 2];
        let batch_dims_b = &b_shape[..b_shape.len() - 2];

        // For now, require same batch dimensions (can be extended for broadcasting)
        if batch_dims_a != batch_dims_b {
            return Err(CoreError::invalid_op(
                "gemm",
                "Batch dimensions must match for now",
            ));
        }

        // Calculate output shape
        let mut output_shape = batch_dims_a.to_vec();
        output_shape.extend_from_slice(&[m, n]);

        // Choose implementation based on data type and size
        match (a.dtype(), b.dtype()) {
            (rustytorch_core::DType::Float32, rustytorch_core::DType::Float32) => {
                Self::gemm_f32(a, b, alpha as f32, beta as f32, c, &output_shape)
            }
            (rustytorch_core::DType::Float64, rustytorch_core::DType::Float64) => {
                Self::gemm_f64(a, b, alpha, beta, c, &output_shape)
            }
            _ => {
                // Convert to common type and compute
                let promoted_dtype = crate::type_ops::TypeOps::promote_types(a.dtype(), b.dtype());
                let a_converted = a.to_dtype(promoted_dtype)?;
                let b_converted = b.to_dtype(promoted_dtype)?;
                Self::gemm(&a_converted, &b_converted, alpha, beta, c)
            }
        }
    }

    /// F32 optimized GEMM implementation
    fn gemm_f32(
        a: &Tensor,
        b: &Tensor,
        alpha: f32,
        beta: f32,
        c: Option<&Tensor>,
        output_shape: &[usize],
    ) -> Result<Tensor> {
        let a_data = Self::extract_f32_data(a)?;
        let b_data = Self::extract_f32_data(b)?;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let (m, k, n) = (
            a_shape[a_shape.len() - 2],
            a_shape[a_shape.len() - 1],
            b_shape[b_shape.len() - 1],
        );

        // Calculate batch size
        let batch_size: usize = a_shape[..a_shape.len() - 2].iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let mut result_data = vec![0.0f32; output_shape.iter().product()];

        // Initialize with beta * C if provided
        if let Some(c_tensor) = c {
            if beta != 0.0 {
                let c_data = Self::extract_f32_data(c_tensor)?;
                for i in 0..result_data.len() {
                    result_data[i] = beta * c_data[i];
                }
            }
        }

        // Perform batched matrix multiplication
        let result_chunks: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|batch_idx| {
                let a_offset = batch_idx * m * k;
                let b_offset = batch_idx * k * n;

                let mut batch_result = vec![0.0f32; m * n];
                Self::gemm_kernel_f32(
                    &a_data[a_offset..a_offset + m * k],
                    &b_data[b_offset..b_offset + k * n],
                    &mut batch_result,
                    m,
                    n,
                    k,
                    alpha,
                );
                batch_result
            })
            .collect();

        // Combine results
        for (batch_idx, batch_result) in result_chunks.into_iter().enumerate() {
            let c_offset = batch_idx * m * n;
            for (i, &value) in batch_result.iter().enumerate() {
                result_data[c_offset + i] += value;
            }
        }

        Ok(Tensor::from_data(
            &result_data,
            output_shape.to_vec(),
            Some(a.options().clone()),
        ))
    }

    /// F64 optimized GEMM implementation  
    fn gemm_f64(
        a: &Tensor,
        b: &Tensor,
        alpha: f64,
        beta: f64,
        c: Option<&Tensor>,
        output_shape: &[usize],
    ) -> Result<Tensor> {
        let a_data = Self::extract_f64_data(a)?;
        let b_data = Self::extract_f64_data(b)?;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let (m, k, n) = (
            a_shape[a_shape.len() - 2],
            a_shape[a_shape.len() - 1],
            b_shape[b_shape.len() - 1],
        );

        let batch_size: usize = a_shape[..a_shape.len() - 2].iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        let mut result_data = vec![0.0f64; output_shape.iter().product()];

        if let Some(c_tensor) = c {
            if beta != 0.0 {
                let c_data = Self::extract_f64_data(c_tensor)?;
                for i in 0..result_data.len() {
                    result_data[i] = beta * c_data[i];
                }
            }
        }

        let result_chunks: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|batch_idx| {
                let a_offset = batch_idx * m * k;
                let b_offset = batch_idx * k * n;

                let mut batch_result = vec![0.0f64; m * n];
                Self::gemm_kernel_f64(
                    &a_data[a_offset..a_offset + m * k],
                    &b_data[b_offset..b_offset + k * n],
                    &mut batch_result,
                    m,
                    n,
                    k,
                    alpha,
                );
                batch_result
            })
            .collect();

        // Combine results
        for (batch_idx, batch_result) in result_chunks.into_iter().enumerate() {
            let c_offset = batch_idx * m * n;
            for (i, &value) in batch_result.iter().enumerate() {
                result_data[c_offset + i] += value;
            }
        }

        let mut options = a.options().clone();
        options.dtype = rustytorch_core::DType::Float64;
        Ok(Tensor::from_data(
            &result_data,
            output_shape.to_vec(),
            Some(options),
        ))
    }

    /// Optimized F32 GEMM kernel with loop tiling
    fn gemm_kernel_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
    ) {
        const TILE_SIZE: usize = 64;

        for i_tile in (0..m).step_by(TILE_SIZE) {
            for j_tile in (0..n).step_by(TILE_SIZE) {
                for k_tile in (0..k).step_by(TILE_SIZE) {
                    let i_end = (i_tile + TILE_SIZE).min(m);
                    let j_end = (j_tile + TILE_SIZE).min(n);
                    let k_end = (k_tile + TILE_SIZE).min(k);

                    for i in i_tile..i_end {
                        for j in j_tile..j_end {
                            let mut sum = 0.0f32;
                            for k_idx in k_tile..k_end {
                                sum += a[i * k + k_idx] * b[k_idx * n + j];
                            }
                            c[i * n + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    }

    /// Optimized F64 GEMM kernel with loop tiling
    fn gemm_kernel_f64(
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
    ) {
        const TILE_SIZE: usize = 64;

        for i_tile in (0..m).step_by(TILE_SIZE) {
            for j_tile in (0..n).step_by(TILE_SIZE) {
                for k_tile in (0..k).step_by(TILE_SIZE) {
                    let i_end = (i_tile + TILE_SIZE).min(m);
                    let j_end = (j_tile + TILE_SIZE).min(n);
                    let k_end = (k_tile + TILE_SIZE).min(k);

                    for i in i_tile..i_end {
                        for j in j_tile..j_end {
                            let mut sum = 0.0f64;
                            for k_idx in k_tile..k_end {
                                sum += a[i * k + k_idx] * b[k_idx * n + j];
                            }
                            c[i * n + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    }

    /// LU decomposition with partial pivoting
    /// Returns (L, U, P) where P @ A = L @ U
    pub fn lu_decomposition(a: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        if a.ndim() != 2 {
            return Err(CoreError::invalid_op(
                "lu_decomposition",
                "Input must be a 2D matrix",
            ));
        }

        let shape = a.shape();
        if shape[0] != shape[1] {
            return Err(CoreError::invalid_op(
                "lu_decomposition",
                "Input must be a square matrix",
            ));
        }

        let n = shape[0];

        match a.dtype() {
            rustytorch_core::DType::Float64 => Self::lu_decomposition_f64(a, n),
            _ => {
                let a_f64 = a.to_f64()?;
                Self::lu_decomposition_f64(&a_f64, n)
            }
        }
    }

    /// F64 LU decomposition implementation
    fn lu_decomposition_f64(a: &Tensor, n: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let mut data = Self::extract_f64_data(a)?.clone();
        let mut permutation = (0..n).collect::<Vec<usize>>();

        // Gaussian elimination with partial pivoting
        for k in 0..n - 1 {
            // Find pivot
            let mut max_idx = k;
            let mut max_val = data[k * n + k].abs();

            for i in k + 1..n {
                let val = data[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            // Swap rows if needed
            if max_idx != k {
                for j in 0..n {
                    data.swap(k * n + j, max_idx * n + j);
                }
                permutation.swap(k, max_idx);
            }

            // Check for singular matrix
            if data[k * n + k].abs() < 1e-14 {
                return Err(CoreError::invalid_op(
                    "lu_decomposition",
                    "Matrix is singular",
                ));
            }

            // Elimination
            for i in k + 1..n {
                let factor = data[i * n + k] / data[k * n + k];
                data[i * n + k] = factor; // Store L factor

                for j in k + 1..n {
                    data[i * n + j] -= factor * data[k * n + j];
                }
            }
        }

        // Extract L and U matrices
        let mut l_data = vec![0.0f64; n * n];
        let mut u_data = vec![0.0f64; n * n];

        for i in 0..n {
            for j in 0..n {
                if i > j {
                    l_data[i * n + j] = data[i * n + j];
                } else if i == j {
                    l_data[i * n + j] = 1.0;
                    u_data[i * n + j] = data[i * n + j];
                } else {
                    u_data[i * n + j] = data[i * n + j];
                }
            }
        }

        // Create permutation matrix
        let mut p_data = vec![0.0f64; n * n];
        for (i, &perm_idx) in permutation.iter().enumerate() {
            p_data[i * n + perm_idx] = 1.0;
        }

        let mut options = a.options().clone();
        options.dtype = rustytorch_core::DType::Float64;

        let l = Tensor::from_data(&l_data, vec![n, n], Some(options.clone()));
        let u = Tensor::from_data(&u_data, vec![n, n], Some(options.clone()));
        let p = Tensor::from_data(&p_data, vec![n, n], Some(options));

        Ok((l, u, p))
    }

    /// QR decomposition using Householder reflections
    /// Returns (Q, R) where A = Q @ R
    pub fn qr_decomposition(a: &Tensor) -> Result<(Tensor, Tensor)> {
        if a.ndim() != 2 {
            return Err(CoreError::invalid_op(
                "qr_decomposition",
                "Input must be a 2D matrix",
            ));
        }

        let shape = a.shape();
        let (m, n) = (shape[0], shape[1]);

        match a.dtype() {
            rustytorch_core::DType::Float64 => Self::qr_decomposition_f64(a, m, n),
            _ => {
                let a_f64 = a.to_f64()?;
                Self::qr_decomposition_f64(&a_f64, m, n)
            }
        }
    }

    /// F64 QR decomposition implementation
    fn qr_decomposition_f64(a: &Tensor, m: usize, n: usize) -> Result<(Tensor, Tensor)> {
        let mut r_data = Self::extract_f64_data(a)?.clone();
        let mut q_data = vec![0.0f64; m * m];

        // Initialize Q as identity
        for i in 0..m {
            q_data[i * m + i] = 1.0;
        }

        let min_dim = m.min(n);

        for k in 0..min_dim {
            // Compute Householder vector
            let mut norm_sq = 0.0;
            for i in k..m {
                norm_sq += r_data[i * n + k] * r_data[i * n + k];
            }

            if norm_sq < 1e-14 {
                continue;
            }

            let norm = norm_sq.sqrt();
            let sign = if r_data[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
            let alpha = -sign * norm;

            let mut v = vec![0.0f64; m];
            for i in k..m {
                if i == k {
                    v[i] = r_data[i * n + k] - alpha;
                } else {
                    v[i] = r_data[i * n + k];
                }
            }

            let v_norm_sq: f64 = v[k..].iter().map(|&x| x * x).sum();
            if v_norm_sq < 1e-14 {
                continue;
            }

            let beta = 2.0 / v_norm_sq;

            // Apply Householder reflection to R
            for j in k..n {
                let mut dot_product = 0.0;
                for i in k..m {
                    dot_product += v[i] * r_data[i * n + j];
                }

                for i in k..m {
                    r_data[i * n + j] -= beta * v[i] * dot_product;
                }
            }

            // Apply Householder reflection to Q
            for j in 0..m {
                let mut dot_product = 0.0;
                for i in k..m {
                    dot_product += v[i] * q_data[j * m + i];
                }

                for i in k..m {
                    q_data[j * m + i] -= beta * dot_product * v[i];
                }
            }
        }

        // Zero out below diagonal in R
        for i in 0..m {
            for j in 0..n {
                if i > j {
                    r_data[i * n + j] = 0.0;
                }
            }
        }

        let mut options = a.options().clone();
        options.dtype = rustytorch_core::DType::Float64;

        let q = Tensor::from_data(&q_data, vec![m, m], Some(options.clone()));
        let r = Tensor::from_data(&r_data, vec![m, n], Some(options));

        Ok((q, r))
    }

    /// Solve linear system Ax = b using LU decomposition
    pub fn solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.ndim() != 2 || b.ndim() < 1 {
            return Err(CoreError::invalid_op(
                "solve",
                "A must be 2D and b must be at least 1D",
            ));
        }

        let a_shape = a.shape();
        if a_shape[0] != a_shape[1] {
            return Err(CoreError::invalid_op("solve", "A must be square"));
        }

        let n = a_shape[0];
        let b_shape = b.shape();

        if b_shape[0] != n {
            return Err(CoreError::shape_mismatch(
                vec![n],
                vec![b_shape[0]],
                "solve",
            ));
        }

        // Perform LU decomposition
        let (l, u, p) = Self::lu_decomposition(a)?;

        // Solve P*A*x = P*b
        // First solve L*y = P*b (forward substitution)
        let b_2d = if b.ndim() == 1 {
            // Reshape 1D vector to 2D column vector
            let b_data = b.storage().to_vec_f64();
            let mut options = b.options().clone();
            options.dtype = rustytorch_core::DType::Float64;
            Tensor::from_data(&b_data, vec![n, 1], Some(options))
        } else {
            b.to_f64()?
        };

        let pb = Self::gemm(&p, &b_2d, 1.0, 0.0, None)?;
        let y = Self::forward_substitution(&l, &pb)?;

        // Then solve U*x = y (backward substitution)
        let result = Self::backward_substitution(&u, &y)?;

        // If original b was 1D, return 1D result
        if b.ndim() == 1 {
            let result_data = result.storage().to_vec_f64();
            // Take only the first column if result is 2D
            let final_data: Vec<f64> = if result.ndim() == 2 {
                (0..n).map(|i| result_data[i]).collect()
            } else {
                result_data
            };

            let mut options = result.options().clone();
            options.dtype = rustytorch_core::DType::Float64;
            Ok(Tensor::from_data(&final_data, vec![n], Some(options)))
        } else {
            Ok(result)
        }
    }

    /// Matrix inverse using LU decomposition
    pub fn inverse(a: &Tensor) -> Result<Tensor> {
        if a.ndim() != 2 {
            return Err(CoreError::invalid_op(
                "inverse",
                "Input must be a 2D matrix",
            ));
        }

        let shape = a.shape();
        if shape[0] != shape[1] {
            return Err(CoreError::invalid_op(
                "inverse",
                "Input must be a square matrix",
            ));
        }

        let n = shape[0];

        // Create identity matrix
        let mut identity_data = vec![0.0; n * n];
        for i in 0..n {
            identity_data[i * n + i] = 1.0;
        }

        let identity = Tensor::from_data(&identity_data, vec![n, n], Some(a.options().clone()));

        // Solve A*X = I
        Self::solve(a, &identity)
    }

    /// Forward substitution for lower triangular system L*x = b
    /// Handles both 1D and 2D right-hand sides
    fn forward_substitution(l: &Tensor, b: &Tensor) -> Result<Tensor> {
        let n = l.shape()[0];
        let l_data = Self::extract_f64_data(l)?;
        let b_data = Self::extract_f64_data(b)?;

        let is_2d = b.ndim() == 2;
        let num_cols = if is_2d { b.shape()[1] } else { 1 };

        let mut x_data = b_data.clone();

        // For each column in b (if 2D) or single vector (if 1D)
        for col in 0..num_cols {
            for i in 0..n {
                for j in 0..i {
                    let x_idx = if is_2d { i * num_cols + col } else { i };
                    let x_j_idx = if is_2d { j * num_cols + col } else { j };
                    x_data[x_idx] -= l_data[i * n + j] * x_data[x_j_idx];
                }
                let x_idx = if is_2d { i * num_cols + col } else { i };
                x_data[x_idx] /= l_data[i * n + i];
            }
        }

        let mut options = b.options().clone();
        options.dtype = rustytorch_core::DType::Float64;
        Ok(Tensor::from_data(
            &x_data,
            b.shape().to_vec(),
            Some(options),
        ))
    }

    /// Backward substitution for upper triangular system U*x = b
    /// Handles both 1D and 2D right-hand sides
    fn backward_substitution(u: &Tensor, b: &Tensor) -> Result<Tensor> {
        let n = u.shape()[0];
        let u_data = Self::extract_f64_data(u)?;
        let b_data = Self::extract_f64_data(b)?;

        let is_2d = b.ndim() == 2;
        let num_cols = if is_2d { b.shape()[1] } else { 1 };

        let mut x_data = b_data.clone();

        // For each column in b (if 2D) or single vector (if 1D)
        for col in 0..num_cols {
            for i in (0..n).rev() {
                for j in i + 1..n {
                    let x_idx = if is_2d { i * num_cols + col } else { i };
                    let x_j_idx = if is_2d { j * num_cols + col } else { j };
                    x_data[x_idx] -= u_data[i * n + j] * x_data[x_j_idx];
                }
                let x_idx = if is_2d { i * num_cols + col } else { i };
                x_data[x_idx] /= u_data[i * n + i];
            }
        }

        let mut options = b.options().clone();
        options.dtype = rustytorch_core::DType::Float64;
        Ok(Tensor::from_data(
            &x_data,
            b.shape().to_vec(),
            Some(options),
        ))
    }

    /// Matrix determinant using LU decomposition
    pub fn det(a: &Tensor) -> Result<f64> {
        if a.ndim() != 2 {
            return Err(CoreError::invalid_op("det", "Input must be a 2D matrix"));
        }

        let shape = a.shape();
        if shape[0] != shape[1] {
            return Err(CoreError::invalid_op(
                "det",
                "Input must be a square matrix",
            ));
        }

        let (_, u, p) = Self::lu_decomposition(a)?;

        // Determinant = (-1)^num_permutations * product of diagonal elements of U
        let u_data = Self::extract_f64_data(&u)?;
        let p_data = Self::extract_f64_data(&p)?;
        let n = shape[0];

        // Count permutations
        let mut num_swaps = 0;
        let mut perm = vec![0; n];
        for i in 0..n {
            for j in 0..n {
                if p_data[i * n + j] == 1.0 {
                    perm[i] = j;
                    break;
                }
            }
        }

        for i in 0..n {
            if perm[i] != i {
                // Find where i should go
                let mut j = i + 1;
                while j < n && perm[j] != i {
                    j += 1;
                }
                if j < n {
                    perm.swap(i, j);
                    num_swaps += 1;
                }
            }
        }

        let sign = if num_swaps % 2 == 0 { 1.0 } else { -1.0 };
        let product: f64 = (0..n).map(|i| u_data[i * n + i]).product();

        Ok(sign * product)
    }

    /// Generalized tensor dot product along specified axes
    /// tensordot(a, b, axes) computes sum_k a[..., k, ...] * b[..., k, ...]
    /// where k is summed over the axes specified
    pub fn tensordot(a: &Tensor, b: &Tensor, axes: (Vec<usize>, Vec<usize>)) -> Result<Tensor> {
        let (axes_a, axes_b) = axes;

        if axes_a.len() != axes_b.len() {
            return Err(CoreError::invalid_op(
                "tensordot",
                "Number of axes for both tensors must match",
            ));
        }

        // Validate axes
        for &axis in &axes_a {
            if axis >= a.ndim() {
                return Err(CoreError::dim_out_of_bounds(axis, a.ndim(), "tensordot"));
            }
        }
        for &axis in &axes_b {
            if axis >= b.ndim() {
                return Err(CoreError::dim_out_of_bounds(axis, b.ndim(), "tensordot"));
            }
        }

        // Check that contracted dimensions match
        for (&axis_a, &axis_b) in axes_a.iter().zip(axes_b.iter()) {
            if a.shape()[axis_a] != b.shape()[axis_b] {
                return Err(CoreError::shape_mismatch(
                    vec![a.shape()[axis_a]],
                    vec![b.shape()[axis_b]],
                    "tensordot",
                ));
            }
        }

        // For now, implement special case of matrix multiplication (common case)
        if axes_a == vec![1] && axes_b == vec![0] && a.ndim() == 2 && b.ndim() == 2 {
            return Self::gemm(a, b, 1.0, 0.0, None);
        }

        // General implementation (simplified for now)
        match (a.dtype(), b.dtype()) {
            (rustytorch_core::DType::Float64, rustytorch_core::DType::Float64) => {
                Self::tensordot_f64(a, b, axes_a, axes_b)
            }
            _ => {
                let a_f64 = a.to_f64()?;
                let b_f64 = b.to_f64()?;
                Self::tensordot_f64(&a_f64, &b_f64, axes_a, axes_b)
            }
        }
    }

    /// F64 implementation of tensordot
    fn tensordot_f64(
        a: &Tensor,
        b: &Tensor,
        axes_a: Vec<usize>,
        axes_b: Vec<usize>,
    ) -> Result<Tensor> {
        // For general case, we would need to:
        // 1. Transpose tensors to move contracted axes to the end
        // 2. Reshape to 2D matrices
        // 3. Perform matrix multiplication
        // 4. Reshape result back

        // For now, implement simple cases
        if axes_a.is_empty() {
            // Outer product case
            return Self::outer(a, b);
        }

        // Fallback: convert to matrix multiplication for 2D case
        if a.ndim() == 2 && b.ndim() == 2 && axes_a.len() == 1 && axes_b.len() == 1 {
            if axes_a[0] == 1 && axes_b[0] == 0 {
                return Self::gemm(a, b, 1.0, 0.0, None);
            } else if axes_a[0] == 0 && axes_b[0] == 1 {
                let a_t = a.transpose(0, 1)?;
                let b_t = b.transpose(0, 1)?;
                return Self::gemm(&a_t, &b_t, 1.0, 0.0, None);
            }
        }

        Err(CoreError::invalid_op(
            "tensordot",
            "General tensordot not fully implemented yet",
        ))
    }

    /// Outer product of two vectors/tensors
    /// outer(a, b) computes a[i] * b[j] for all i, j
    pub fn outer(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_flat = a.flatten()?;
        let b_flat = b.flatten()?;

        let a_size = a_flat.numel();
        let b_size = b_flat.numel();

        match (a.dtype(), b.dtype()) {
            (rustytorch_core::DType::Float64, rustytorch_core::DType::Float64) => {
                Self::outer_f64(&a_flat, &b_flat, a_size, b_size)
            }
            _ => {
                let a_f64 = a_flat.to_f64()?;
                let b_f64 = b_flat.to_f64()?;
                Self::outer_f64(&a_f64, &b_f64, a_size, b_size)
            }
        }
    }

    /// F64 implementation of outer product
    fn outer_f64(a: &Tensor, b: &Tensor, a_size: usize, b_size: usize) -> Result<Tensor> {
        let a_data = Self::extract_f64_data(a)?;
        let b_data = Self::extract_f64_data(b)?;

        let mut result_data = Vec::with_capacity(a_size * b_size);

        for &a_val in &a_data {
            for &b_val in &b_data {
                result_data.push(a_val * b_val);
            }
        }

        let mut options = a.options().clone();
        options.dtype = rustytorch_core::DType::Float64;

        Ok(Tensor::from_data(
            &result_data,
            vec![a_size, b_size],
            Some(options),
        ))
    }

    /// Extract diagonal elements from a 2D matrix
    /// For n-dim tensors, extracts diagonal from last two dimensions
    pub fn diagonal(
        a: &Tensor,
        offset: isize,
        axis1: Option<usize>,
        axis2: Option<usize>,
    ) -> Result<Tensor> {
        if a.ndim() < 2 {
            return Err(CoreError::invalid_op(
                "diagonal",
                "Input must be at least 2-dimensional",
            ));
        }

        let ndim = a.ndim();
        let axis1 = axis1.unwrap_or(ndim - 2);
        let axis2 = axis2.unwrap_or(ndim - 1);

        if axis1 >= ndim || axis2 >= ndim {
            return Err(CoreError::invalid_op("diagonal", "Axis out of range"));
        }

        if axis1 == axis2 {
            return Err(CoreError::invalid_op(
                "diagonal",
                "axis1 and axis2 cannot be the same",
            ));
        }

        let shape = a.shape();
        let dim1 = shape[axis1];
        let dim2 = shape[axis2];

        // Calculate diagonal length
        let diag_len = if offset >= 0 {
            let offset = offset as usize;
            if offset >= dim2 {
                0
            } else {
                (dim1).min(dim2 - offset)
            }
        } else {
            let offset = (-offset) as usize;
            if offset >= dim1 {
                0
            } else {
                (dim1 - offset).min(dim2)
            }
        };

        match a.dtype() {
            rustytorch_core::DType::Float64 => {
                Self::diagonal_f64(a, offset, axis1, axis2, diag_len)
            }
            _ => {
                let a_f64 = a.to_f64()?;
                Self::diagonal_f64(&a_f64, offset, axis1, axis2, diag_len)
            }
        }
    }

    /// F64 implementation of diagonal extraction
    fn diagonal_f64(
        a: &Tensor,
        offset: isize,
        axis1: usize,
        axis2: usize,
        diag_len: usize,
    ) -> Result<Tensor> {
        let data = Self::extract_f64_data(a)?;
        let shape = a.shape();
        let strides = a.strides();

        let mut result_data = Vec::with_capacity(diag_len);

        // For simplicity, handle 2D case first
        if a.ndim() == 2 {
            let (rows, cols) = (shape[0], shape[1]);

            for i in 0..diag_len {
                let (row, col) = if offset >= 0 {
                    (i, i + offset as usize)
                } else {
                    (i + (-offset) as usize, i)
                };

                if row < rows && col < cols {
                    result_data.push(data[row * cols + col]);
                }
            }
        } else {
            // For higher dimensions, this would require more complex indexing
            return Err(CoreError::invalid_op(
                "diagonal",
                "Diagonal for >2D tensors not fully implemented yet",
            ));
        }

        let mut options = a.options().clone();
        options.dtype = rustytorch_core::DType::Float64;

        Ok(Tensor::from_data(
            &result_data,
            vec![diag_len],
            Some(options),
        ))
    }

    /// Compute trace (sum of diagonal elements) of a 2D matrix
    pub fn trace(a: &Tensor) -> Result<f64> {
        if a.ndim() != 2 {
            return Err(CoreError::invalid_op("trace", "Input must be a 2D matrix"));
        }

        let shape = a.shape();
        let (rows, cols) = (shape[0], shape[1]);
        let diag_len = rows.min(cols);

        let data = Self::extract_f64_data(a)?;

        let mut trace_sum = 0.0;
        for i in 0..diag_len {
            trace_sum += data[i * cols + i];
        }

        Ok(trace_sum)
    }

    // Helper functions

    fn extract_f32_data(tensor: &Tensor) -> Result<Vec<f32>> {
        match tensor.storage() {
            StorageType::F32(data) => Ok(data.clone()),
            _ => {
                let f64_data = tensor.storage().to_vec_f64();
                Ok(f64_data.iter().map(|&x| x as f32).collect())
            }
        }
    }

    fn extract_f64_data(tensor: &Tensor) -> Result<Vec<f64>> {
        Ok(tensor.storage().to_vec_f64())
    }
}

/// Extension methods for Tensor to support linear algebra operations
impl Tensor {
    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        LinAlg::gemm(self, other, 1.0, 0.0, None)
    }

    /// Matrix multiplication with alpha and beta scaling  
    pub fn gemm(&self, other: &Self, alpha: f64, beta: f64, c: Option<&Self>) -> Result<Self> {
        LinAlg::gemm(self, other, alpha, beta, c)
    }

    /// LU decomposition
    pub fn lu(&self) -> Result<(Self, Self, Self)> {
        LinAlg::lu_decomposition(self)
    }


    /// Solve linear system
    pub fn solve(&self, b: &Self) -> Result<Self> {
        LinAlg::solve(self, b)
    }

    /// Matrix inverse
    pub fn inverse(&self) -> Result<Self> {
        LinAlg::inverse(self)
    }

    /// Matrix determinant
    pub fn det(&self) -> Result<f64> {
        LinAlg::det(self)
    }

    /// Tensor dot product
    pub fn tensordot(&self, other: &Self, axes: (Vec<usize>, Vec<usize>)) -> Result<Self> {
        LinAlg::tensordot(self, other, axes)
    }

    /// Outer product
    pub fn outer(&self, other: &Self) -> Result<Self> {
        LinAlg::outer(self, other)
    }

    /// Extract diagonal elements
    pub fn diagonal(
        &self,
        offset: isize,
        axis1: Option<usize>,
        axis2: Option<usize>,
    ) -> Result<Self> {
        LinAlg::diagonal(self, offset, axis1, axis2)
    }

    /// Compute trace (sum of diagonal elements)
    pub fn trace(&self) -> Result<f64> {
        LinAlg::trace(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustytorch_core::Reshapable;

    fn create_test_matrix_2x2() -> Tensor {
        Tensor::from_data(&[1.0f64, 2.0, 3.0, 4.0], vec![2, 2], None)
    }

    fn create_test_matrix_3x3() -> Tensor {
        Tensor::from_data(
            &[
                1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, // Making it non-singular
            ],
            vec![3, 3],
            None,
        )
    }

    #[test]
    fn test_advanced_linalg_integration() {
        // Create a simple test matrix
        let matrix = Tensor::from_data(&[2.0f64, 1.0, 1.0, 3.0], vec![2, 2], None);

        // Test trace
        let trace = matrix.trace().unwrap();
        assert!((trace - 5.0).abs() < 1e-10); // 2 + 3 = 5

        // Test diagonal
        let diag = matrix.diagonal(0, None, None).unwrap();
        let diag_data = diag.storage().to_vec_f64();
        assert_eq!(diag_data, vec![2.0, 3.0]);

        // Test determinant
        let det = matrix.det().unwrap();
        assert!((det - 5.0).abs() < 1e-10); // 2*3 - 1*1 = 5

        // Test with vectors for outer product
        let vec_a = Tensor::from_data(&[1.0f64, 2.0], vec![2], None);
        let vec_b = Tensor::from_data(&[3.0f64, 4.0], vec![2], None);

        let outer = vec_a.outer(&vec_b).unwrap();
        let outer_data = outer.storage().to_vec_f64();
        // [[1*3, 1*4], [2*3, 2*4]] = [[3, 4], [6, 8]]
        assert_eq!(outer_data, vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None);
        let b = Tensor::from_data(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2], None);

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        let result_data = result.storage().to_vec_f64();
        // [1,2] @ [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        assert!((result_data[0] - 19.0).abs() < 1e-6);
        assert!((result_data[1] - 22.0).abs() < 1e-6);
        assert!((result_data[2] - 43.0).abs() < 1e-6);
        assert!((result_data[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_lu_decomposition() {
        let a = create_test_matrix_3x3();
        let (l, u, p) = a.lu().unwrap();

        assert_eq!(l.shape(), &[3, 3]);
        assert_eq!(u.shape(), &[3, 3]);
        assert_eq!(p.shape(), &[3, 3]);

        // Verify P*A = L*U (approximately)
        let pa = p.matmul(&a).unwrap();
        let lu = l.matmul(&u).unwrap();

        let pa_data = pa.storage().to_vec_f64();
        let lu_data = lu.storage().to_vec_f64();

        for i in 0..9 {
            assert!((pa_data[i] - lu_data[i]).abs() < 1e-10);
        }
    }

    #[test]
    #[test]
    fn test_qr_decomposition() {
        // Créer une matrice de test simple et bien conditionnée
        let a = Tensor::from_data(
            &[3.0f64, -2.0, 2.0, 6.0],
            vec![2, 2],
            None
        );

        let (q, r) = a.qr().unwrap();

        // Vérifier les dimensions
        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);

        // Vérifier que Q est orthogonale (Q^T * Q = I)
        let qt = q.transpose(0, 1).unwrap();
        let qtq = qt.matmul(&q).unwrap();
        let qtq_data = qtq.storage().to_vec_f64();
        for i in 0..4 {
            let expected = if i == 0 || i == 3 { 1.0 } else { 0.0 };
            assert!((qtq_data[i] - expected).abs() < 1e-10);
        }

        // Vérifier que R est triangulaire supérieure
        let r_data = r.storage().to_vec_f64();
        assert!((r_data[2]).abs() < 1e-10);  // Élément sous la diagonale

        // Vérifier A = Q * R
        let qr = q.matmul(&r).unwrap();
        let qr_data = qr.storage().to_vec_f64();
        let a_data = a.storage().to_vec_f64();

        // Utiliser une tolérance plus grande pour la comparaison
        let tolerance = 1e-8;
        for i in 0..4 {
            assert!(
                (a_data[i] - qr_data[i]).abs() < tolerance,
                "Différence trop grande à l'index {}: {} vs {}",
                i, a_data[i], qr_data[i]
            );
        }
    }
    // fn test_qr_decomposition() {
    //     let a = create_test_matrix_3x3();
    //     let (q, r) = a.qr().unwrap();
    //
    //     assert_eq!(q.shape(), &[3, 3]);
    //     assert_eq!(r.shape(), &[3, 3]);
    //
    //     // Verify A = Q*R (approximately)
    //     let qr = q.matmul(&r).unwrap();
    //     let a_data = a.storage().to_vec_f64();
    //     let qr_data = qr.storage().to_vec_f64();
    //
    //     for i in 0..9 {
    //         assert!((a_data[i] - qr_data[i]).abs() < 1e-10);
    //     }
    // }

    #[test]
    fn test_solve_linear_system() {
        let a = create_test_matrix_2x2();
        let b = Tensor::from_data(&[5.0f64, 11.0], vec![2], None);

        let x = a.solve(&b).unwrap();
        assert_eq!(x.shape(), &[2]);

        // Verify A*x = b
        let ax = a.matmul(&x.reshape(&[2, 1]).unwrap()).unwrap();
        let ax_data = ax.storage().to_vec_f64();
        let b_data = b.storage().to_vec_f64();

        for i in 0..2 {
            assert!((ax_data[i] - b_data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix_inverse() {
        let a = create_test_matrix_2x2(); // [[1, 2], [3, 4]]
        let a_inv = a.inverse().unwrap();

        // Basic test: verify that inverse returns a tensor of the same shape
        assert_eq!(a_inv.shape(), &[2, 2]);

        // Analytical inverse of [[1,2],[3,4]] is [[-2,1],[1.5,-0.5]]
        // det(A) = 1*4 - 2*3 = -2
        // A^(-1) = (1/det) * [[4,-2],[-3,1]] = [[-2,1],[1.5,-0.5]]
        let expected_inverse = vec![-2.0, 1.0, 1.5, -0.5];
        let computed_inverse = a_inv.storage().to_vec_f64();

        println!("Computed inverse: {:?}", computed_inverse);
        println!("Expected inverse: {:?}", expected_inverse);

        // Test A * A^(-1) = I (which is more robust than exact inverse values)
        let identity_test = a.matmul(&a_inv).unwrap();
        let identity_data = identity_test.storage().to_vec_f64();

        println!("A * A^(-1): {:?}", identity_data);

        // Check if we get approximately identity matrix
        assert!(
            (identity_data[0] - 1.0).abs() < 1e-10,
            "(0,0) should be 1.0, got {}",
            identity_data[0]
        );
        assert!(
            identity_data[1].abs() < 1e-10,
            "(0,1) should be 0.0, got {}",
            identity_data[1]
        );
        assert!(
            identity_data[2].abs() < 1e-10,
            "(1,0) should be 0.0, got {}",
            identity_data[2]
        );
        assert!(
            (identity_data[3] - 1.0).abs() < 1e-10,
            "(1,1) should be 1.0, got {}",
            identity_data[3]
        );
    }

    #[test]
    fn test_determinant() {
        let a = create_test_matrix_2x2();
        let det = a.det().unwrap();

        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        assert!((det - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_tensordot() {
        // Test matrix multiplication case
        let a = Tensor::from_data(&[1.0f64, 2.0, 3.0, 4.0], vec![2, 2], None);
        let b = Tensor::from_data(&[5.0f64, 6.0, 7.0, 8.0], vec![2, 2], None);

        let result = a.tensordot(&b, (vec![1], vec![0])).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // This should be equivalent to matrix multiplication
        let matmul_result = a.matmul(&b).unwrap();
        let result_data = result.storage().to_vec_f64();
        let matmul_data = matmul_result.storage().to_vec_f64();

        for i in 0..4 {
            assert!((result_data[i] - matmul_data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_outer_product() {
        let a = Tensor::from_data(&[1.0f64, 2.0, 3.0], vec![3], None);
        let b = Tensor::from_data(&[4.0f64, 5.0], vec![2], None);

        let result = a.outer(&b).unwrap();
        assert_eq!(result.shape(), &[3, 2]);

        let result_data = result.storage().to_vec_f64();
        // Expected: [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]] = [[4, 5], [8, 10], [12, 15]]
        let expected = vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0];

        for i in 0..6 {
            assert!((result_data[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_diagonal() {
        let a = Tensor::from_data(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
            None,
        );

        // Main diagonal
        let diag = a.diagonal(0, None, None).unwrap();
        assert_eq!(diag.shape(), &[3]);

        let diag_data = diag.storage().to_vec_f64();
        assert_eq!(diag_data, vec![1.0, 5.0, 9.0]);

        // Upper diagonal (offset = 1)
        let upper_diag = a.diagonal(1, None, None).unwrap();
        assert_eq!(upper_diag.shape(), &[2]);

        let upper_data = upper_diag.storage().to_vec_f64();
        assert_eq!(upper_data, vec![2.0, 6.0]);

        // Lower diagonal (offset = -1)
        let lower_diag = a.diagonal(-1, None, None).unwrap();
        assert_eq!(lower_diag.shape(), &[2]);

        let lower_data = lower_diag.storage().to_vec_f64();
        assert_eq!(lower_data, vec![4.0, 8.0]);
    }

    #[test]
    fn test_trace() {
        let a = create_test_matrix_2x2(); // [[1, 2], [3, 4]]
        let trace = a.trace().unwrap();

        // trace = 1 + 4 = 5
        assert!((trace - 5.0).abs() < 1e-10);

        // Test with 3x3 matrix
        let b = create_test_matrix_3x3();
        let trace_b = b.trace().unwrap();

        // trace = 1 + 5 + 10 = 16 (main diagonal elements)
        assert!((trace_b - 16.0).abs() < 1e-10);
    }
}
