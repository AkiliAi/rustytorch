//! Optimized operations for tensor computations
//!
//! This module provides vectorized implementations using stable Rust
//! and will be extended with portable SIMD when it becomes stable.

use rayon::prelude::*;

/// Optimization threshold - use parallel processing for arrays larger than this
pub const PARALLEL_THRESHOLD: usize = 1000;

/// Optimized binary operations for f32
pub struct F32Ops;

impl F32Ops {
    /// Optimized element-wise addition
    pub fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if a.len() > PARALLEL_THRESHOLD {
            // Use parallel iterator for large arrays
            result
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(r, (&av, &bv))| *r = av + bv);
        } else {
            // Use SIMD-friendly sequential loop for smaller arrays
            Self::sequential_add(a, b, result);
        }
    }

    /// Sequential addition optimized for auto-vectorization
    #[inline]
    fn sequential_add(a: &[f32], b: &[f32], result: &mut [f32]) {
        // This loop pattern is optimized for LLVM auto-vectorization
        for i in 0..a.len() {
            unsafe {
                *result.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
            }
        }
    }

    /// Optimized element-wise subtraction
    pub fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if a.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(r, (&av, &bv))| *r = av - bv);
        } else {
            for i in 0..a.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = *a.get_unchecked(i) - *b.get_unchecked(i);
                }
            }
        }
    }

    /// Optimized element-wise multiplication
    pub fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if a.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(r, (&av, &bv))| *r = av * bv);
        } else {
            for i in 0..a.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
                }
            }
        }
    }

    /// Optimized element-wise division
    pub fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if a.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(r, (&av, &bv))| *r = av / bv);
        } else {
            for i in 0..a.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = *a.get_unchecked(i) / *b.get_unchecked(i);
                }
            }
        }
    }

    /// Optimized negation
    pub fn neg(input: &[f32], result: &mut [f32]) {
        assert_eq!(input.len(), result.len());

        if input.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(input.par_iter())
                .for_each(|(r, &v)| *r = -v);
        } else {
            for i in 0..input.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = -*input.get_unchecked(i);
                }
            }
        }
    }

    /// Optimized absolute value
    pub fn abs(input: &[f32], result: &mut [f32]) {
        assert_eq!(input.len(), result.len());

        if input.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(input.par_iter())
                .for_each(|(r, &v)| *r = v.abs());
        } else {
            for i in 0..input.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = input.get_unchecked(i).abs();
                }
            }
        }
    }

    /// Optimized sum reduction
    pub fn sum(input: &[f32]) -> f32 {
        if input.len() > PARALLEL_THRESHOLD {
            input.par_iter().sum()
        } else {
            // Use Kahan summation for better numerical stability
            let mut sum = 0.0f32;
            let mut c = 0.0f32; // Compensation for lost low-order bits

            for &x in input {
                let y = x - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            sum
        }
    }

    /// Optimized min reduction
    pub fn min(input: &[f32]) -> f32 {
        if input.is_empty() {
            return f32::NAN;
        }

        if input.len() > PARALLEL_THRESHOLD {
            input
                .par_iter()
                .fold(|| f32::INFINITY, |acc, &x| acc.min(x))
                .reduce(|| f32::INFINITY, |a, b| a.min(b))
        } else {
            let mut min = input[0];
            for &x in &input[1..] {
                min = min.min(x);
            }
            min
        }
    }

    /// Optimized max reduction
    pub fn max(input: &[f32]) -> f32 {
        if input.is_empty() {
            return f32::NAN;
        }

        if input.len() > PARALLEL_THRESHOLD {
            input
                .par_iter()
                .fold(|| f32::NEG_INFINITY, |acc, &x| acc.max(x))
                .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b))
        } else {
            let mut max = input[0];
            for &x in &input[1..] {
                max = max.max(x);
            }
            max
        }
    }

    /// Optimized mean calculation
    pub fn mean(input: &[f32]) -> f32 {
        if input.is_empty() {
            return f32::NAN;
        }
        Self::sum(input) / input.len() as f32
    }
}

/// Optimized binary operations for f64
pub struct F64Ops;

impl F64Ops {
    /// Optimized element-wise addition
    pub fn add(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if a.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(r, (&av, &bv))| *r = av + bv);
        } else {
            for i in 0..a.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
                }
            }
        }
    }

    /// Optimized element-wise subtraction
    pub fn sub(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if a.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(r, (&av, &bv))| *r = av - bv);
        } else {
            for i in 0..a.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = *a.get_unchecked(i) - *b.get_unchecked(i);
                }
            }
        }
    }

    /// Optimized element-wise multiplication
    pub fn mul(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if a.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(r, (&av, &bv))| *r = av * bv);
        } else {
            for i in 0..a.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
                }
            }
        }
    }

    /// Optimized element-wise division
    pub fn div(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());

        if a.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(r, (&av, &bv))| *r = av / bv);
        } else {
            for i in 0..a.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = *a.get_unchecked(i) / *b.get_unchecked(i);
                }
            }
        }
    }

    /// Optimized negation
    pub fn neg(input: &[f64], result: &mut [f64]) {
        assert_eq!(input.len(), result.len());

        if input.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(input.par_iter())
                .for_each(|(r, &v)| *r = -v);
        } else {
            for i in 0..input.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = -*input.get_unchecked(i);
                }
            }
        }
    }

    /// Optimized absolute value
    pub fn abs(input: &[f64], result: &mut [f64]) {
        assert_eq!(input.len(), result.len());

        if input.len() > PARALLEL_THRESHOLD {
            result
                .par_iter_mut()
                .zip(input.par_iter())
                .for_each(|(r, &v)| *r = v.abs());
        } else {
            for i in 0..input.len() {
                unsafe {
                    *result.get_unchecked_mut(i) = input.get_unchecked(i).abs();
                }
            }
        }
    }

    /// Optimized sum reduction with Kahan summation
    pub fn sum(input: &[f64]) -> f64 {
        if input.len() > PARALLEL_THRESHOLD {
            // For parallel case, we still use Kahan summation in chunks
            input.par_chunks(1000).map(Self::kahan_sum).sum()
        } else {
            Self::kahan_sum(input)
        }
    }

    /// Kahan summation algorithm for better numerical precision
    fn kahan_sum(input: &[f64]) -> f64 {
        let mut sum = 0.0f64;
        let mut c = 0.0f64; // Compensation for lost low-order bits

        for &x in input {
            let y = x - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }

    /// Optimized min reduction
    pub fn min(input: &[f64]) -> f64 {
        if input.is_empty() {
            return f64::NAN;
        }

        if input.len() > PARALLEL_THRESHOLD {
            input
                .par_iter()
                .fold(|| f64::INFINITY, |acc, &x| acc.min(x))
                .reduce(|| f64::INFINITY, |a, b| a.min(b))
        } else {
            let mut min = input[0];
            for &x in &input[1..] {
                min = min.min(x);
            }
            min
        }
    }

    /// Optimized max reduction
    pub fn max(input: &[f64]) -> f64 {
        if input.is_empty() {
            return f64::NAN;
        }

        if input.len() > PARALLEL_THRESHOLD {
            input
                .par_iter()
                .fold(|| f64::NEG_INFINITY, |acc, &x| acc.max(x))
                .reduce(|| f64::NEG_INFINITY, |a, b| a.max(b))
        } else {
            let mut max = input[0];
            for &x in &input[1..] {
                max = max.max(x);
            }
            max
        }
    }

    /// Optimized mean calculation
    pub fn mean(input: &[f64]) -> f64 {
        if input.is_empty() {
            return f64::NAN;
        }
        Self::sum(input) / input.len() as f64
    }
}

/// Block-based matrix multiplication optimization
pub struct MatMulOps;

impl MatMulOps {
    const BLOCK_SIZE: usize = 64; // Cache-friendly block size

    /// Optimized matrix multiplication for f32
    pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        // Clear result matrix
        c.fill(0.0);

        // Use blocked matrix multiplication for better cache performance
        for bi in (0..m).step_by(Self::BLOCK_SIZE) {
            for bj in (0..n).step_by(Self::BLOCK_SIZE) {
                for bk in (0..k).step_by(Self::BLOCK_SIZE) {
                    let end_i = (bi + Self::BLOCK_SIZE).min(m);
                    let end_j = (bj + Self::BLOCK_SIZE).min(n);
                    let end_k = (bk + Self::BLOCK_SIZE).min(k);

                    for i in bi..end_i {
                        for j in bj..end_j {
                            let mut sum = 0.0f32;
                            for kk in bk..end_k {
                                sum += a[i * k + kk] * b[kk * n + j];
                            }
                            c[i * n + j] += sum;
                        }
                    }
                }
            }
        }
    }

    /// Optimized matrix multiplication for f64
    pub fn matmul_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        c.fill(0.0);

        for bi in (0..m).step_by(Self::BLOCK_SIZE) {
            for bj in (0..n).step_by(Self::BLOCK_SIZE) {
                for bk in (0..k).step_by(Self::BLOCK_SIZE) {
                    let end_i = (bi + Self::BLOCK_SIZE).min(m);
                    let end_j = (bj + Self::BLOCK_SIZE).min(n);
                    let end_k = (bk + Self::BLOCK_SIZE).min(k);

                    for i in bi..end_i {
                        for j in bj..end_j {
                            let mut sum = 0.0f64;
                            for kk in bk..end_k {
                                sum += a[i * k + kk] * b[kk * n + j];
                            }
                            c[i * n + j] += sum;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_add() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0];
        let mut result = [0.0; 5];
        let expected = [2.0, 3.0, 4.0, 5.0, 6.0];

        F32Ops::add(&a, &b, &mut result);

        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-6,
                "Mismatch at index {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_f32_sum() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = F32Ops::sum(&input);
        let expected = 15.0;

        assert!(
            (result - expected).abs() < 1e-6,
            "Sum mismatch: {} != {}",
            result,
            expected
        );
    }

    #[test]
    fn test_f32_min_max() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

        let min = F32Ops::min(&input);
        let max = F32Ops::max(&input);

        assert_eq!(min, 1.0);
        assert_eq!(max, 9.0);
    }

    #[test]
    fn test_matmul_f32() {
        // 2x3 * 3x2 = 2x2
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 matrix
        let mut c = [0.0; 4]; // 2x2 result

        MatMulOps::matmul_f32(&a, &b, &mut c, 2, 2, 3);

        let expected = [58.0, 64.0, 139.0, 154.0];

        for (i, (&res, &exp)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-6,
                "MatMul mismatch at index {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }
}
