//! Advanced random number generation for tensors
//!
//! This module implements various random number generators commonly used in
//! machine learning and scientific computing applications.

use crate::Tensor;
use rand::{thread_rng, Rng};
use rand_distr::{Bernoulli, Distribution, Normal, StandardNormal, Uniform};
use rustytorch_core::{CoreError, DType, Result, TensorOptions};

/// Advanced random number generators
pub struct RandomGenerators;

impl RandomGenerators {
    /// Generate tensor with random numbers from standard normal distribution N(0, 1)
    /// Equivalent to PyTorch's torch.randn()
    pub fn randn(shape: Vec<usize>, options: Option<TensorOptions>) -> Result<Tensor> {
        let options = options.unwrap_or_default();
        let total_size: usize = shape.iter().product();

        let mut rng = thread_rng();
        let normal = StandardNormal;

        match options.dtype {
            DType::Float32 => {
                let data: Vec<f32> = (0..total_size).map(|_| normal.sample(&mut rng)).collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::Float64 => {
                let data: Vec<f64> = (0..total_size).map(|_| normal.sample(&mut rng)).collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            _ => {
                // For non-float types, generate as f64 then convert
                let data: Vec<f64> = (0..total_size).map(|_| normal.sample(&mut rng)).collect();
                let temp_tensor = Tensor::from_data(&data, shape.clone(), None);
                temp_tensor.to_dtype(options.dtype)
            }
        }
    }

    /// Generate tensor with random numbers from normal distribution N(mean, std²)
    /// Equivalent to PyTorch's torch.normal()
    pub fn normal(
        mean: f64,
        std: f64,
        shape: Vec<usize>,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        if std <= 0.0 {
            return Err(CoreError::invalid_op(
                "normal",
                "Standard deviation must be positive",
            ));
        }

        let options = options.unwrap_or_default();
        let total_size: usize = shape.iter().product();

        let mut rng = thread_rng();
        let normal = Normal::new(mean, std).map_err(|e| {
            CoreError::invalid_op("normal", &format!("Invalid normal distribution: {}", e))
        })?;

        match options.dtype {
            DType::Float32 => {
                let data: Vec<f32> = (0..total_size)
                    .map(|_| normal.sample(&mut rng) as f32)
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::Float64 => {
                let data: Vec<f64> = (0..total_size).map(|_| normal.sample(&mut rng)).collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            _ => {
                // For non-float types, generate as f64 then convert
                let data: Vec<f64> = (0..total_size).map(|_| normal.sample(&mut rng)).collect();
                let temp_tensor = Tensor::from_data(&data, shape.clone(), None);
                temp_tensor.to_dtype(options.dtype)
            }
        }
    }

    /// Generate tensor with random integers in range [low, high)
    /// Equivalent to PyTorch's torch.randint()
    pub fn randint(
        low: i64,
        high: i64,
        shape: Vec<usize>,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        if low >= high {
            return Err(CoreError::invalid_op(
                "randint",
                "low must be less than high",
            ));
        }

        let options = options.unwrap_or_default();
        let total_size: usize = shape.iter().product();

        let mut rng = thread_rng();
        let uniform = Uniform::new(low, high);

        match options.dtype {
            DType::Int8 => {
                let data: Vec<i8> = (0..total_size)
                    .map(|_| {
                        let val = uniform.sample(&mut rng);
                        if val >= i8::MIN as i64 && val <= i8::MAX as i64 {
                            val as i8
                        } else {
                            (val % (i8::MAX as i64 - i8::MIN as i64 + 1) + i8::MIN as i64) as i8
                        }
                    })
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::Int16 => {
                let data: Vec<i16> = (0..total_size)
                    .map(|_| {
                        let val = uniform.sample(&mut rng);
                        if val >= i16::MIN as i64 && val <= i16::MAX as i64 {
                            val as i16
                        } else {
                            (val % (i16::MAX as i64 - i16::MIN as i64 + 1) + i16::MIN as i64) as i16
                        }
                    })
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::Int32 => {
                let data: Vec<i32> = (0..total_size)
                    .map(|_| {
                        let val = uniform.sample(&mut rng);
                        if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
                            val as i32
                        } else {
                            (val % (i32::MAX as i64 - i32::MIN as i64 + 1) + i32::MIN as i64) as i32
                        }
                    })
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::Int64 => {
                let data: Vec<i64> = (0..total_size).map(|_| uniform.sample(&mut rng)).collect();
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let temp_tensor = Tensor::from_data(&data_f64, shape, None);
                temp_tensor.to_dtype(DType::Int64)
            }
            DType::UInt8 => {
                // For unsigned types, adjust range
                let low_u = if low < 0 { 0 } else { low as u64 };
                let high_u = if high < 0 { 0 } else { high as u64 };
                let uniform_u = Uniform::new(low_u, high_u);

                let data: Vec<u8> = (0..total_size)
                    .map(|_| {
                        let val = uniform_u.sample(&mut rng);
                        if val <= u8::MAX as u64 {
                            val as u8
                        } else {
                            (val % (u8::MAX as u64 + 1)) as u8
                        }
                    })
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::UInt16 => {
                let low_u = if low < 0 { 0 } else { low as u64 };
                let high_u = if high < 0 { 0 } else { high as u64 };
                let uniform_u = Uniform::new(low_u, high_u);

                let data: Vec<u16> = (0..total_size)
                    .map(|_| {
                        let val = uniform_u.sample(&mut rng);
                        if val <= u16::MAX as u64 {
                            val as u16
                        } else {
                            (val % (u16::MAX as u64 + 1)) as u16
                        }
                    })
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::UInt32 => {
                let low_u = if low < 0 { 0 } else { low as u64 };
                let high_u = if high < 0 { 0 } else { high as u64 };
                let uniform_u = Uniform::new(low_u, high_u);

                let data: Vec<u32> = (0..total_size)
                    .map(|_| {
                        let val = uniform_u.sample(&mut rng);
                        if val <= u32::MAX as u64 {
                            val as u32
                        } else {
                            (val % (u32::MAX as u64 + 1)) as u32
                        }
                    })
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::UInt64 => {
                let low_u = if low < 0 { 0 } else { low as u64 };
                let high_u = if high < 0 { 0 } else { high as u64 };
                let uniform_u = Uniform::new(low_u, high_u);

                let data: Vec<u64> = (0..total_size)
                    .map(|_| uniform_u.sample(&mut rng))
                    .collect();
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let temp_tensor = Tensor::from_data(&data_f64, shape, None);
                temp_tensor.to_dtype(DType::UInt64)
            }
            _ => {
                // For float types, generate integers then convert
                let data: Vec<i64> = (0..total_size).map(|_| uniform.sample(&mut rng)).collect();
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let temp_tensor = Tensor::from_data(&data_f64, shape.clone(), None);
                temp_tensor.to_dtype(options.dtype)
            }
        }
    }

    /// Generate tensor with random boolean values from Bernoulli distribution
    /// Equivalent to PyTorch's torch.bernoulli()
    pub fn bernoulli(p: f64, shape: Vec<usize>, options: Option<TensorOptions>) -> Result<Tensor> {
        if !(0.0..=1.0).contains(&p) {
            return Err(CoreError::invalid_op(
                "bernoulli",
                "Probability p must be between 0.0 and 1.0",
            ));
        }

        let options = options.unwrap_or_default();
        let total_size: usize = shape.iter().product();

        let mut rng = thread_rng();
        let bernoulli = Bernoulli::new(p).map_err(|e| {
            CoreError::invalid_op(
                "bernoulli",
                &format!("Invalid Bernoulli distribution: {}", e),
            )
        })?;

        match options.dtype {
            DType::Bool => {
                let data: Vec<bool> = (0..total_size)
                    .map(|_| bernoulli.sample(&mut rng))
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::Float32 => {
                let data: Vec<f32> = (0..total_size)
                    .map(|_| if bernoulli.sample(&mut rng) { 1.0 } else { 0.0 })
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::Float64 => {
                let data: Vec<f64> = (0..total_size)
                    .map(|_| if bernoulli.sample(&mut rng) { 1.0 } else { 0.0 })
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => {
                let data: Vec<i64> = (0..total_size)
                    .map(|_| if bernoulli.sample(&mut rng) { 1 } else { 0 })
                    .collect();
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let temp_tensor = Tensor::from_data(&data_f64, shape.clone(), None);
                temp_tensor.to_dtype(options.dtype)
            }
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => {
                let data: Vec<u64> = (0..total_size)
                    .map(|_| if bernoulli.sample(&mut rng) { 1 } else { 0 })
                    .collect();
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let temp_tensor = Tensor::from_data(&data_f64, shape.clone(), None);
                temp_tensor.to_dtype(options.dtype)
            }
            _ => {
                // For other types, generate as bool then convert
                let data: Vec<bool> = (0..total_size)
                    .map(|_| bernoulli.sample(&mut rng))
                    .collect();
                let temp_tensor = Tensor::from_data(&data, shape.clone(), None);
                temp_tensor.to_dtype(options.dtype)
            }
        }
    }

    /// Generate tensor with random uniform values in range [low, high)
    /// Equivalent to PyTorch's torch.uniform_()
    pub fn uniform(
        low: f64,
        high: f64,
        shape: Vec<usize>,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        if low >= high {
            return Err(CoreError::invalid_op(
                "uniform",
                "low must be less than high",
            ));
        }

        let options = options.unwrap_or_default();
        let total_size: usize = shape.iter().product();

        let mut rng = thread_rng();
        let uniform = Uniform::new(low, high);

        match options.dtype {
            DType::Float32 => {
                let data: Vec<f32> = (0..total_size)
                    .map(|_| uniform.sample(&mut rng) as f32)
                    .collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            DType::Float64 => {
                let data: Vec<f64> = (0..total_size).map(|_| uniform.sample(&mut rng)).collect();
                Ok(Tensor::from_data(&data, shape, Some(options)))
            }
            _ => {
                // For non-float types, generate as f64 then convert
                let data: Vec<f64> = (0..total_size).map(|_| uniform.sample(&mut rng)).collect();
                let temp_tensor = Tensor::from_data(&data, shape.clone(), None);
                temp_tensor.to_dtype(options.dtype)
            }
        }
    }

    /// Generate tensor with random multinomial samples
    /// Simplified version of PyTorch's torch.multinomial()
    pub fn multinomial(weights: &Tensor, num_samples: usize, replacement: bool) -> Result<Tensor> {
        if weights.ndim() != 1 {
            return Err(CoreError::invalid_op(
                "multinomial",
                "weights must be a 1D tensor",
            ));
        }

        let weight_data = weights.storage().to_vec_f64();
        let num_categories = weight_data.len();

        // Normalize weights to probabilities
        let sum: f64 = weight_data.iter().sum();
        if sum <= 0.0 {
            return Err(CoreError::invalid_op(
                "multinomial",
                "weights must sum to positive value",
            ));
        }

        let probabilities: Vec<f64> = weight_data.iter().map(|&w| w / sum).collect();

        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(num_samples);
        let mut available_indices: Vec<usize> = (0..num_categories).collect();

        for _ in 0..num_samples {
            if available_indices.is_empty() && !replacement {
                return Err(CoreError::invalid_op(
                    "multinomial",
                    "Not enough categories for sampling without replacement",
                ));
            }

            let random_val: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut selected_idx = 0;

            for (i, &prob_idx) in available_indices.iter().enumerate() {
                cumulative += probabilities[prob_idx];
                if random_val <= cumulative {
                    selected_idx = i;
                    break;
                }
            }

            let category = available_indices[selected_idx];
            samples.push(category as i64);

            if !replacement {
                available_indices.remove(selected_idx);
            }
        }

        let mut options = TensorOptions::default();
        options.dtype = DType::Int64;

        // Convert i64 to f64 for from_data compatibility
        let samples_f64: Vec<f64> = samples.into_iter().map(|x| x as f64).collect();
        let temp_tensor = Tensor::from_data(&samples_f64, vec![num_samples], None);
        temp_tensor.to_dtype(DType::Int64)
    }
}

/// Extension methods for Tensor to support random number generation
impl Tensor {
    /// Generate tensor with standard normal distribution N(0,1)
    pub fn randn(shape: Vec<usize>, options: Option<TensorOptions>) -> Result<Self> {
        RandomGenerators::randn(shape, options)
    }

    /// Generate tensor with normal distribution N(mean, std²)
    pub fn normal(
        mean: f64,
        std: f64,
        shape: Vec<usize>,
        options: Option<TensorOptions>,
    ) -> Result<Self> {
        RandomGenerators::normal(mean, std, shape, options)
    }

    /// Generate tensor with random integers in range [low, high)
    pub fn randint(
        low: i64,
        high: i64,
        shape: Vec<usize>,
        options: Option<TensorOptions>,
    ) -> Result<Self> {
        RandomGenerators::randint(low, high, shape, options)
    }

    /// Generate tensor with Bernoulli distribution (probability p)
    pub fn bernoulli(p: f64, shape: Vec<usize>, options: Option<TensorOptions>) -> Result<Self> {
        RandomGenerators::bernoulli(p, shape, options)
    }

    /// Generate tensor with uniform distribution in range [low, high)
    pub fn uniform(
        low: f64,
        high: f64,
        shape: Vec<usize>,
        options: Option<TensorOptions>,
    ) -> Result<Self> {
        RandomGenerators::uniform(low, high, shape, options)
    }

    /// Generate multinomial samples from this tensor (treated as weights)
    pub fn multinomial(&self, num_samples: usize, replacement: bool) -> Result<Self> {
        RandomGenerators::multinomial(self, num_samples, replacement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randn() {
        let tensor = Tensor::randn(vec![100], None).unwrap();
        assert_eq!(tensor.shape(), &[100]);
        assert_eq!(tensor.dtype(), DType::Float32);

        // Check that values are roughly normally distributed
        let data = tensor.storage().to_vec_f64();
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!((mean).abs() < 0.3); // Should be close to 0
    }

    #[test]
    fn test_normal() {
        let tensor = Tensor::normal(5.0, 2.0, vec![100], None).unwrap();
        assert_eq!(tensor.shape(), &[100]);

        let data = tensor.storage().to_vec_f64();
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!((mean - 5.0).abs() < 1.0); // Should be close to 5.0
    }

    #[test]
    fn test_randint() {
        let mut options = TensorOptions::default();
        options.dtype = DType::Int32;

        let tensor = Tensor::randint(0, 10, vec![50], Some(options)).unwrap();
        assert_eq!(tensor.shape(), &[50]);
        assert_eq!(tensor.dtype(), DType::Int32);

        // Check that all values are in range [0, 10)
        let data = tensor.storage().to_vec_f64();
        for &val in &data {
            assert!(val >= 0.0 && val < 10.0);
        }
    }

    #[test]
    fn test_bernoulli() {
        let mut options = TensorOptions::default();
        options.dtype = DType::Bool;

        let tensor = Tensor::bernoulli(0.5, vec![100], Some(options)).unwrap();
        assert_eq!(tensor.shape(), &[100]);
        assert_eq!(tensor.dtype(), DType::Bool);

        // Check that roughly half are true (with some tolerance)
        let data = tensor.storage().to_vec_f64();
        let true_count = data.iter().filter(|&&x| x != 0.0).count();
        assert!(true_count > 30 && true_count < 70); // Rough check for 50% ± 20%
    }

    #[test]
    fn test_uniform() {
        let tensor = Tensor::uniform(2.0, 8.0, vec![100], None).unwrap();
        assert_eq!(tensor.shape(), &[100]);

        let data = tensor.storage().to_vec_f64();
        for &val in &data {
            assert!(val >= 2.0 && val < 8.0);
        }
    }

    #[test]
    fn test_multinomial() {
        let weights = Tensor::from_data(&[1.0f64, 2.0, 3.0, 4.0], vec![4], None);
        let samples = weights.multinomial(10, true).unwrap();

        assert_eq!(samples.shape(), &[10]);
        assert_eq!(samples.dtype(), DType::Int64);

        let sample_data = samples.storage().to_vec_f64();
        for &sample in &sample_data {
            assert!(sample >= 0.0 && sample < 4.0);
        }
    }

    #[test]
    fn test_error_cases() {
        // Test invalid normal distribution
        assert!(Tensor::normal(0.0, -1.0, vec![10], None).is_err());

        // Test invalid randint range
        assert!(Tensor::randint(10, 5, vec![10], None).is_err());

        // Test invalid bernoulli probability
        assert!(Tensor::bernoulli(1.5, vec![10], None).is_err());

        // Test invalid uniform range
        assert!(Tensor::uniform(5.0, 2.0, vec![10], None).is_err());
    }
}
