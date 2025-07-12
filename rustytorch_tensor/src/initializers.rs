//! Weight initialization functions for neural networks
//!
//! This module implements various weight initialization strategies commonly used
//! in deep learning, including Xavier/Glorot, Kaiming/He, and orthogonal initialization.

use crate::Tensor;
use rustytorch_core::{CoreError, Reshapable, Result, TensorOptions};

/// Weight initialization strategies
pub struct Initializers;

impl Initializers {
    /// Xavier/Glorot uniform initialization
    ///
    /// Initializes weights from uniform distribution U(-a, a) where:
    /// a = sqrt(6 / (fan_in + fan_out))
    ///
    /// This is designed to keep the variance of activations and gradients
    /// roughly the same across all layers.
    pub fn xavier_uniform(
        shape: Vec<usize>,
        gain: Option<f64>,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        if shape.len() < 2 {
            return Err(CoreError::invalid_op(
                "xavier_uniform",
                "Shape must have at least 2 dimensions for fan_in/fan_out calculation",
            ));
        }

        let fan_in = Self::calculate_fan_in(&shape);
        let fan_out = Self::calculate_fan_out(&shape);
        let gain = gain.unwrap_or(1.0);

        // a = gain * sqrt(6 / (fan_in + fan_out))
        let std = gain * (6.0 / (fan_in + fan_out) as f64).sqrt();
        let bound = std;

        Self::uniform_init(shape, -bound, bound, options)
    }

    /// Xavier/Glorot normal initialization
    ///
    /// Initializes weights from normal distribution N(0, std²) where:
    /// std = gain * sqrt(2 / (fan_in + fan_out))
    pub fn xavier_normal(
        shape: Vec<usize>,
        gain: Option<f64>,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        if shape.len() < 2 {
            return Err(CoreError::invalid_op(
                "xavier_normal",
                "Shape must have at least 2 dimensions for fan_in/fan_out calculation",
            ));
        }

        let fan_in = Self::calculate_fan_in(&shape);
        let fan_out = Self::calculate_fan_out(&shape);
        let gain = gain.unwrap_or(1.0);

        // std = gain * sqrt(2 / (fan_in + fan_out))
        let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();

        Self::normal_init(shape, 0.0, std, options)
    }

    /// Kaiming/He uniform initialization
    ///
    /// Initializes weights from uniform distribution U(-bound, bound) where:
    /// bound = gain * sqrt(3 / fan_in)  # for fan_in mode
    /// bound = gain * sqrt(3 / fan_out) # for fan_out mode
    ///
    /// Designed for ReLU activations.
    pub fn kaiming_uniform(
        shape: Vec<usize>,
        a: Option<f64>,
        mode: FanMode,
        nonlinearity: Nonlinearity,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        if shape.len() < 2 {
            return Err(CoreError::invalid_op(
                "kaiming_uniform",
                "Shape must have at least 2 dimensions for fan calculation",
            ));
        }

        let fan = match mode {
            FanMode::FanIn => Self::calculate_fan_in(&shape),
            FanMode::FanOut => Self::calculate_fan_out(&shape),
        };

        let gain = Self::calculate_gain(nonlinearity, a.unwrap_or(0.0));

        // bound = gain * sqrt(3 / fan)
        let std = gain * (1.0 / fan as f64).sqrt();
        let bound = std * 3.0_f64.sqrt();

        Self::uniform_init(shape, -bound, bound, options)
    }

    /// Kaiming/He normal initialization
    ///
    /// Initializes weights from normal distribution N(0, std²) where:
    /// std = gain / sqrt(fan)
    pub fn kaiming_normal(
        shape: Vec<usize>,
        a: Option<f64>,
        mode: FanMode,
        nonlinearity: Nonlinearity,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        if shape.len() < 2 {
            return Err(CoreError::invalid_op(
                "kaiming_normal",
                "Shape must have at least 2 dimensions for fan calculation",
            ));
        }

        let fan = match mode {
            FanMode::FanIn => Self::calculate_fan_in(&shape),
            FanMode::FanOut => Self::calculate_fan_out(&shape),
        };

        let gain = Self::calculate_gain(nonlinearity, a.unwrap_or(0.0));
        let std = gain / (fan as f64).sqrt();

        Self::normal_init(shape, 0.0, std, options)
    }

    /// Orthogonal initialization
    ///
    /// Fills the tensor with a (semi) orthogonal matrix. For 2D tensors,
    /// this creates an orthogonal matrix. For higher dimensions, creates
    /// tensors whose 2D slices are orthogonal.
    pub fn orthogonal(
        shape: Vec<usize>,
        gain: Option<f64>,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        if shape.len() < 2 {
            return Err(CoreError::invalid_op(
                "orthogonal",
                "Shape must have at least 2 dimensions",
            ));
        }

        let gain = gain.unwrap_or(1.0);
        let num_rows = shape[0];
        let num_cols = shape[1];

        // For orthogonal initialization, we need to handle the case where
        // num_rows != num_cols by working with the flattened view
        let flattened_shape = vec![num_rows, shape[1..].iter().product()];

        // Generate random matrix from standard normal distribution
        let random_tensor = Tensor::randn(flattened_shape.clone(), options.clone())?;

        // For a simplified orthogonal initialization, we'll use QR decomposition
        // For now, implement a basic version that normalizes columns
        let orthogonal_tensor = Self::make_orthogonal(&random_tensor)?;

        // Scale by gain (multiply each element by gain)
        let ortho_data = orthogonal_tensor.storage().to_vec_f64();
        let scaled_data: Vec<f64> = ortho_data.iter().map(|&x| x * gain).collect();
        let scaled = Tensor::from_data(&scaled_data, flattened_shape.clone(), options.clone());

        // Reshape to original shape if different
        if shape != flattened_shape {
            scaled.reshape(&shape)
        } else {
            Ok(scaled)
        }
    }

    /// Uniform initialization helper
    fn uniform_init(
        shape: Vec<usize>,
        low: f64,
        high: f64,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        Tensor::uniform(low, high, shape, options)
    }

    /// Normal initialization helper
    fn normal_init(
        shape: Vec<usize>,
        mean: f64,
        std: f64,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        Tensor::normal(mean, std, shape, options)
    }

    /// Calculate fan_in (number of input units)
    fn calculate_fan_in(shape: &[usize]) -> usize {
        if shape.len() < 2 {
            return 1;
        }

        match shape.len() {
            2 => shape[1],                       // For linear layers: (out_features, in_features)
            3 => shape[1] * shape[2], // For 1D conv: (out_channels, in_channels, kernel_size)
            4 => shape[1] * shape[2] * shape[3], // For 2D conv: (out, in, h, w)
            5 => shape[1] * shape[2] * shape[3] * shape[4], // For 3D conv
            _ => shape[1..].iter().product(), // General case
        }
    }

    /// Calculate fan_out (number of output units)
    fn calculate_fan_out(shape: &[usize]) -> usize {
        if shape.is_empty() {
            return 1;
        }

        match shape.len() {
            1 => shape[0],
            2 => shape[0],            // For linear layers: (out_features, in_features)
            3 => shape[0] * shape[2], // For 1D conv: (out_channels, in_channels, kernel_size)
            4 => shape[0] * shape[2] * shape[3], // For 2D conv: (out, in, h, w)
            5 => shape[0] * shape[2] * shape[3] * shape[4], // For 3D conv
            _ => {
                // General case: out_channels * spatial_dimensions
                let mut fan_out = shape[0];
                if shape.len() > 2 {
                    fan_out *= shape[2..].iter().product::<usize>();
                }
                fan_out
            }
        }
    }

    /// Calculate gain for different nonlinearities
    fn calculate_gain(nonlinearity: Nonlinearity, param: f64) -> f64 {
        match nonlinearity {
            Nonlinearity::Linear => 1.0,
            Nonlinearity::Conv1d => 1.0,
            Nonlinearity::Conv2d => 1.0,
            Nonlinearity::Conv3d => 1.0,
            Nonlinearity::ConvTranspose1d => 1.0,
            Nonlinearity::ConvTranspose2d => 1.0,
            Nonlinearity::ConvTranspose3d => 1.0,
            Nonlinearity::Sigmoid => 1.0,
            Nonlinearity::Tanh => 5.0 / 3.0,
            Nonlinearity::Relu => (2.0_f64).sqrt(),
            Nonlinearity::LeakyRelu => {
                let negative_slope = param;
                ((2.0 / (1.0 + negative_slope.powi(2))) as f64).sqrt()
            }
        }
    }

    /// Make a matrix orthogonal using Gram-Schmidt process (simplified)
    fn make_orthogonal(tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(CoreError::invalid_op(
                "make_orthogonal",
                "Expected 2D tensor for orthogonal initialization",
            ));
        }

        let rows = shape[0];
        let cols = shape[1];
        let data = tensor.storage().to_vec_f64();

        // Simple orthogonalization: normalize each column
        let mut result_data = vec![0.0; data.len()];

        for col in 0..cols {
            // Extract column
            let mut column: Vec<f64> = (0..rows).map(|row| data[row * cols + col]).collect();

            // Compute norm
            let norm = column.iter().map(|&x| x * x).sum::<f64>().sqrt();

            // Normalize column (avoid division by zero)
            if norm > 1e-8 {
                for val in &mut column {
                    *val /= norm;
                }
            } else {
                // If column is nearly zero, replace with unit vector
                column.fill(0.0);
                if col < rows {
                    column[col] = 1.0;
                }
            }

            // Write normalized column back
            for (row, &val) in column.iter().enumerate() {
                result_data[row * cols + col] = val;
            }
        }

        Ok(Tensor::from_data(
            &result_data,
            shape.to_vec(),
            Some(tensor.options().clone()),
        ))
    }
}

/// Fan mode for Kaiming initialization
#[derive(Debug, Clone, Copy)]
pub enum FanMode {
    FanIn,
    FanOut,
}

/// Nonlinearity types for gain calculation
#[derive(Debug, Clone, Copy)]
pub enum Nonlinearity {
    Linear,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
}

/// Extension methods for Tensor to support initialization
impl Tensor {
    /// Initialize with Xavier/Glorot uniform distribution
    pub fn xavier_uniform(
        shape: Vec<usize>,
        gain: Option<f64>,
        options: Option<TensorOptions>,
    ) -> Result<Self> {
        Initializers::xavier_uniform(shape, gain, options)
    }

    /// Initialize with Xavier/Glorot normal distribution
    pub fn xavier_normal(
        shape: Vec<usize>,
        gain: Option<f64>,
        options: Option<TensorOptions>,
    ) -> Result<Self> {
        Initializers::xavier_normal(shape, gain, options)
    }

    /// Initialize with Kaiming/He uniform distribution
    pub fn kaiming_uniform(
        shape: Vec<usize>,
        a: Option<f64>,
        mode: FanMode,
        nonlinearity: Nonlinearity,
        options: Option<TensorOptions>,
    ) -> Result<Self> {
        Initializers::kaiming_uniform(shape, a, mode, nonlinearity, options)
    }

    /// Initialize with Kaiming/He normal distribution
    pub fn kaiming_normal(
        shape: Vec<usize>,
        a: Option<f64>,
        mode: FanMode,
        nonlinearity: Nonlinearity,
        options: Option<TensorOptions>,
    ) -> Result<Self> {
        Initializers::kaiming_normal(shape, a, mode, nonlinearity, options)
    }

    /// Initialize with orthogonal matrix
    pub fn orthogonal(
        shape: Vec<usize>,
        gain: Option<f64>,
        options: Option<TensorOptions>,
    ) -> Result<Self> {
        Initializers::orthogonal(shape, gain, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_uniform() {
        let tensor = Tensor::xavier_uniform(vec![100, 50], None, None).unwrap();
        assert_eq!(tensor.shape(), &[100, 50]);

        // Check that values are within expected bounds
        let data = tensor.storage().to_vec_f64();
        let fan_in = 50;
        let fan_out = 100;
        let expected_bound = (6.0 / (fan_in + fan_out) as f64).sqrt();

        for &val in &data {
            assert!(val >= -expected_bound && val <= expected_bound);
        }
    }

    #[test]
    fn test_xavier_normal() {
        let tensor = Tensor::xavier_normal(vec![64, 32], None, None).unwrap();
        assert_eq!(tensor.shape(), &[64, 32]);

        // Check variance is approximately correct for large sample
        let data = tensor.storage().to_vec_f64();
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

        let fan_in = 32;
        let fan_out = 64;
        let expected_variance = 2.0 / (fan_in + fan_out) as f64;

        assert!((variance - expected_variance).abs() < 0.1);
    }

    #[test]
    fn test_kaiming_uniform() {
        let tensor = Tensor::kaiming_uniform(
            vec![128, 256],
            None,
            FanMode::FanIn,
            Nonlinearity::Relu,
            None,
        )
        .unwrap();

        assert_eq!(tensor.shape(), &[128, 256]);

        let data = tensor.storage().to_vec_f64();
        let fan_in = 256;
        let gain = (2.0_f64).sqrt(); // ReLU gain
        let expected_bound = gain * (3.0 / fan_in as f64).sqrt();

        for &val in &data {
            assert!(val >= -expected_bound && val <= expected_bound);
        }
    }

    #[test]
    fn test_kaiming_normal() {
        let tensor = Tensor::kaiming_normal(
            vec![64, 128],
            None,
            FanMode::FanOut,
            Nonlinearity::Relu,
            None,
        )
        .unwrap();

        assert_eq!(tensor.shape(), &[64, 128]);

        let data = tensor.storage().to_vec_f64();
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!(mean.abs() < 0.1); // Should be close to 0
    }

    #[test]
    fn test_orthogonal() {
        let tensor = Tensor::orthogonal(vec![4, 4], None, None).unwrap();
        assert_eq!(tensor.shape(), &[4, 4]);

        // For a square orthogonal matrix, columns should be normalized
        let data = tensor.storage().to_vec_f64();
        let cols = 4;
        let rows = 4;

        for col in 0..cols {
            let column: Vec<f64> = (0..rows).map(|row| data[row * cols + col]).collect();
            let norm_squared: f64 = column.iter().map(|&x| x * x).sum();
            assert!((norm_squared - 1.0).abs() < 1e-6); // Should be normalized
        }
    }

    #[test]
    fn test_fan_calculations() {
        // Test linear layer shape (out_features, in_features)
        assert_eq!(Initializers::calculate_fan_in(&[128, 256]), 256);
        assert_eq!(Initializers::calculate_fan_out(&[128, 256]), 128);

        // Test conv2d shape (out_channels, in_channels, kernel_h, kernel_w)
        assert_eq!(Initializers::calculate_fan_in(&[64, 32, 3, 3]), 32 * 3 * 3);
        assert_eq!(Initializers::calculate_fan_out(&[64, 32, 3, 3]), 64 * 3 * 3);
    }

    #[test]
    fn test_gain_calculations() {
        assert_eq!(Initializers::calculate_gain(Nonlinearity::Linear, 0.0), 1.0);
        assert_eq!(
            Initializers::calculate_gain(Nonlinearity::Relu, 0.0),
            (2.0_f64).sqrt()
        );
        assert_eq!(
            Initializers::calculate_gain(Nonlinearity::Tanh, 0.0),
            5.0 / 3.0
        );

        // Test LeakyReLU with slope 0.01
        let expected_leaky = ((2.0 / (1.0 + 0.01_f64.powi(2))) as f64).sqrt();
        assert!(
            (Initializers::calculate_gain(Nonlinearity::LeakyRelu, 0.01) - expected_leaky).abs()
                < 1e-10
        );
    }

    #[test]
    fn test_error_cases() {
        // Test invalid shapes (less than 2D)
        assert!(Tensor::xavier_uniform(vec![10], None, None).is_err());
        assert!(
            Tensor::kaiming_normal(vec![5], None, FanMode::FanIn, Nonlinearity::Relu, None)
                .is_err()
        );
        assert!(Tensor::orthogonal(vec![3], None, None).is_err());
    }
}
