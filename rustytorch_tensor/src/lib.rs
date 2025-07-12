//rustytorch_tensor/src/lib.rs

use rustytorch_core::{CoreError, DType, Device, Reduction, Reshapable, Result, TensorOptions};

use rand::Rng;
use std::sync::Arc;
// use rustytorch_tensor::tensor_errors::TensorError;

// Public exports for initialization functionality
pub use initializers::{FanMode, Initializers, Nonlinearity};
// Public exports for decomposition functionality
pub use decompositions::Decompositions;

// use std::simd::f32x8;
use rayon::prelude::*;


pub mod broadcastings;
pub mod decompositions;
pub mod f16_support;
pub mod indexing;
pub mod initializers;
pub mod linalg;
pub mod memory_pool;
mod numeric_ops;
pub mod padding;
pub mod random_generators;
pub mod reductions;
pub mod simd_ops;
pub mod storage;
pub mod tensor_comparison;
mod tensor_errors;
pub mod tensor_ops;
pub mod tensor_optims;
pub mod tensor_view;
pub mod type_ops;

use storage::StorageType;
// use crate::tensor_errors::TensorError;
// use crate::tensor_errors::TensorErrorType::ShapeMismatch;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    storage: Arc<StorageType>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    options: TensorOptions,
}

impl Tensor {
    /// Crée un nouveau tenseur à partir d'un vecteur de données
    pub fn from_data<T: Into<f64> + Copy>(
        data: &[T],
        shape: Vec<usize>,
        options: Option<TensorOptions>,
    ) -> Self {
        let options = options.unwrap_or_default();
        let total_size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total_size,
            "Shape size mismatch with data length"
        );

        // Convertir les données en type approprié et créer le stockage
        let storage = match options.dtype {
            DType::Float16 => {
                // For now, store F16 as F32 internally
                let float_data: Vec<f32> = data.iter().map(|&v| v.into() as f32).collect();
                StorageType::from_f32(&float_data)
            }
            DType::Float32 => {
                let float_data: Vec<f32> = data.iter().map(|&v| v.into() as f32).collect();
                StorageType::from_f32(&float_data)
            }
            DType::Float64 => {
                let float_data: Vec<f64> = data.iter().map(|&v| v.into()).collect();
                StorageType::from_f64(&float_data)
            }
            DType::Int8 => {
                let int_data: Vec<i8> = data.iter().map(|&v| v.into() as i8).collect();
                StorageType::from_i8(&int_data)
            }
            DType::Int16 => {
                let int_data: Vec<i16> = data.iter().map(|&v| v.into() as i16).collect();
                StorageType::from_i16(&int_data)
            }
            DType::Int32 => {
                let int_data: Vec<i32> = data.iter().map(|&v| v.into() as i32).collect();
                StorageType::from_i32(&int_data)
            }
            DType::Int64 => {
                let int_data: Vec<i64> = data.iter().map(|&v| v.into() as i64).collect();
                StorageType::from_i64(&int_data)
            }
            DType::UInt8 => {
                let uint_data: Vec<u8> = data.iter().map(|&v| v.into() as u8).collect();
                StorageType::from_u8(&uint_data)
            }
            DType::UInt16 => {
                let uint_data: Vec<u16> = data.iter().map(|&v| v.into() as u16).collect();
                StorageType::from_u16(&uint_data)
            }
            DType::UInt32 => {
                let uint_data: Vec<u32> = data.iter().map(|&v| v.into() as u32).collect();
                StorageType::from_u32(&uint_data)
            }
            DType::UInt64 => {
                let uint_data: Vec<u64> = data.iter().map(|&v| v.into() as u64).collect();
                StorageType::from_u64(&uint_data)
            }
            DType::Bool => {
                let bool_data: Vec<bool> = data.iter().map(|&v| v.into() != 0.0).collect();
                StorageType::from_bool(&bool_data)
            }
            DType::Complex64 => {
                use num_complex::Complex;
                let complex_data: Vec<Complex<f32>> = data
                    .iter()
                    .map(|&v| Complex::new(v.into() as f32, 0.0))
                    .collect();
                StorageType::from_complex64(&complex_data)
            }
            DType::Complex128 => {
                use num_complex::Complex;
                let complex_data: Vec<Complex<f64>> =
                    data.iter().map(|&v| Complex::new(v.into(), 0.0)).collect();
                StorageType::from_complex128(&complex_data)
            }
        };

        // Calculer les strides (empreintes)
        let mut strides = vec![1; shape.len()];
        if shape.len() > 1 {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        Self {
            storage: Arc::new(storage),
            shape,
            strides,
            offset: 0,
            options,
        }
    }

    /// Crée un tenseur rempli de zéros
    pub fn zeros(shape: Vec<usize>, options: Option<TensorOptions>) -> Self {
        let total_size: usize = shape.iter().product();
        let zeros = vec![0.0; total_size];
        Self::from_data(&zeros, shape, options)
    }

    /// Crée un tenseur rempli de uns
    pub fn ones(shape: Vec<usize>, options: Option<TensorOptions>) -> Self {
        let total_size: usize = shape.iter().product();
        let ones = vec![1.0; total_size];
        Self::from_data(&ones, shape, options)
    }

    /// Creer un tenseur rempli de valeurs aléatoires uniformes
    pub fn rand(shape: Vec<usize>, options: Option<TensorOptions>) -> Self {
        let mut rng = rand::thread_rng();
        let total_size: usize = shape.iter().product();
        let random_data: Vec<f64> = (0..total_size).map(|_| rng.gen()).collect();
        Self::from_data(&random_data, shape, options)
    }
    
    /// Creer un tenseur rempli d'une valeur spécifique
    pub fn full<T>(shape: Vec<usize>, value: T, dtype: DType) -> Result<Self> 
    where
        T: Into<f64> + Copy,
    {
        let total_size: usize = shape.iter().product();
        let value_f64 = value.into();
        let data = vec![value_f64; total_size];
        let options = TensorOptions::new().dtype(dtype);
        Ok(Self::from_data(&data, shape, Some(options)))
    }

    /// renvoie la forme du tenseur (shape)
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// renvoie la Dimension du tenseur
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// renvoie le nombre d'éléments du tenseur
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Renvoie le type de données du tenseur
    pub fn dtype(&self) -> DType {
        self.options.dtype
    }

    /// Renvoie le type de stockage du tenseur
    pub fn storage(&self) -> &StorageType {
        &self.storage
    }

    /// Renvoie le device
    pub fn device(&self) -> &Device {
        &self.options.device
    }

    // Methods for tensor view support

    /// Get a reference to the storage
    pub fn storage_ref(&self) -> &Arc<StorageType> {
        &self.storage
    }

    /// Get the strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the offset
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the options
    pub fn options(&self) -> &TensorOptions {
        &self.options
    }

    /// Check if tensor is contiguous
    pub fn is_contiguous(&self) -> bool {
        // Check if strides match contiguous layout
        if self.shape.is_empty() {
            return true;
        }

        let mut expected_stride = 1;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }

    /// Create a view of this tensor
    pub fn view(&self) -> tensor_view::TensorView {
        tensor_view::TensorView::new(self)
    }

    /// Create a sliced view of this tensor
    pub fn slice_view(&self, ranges: &[std::ops::Range<usize>]) -> Result<tensor_view::TensorView> {
        let view = self.view();
        view.slice(ranges)
    }

    /// Select an index along a dimension, creating a view
    pub fn select_view(&self, dim: usize, index: usize) -> Result<tensor_view::TensorView> {
        let view = self.view();
        view.select(dim, index)
    }

    /// Create a narrow view along a dimension
    pub fn narrow_view(
        &self,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<tensor_view::TensorView> {
        let view = self.view();
        view.narrow(dim, start, length)
    }

    // New reduction operations

    /// Cumulative sum along axis
    pub fn cumsum(&self, axis: usize) -> Result<Tensor> {
        reductions::AxisReductions::cumsum(self, axis)
    }

    /// Cumulative product along axis
    pub fn cumprod(&self, axis: usize) -> Result<Tensor> {
        reductions::AxisReductions::cumprod(self, axis)
    }

    /// Compute norm of tensor
    pub fn norm(&self, ord: Option<f64>, dim: Option<&[usize]>, keep_dim: bool) -> Result<Tensor> {
        reductions::AxisReductions::norm(self, ord, dim, keep_dim)
    }

    /// Compute Frobenius norm (L2 norm of all elements)
    pub fn frobenius_norm(&self) -> Result<Tensor> {
        reductions::AxisReductions::frobenius_norm(self)
    }

    // Padding and cropping operations

    /// Apply padding to tensor
    pub fn pad(&self, spec: &padding::PaddingSpec) -> Result<Tensor> {
        padding::PaddingOps::pad(self, spec)
    }

    /// Crop tensor to specified region
    pub fn crop(&self, start: &[usize], end: &[usize]) -> Result<Tensor> {
        padding::PaddingOps::crop(self, start, end)
    }

    /// Center crop to specified size
    pub fn center_crop(&self, target_size: &[usize]) -> Result<Tensor> {
        padding::PaddingOps::center_crop(self, target_size)
    }

    /// Zero padding (shorthand for constant padding with 0)
    pub fn zero_pad(&self, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        let spec = padding::PaddingSpec::zeros(padding);
        self.pad(&spec)
    }

    /// Constant padding with specified value
    pub fn constant_pad(&self, padding: Vec<(usize, usize)>, value: f64) -> Result<Tensor> {
        let spec = padding::PaddingSpec::constant(padding, value);
        self.pad(&spec)
    }

    // === Matrix Decomposition Methods ===

    /// Compute Singular Value Decomposition (SVD)
    /// Returns (U, S, V) where A = U * diag(S) * V^T
    pub fn svd(&self, full_matrices: bool) -> Result<(Tensor, Tensor, Tensor)> {
        decompositions::Decompositions::svd(self, full_matrices)
    }

    /// Compute Cholesky decomposition
    /// Returns L (lower triangular) or U (upper triangular) where A = L*L^T or A = U^T*U
    pub fn cholesky(&self, upper: bool) -> Result<Tensor> {
        decompositions::Decompositions::cholesky(self, upper)
    }

    /// Compute QR decomposition
    /// Returns (Q, R) where A = Q * R with Q orthogonal and R upper triangular
    pub fn qr(&self) -> Result<(Tensor, Tensor)> {
        decompositions::Decompositions::qr(self)
    }
}

/// Implémentation NumericOps pour le tenseur

impl Reduction for Tensor {
    type Output = Tensor;
    type Axes = usize;

    fn sum(&self) -> Result<Self::Output> {
        reductions::AxisReductions::sum_dim(self, &[], false)
    }
    fn mean(&self) -> Result<Self::Output> {
        reductions::AxisReductions::mean_dim(self, &[], false)
    }

    fn max(&self) -> Result<Self::Output> {
        // Global max - use argmax to find it
        let argmax_result = reductions::AxisReductions::argmax(self, None, false)?;
        let max_idx = argmax_result.storage().get_f64(0).unwrap() as usize;
        let max_val = self.storage().get_f64(max_idx).unwrap();
        reductions::AxisReductions::create_scalar_tensor(max_val, self.options().clone())
    }

    fn min(&self) -> Result<Self::Output> {
        // Global min - use argmin to find it
        let argmin_result = reductions::AxisReductions::argmin(self, None, false)?;
        let min_idx = argmin_result.storage().get_f64(0).unwrap() as usize;
        let min_val = self.storage().get_f64(min_idx).unwrap();
        reductions::AxisReductions::create_scalar_tensor(min_val, self.options().clone())
    }

    // Advanced reduction methods using reductions module
    fn sum_dim(&self, dim: Self::Axes, keep_dim: bool) -> Result<Self::Output> {
        reductions::AxisReductions::sum_dim(self, &[dim], keep_dim)
    }

    fn mean_dim(&self, dim: Self::Axes, keep_dim: bool) -> Result<Self::Output> {
        reductions::AxisReductions::mean_dim(self, &[dim], keep_dim)
    }

    fn max_dim(&self, dim: Self::Axes, keep_dim: bool) -> Result<(Self::Output, Self::Output)> {
        reductions::AxisReductions::max_dim(self, dim, keep_dim)
    }

    fn min_dim(&self, dim: Self::Axes, keep_dim: bool) -> Result<(Self::Output, Self::Output)> {
        reductions::AxisReductions::min_dim(self, dim, keep_dim)
    }

    fn std(&self, unbiased: bool) -> Result<Self::Output> {
        let all_axes: Vec<usize> = (0..self.ndim()).collect();
        reductions::AxisReductions::std_dim(self, &all_axes, unbiased, false)
    }

    fn var(&self, unbiased: bool) -> Result<Self::Output> {
        let all_axes: Vec<usize> = (0..self.ndim()).collect();
        reductions::AxisReductions::var_dim(self, &all_axes, unbiased, false)
    }

    fn std_dim(&self, dim: Self::Axes, unbiased: bool, keep_dim: bool) -> Result<Self::Output> {
        reductions::AxisReductions::std_dim(self, &[dim], unbiased, keep_dim)
    }

    fn var_dim(&self, dim: Self::Axes, unbiased: bool, keep_dim: bool) -> Result<Self::Output> {
        reductions::AxisReductions::var_dim(self, &[dim], unbiased, keep_dim)
    }

    fn argmax(&self, dim: Option<Self::Axes>, keep_dim: bool) -> Result<Self::Output> {
        reductions::AxisReductions::argmax(self, dim, keep_dim)
    }

    fn argmin(&self, dim: Option<Self::Axes>, keep_dim: bool) -> Result<Self::Output> {
        reductions::AxisReductions::argmin(self, dim, keep_dim)
    }
}

impl Reshapable for Tensor {
    fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        //vérifier que le nombre total d'éléments est le même
        let new_size: usize = shape.iter().product();
        // assert_eq!(self.numel(),new_size, "Shape size n'est pas compatible avec le nombre d'éléments");
        if self.numel() != new_size {
            return Err(CoreError::shape_mismatch(
                vec![self.numel()],
                vec![new_size],
                "reshape",
            ));
        }
        // creer un nouveau tenseur avec la meme memoire mais avec une nouvelle forme
        let mut result = self.clone();
        result.shape = shape.to_vec();

        // Recalculte les strides
        let mut strides = vec![1; shape.len()];
        if shape.len() > 1 {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        result.strides = strides;

        Ok(result)
    }

    // flatten le tenseur
    fn flatten(&self) -> Result<Self> {
        self.reshape(&[self.numel()])
    }

    // transpose le tenseur
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(CoreError::dim_out_of_bounds(
                dim0.max(dim1),
                self.ndim(),
                "transpose",
            ));
        }
        
        if dim0 == dim1 {
            return Ok(self.clone());
        }

        // For now, we'll implement physical transpose (data rearrangement)
        // This ensures operations like matmul work correctly
        let shape = self.shape();
        let mut new_shape = shape.to_vec();
        new_shape.swap(dim0, dim1);

        // Get the current data
        let data = self.storage().to_vec_f64();

        // For 2D case (most common), implement direct transpose
        if self.ndim() == 2 && dim0 != dim1 {
            let rows = shape[0];
            let cols = shape[1];
            let mut transposed_data = vec![0.0; data.len()];

            // Transpose the data: A[i][j] -> A^T[j][i]
            for i in 0..rows {
                for j in 0..cols {
                    transposed_data[j * rows + i] = data[i * cols + j];
                }
            }

            // Create new tensor with transposed data
            return Ok(Tensor::from_data(&transposed_data, new_shape, Some(self.options().clone())));
        }

        // For higher dimensions, fall back to stride-based approach for now
        let mut result = self.clone();
        result.shape.swap(dim0, dim1);
        result.strides.swap(dim0, dim1);
        Ok(result)
    }

    // Missing methods from Reshapable trait
    fn permute(&self, _dims: &[usize]) -> Result<Self> {
        Err(CoreError::invalid_op("permute", "not implemented yet"))
    }

    fn squeeze(&self, _dim: Option<usize>) -> Result<Self> {
        Err(CoreError::invalid_op("squeeze", "not implemented yet"))
    }

    fn unsqueeze(&self, _dim: usize) -> Result<Self> {
        Err(CoreError::invalid_op("unsqueeze", "not implemented yet"))
    }

    fn view(&self, _shape: &[isize]) -> Result<Self> {
        Err(CoreError::invalid_op("view", "not implemented yet"))
    }

    fn broadcast_to(&self, _shape: &[usize]) -> Result<Self> {
        Err(CoreError::invalid_op("broadcast_to", "not implemented yet"))
    }
}

// Unit tests
// rustytorch_tensor/src/lib.rs (partie tests)

#[cfg(test)]
mod tests_tensor_operation {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(&data, vec![2, 3], None);

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_tensor_zeros_ones() {
        let zeros = Tensor::zeros(vec![2, 3], None);
        assert_eq!(zeros.shape(), &[2, 3]);

        let ones = Tensor::ones(vec![3, 2], None);
        assert_eq!(ones.shape(), &[3, 2]);
    }

    // #[test]
    // fn test_tensor_reshape() {
    //     let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    //     let tensor = Tensor::from_data(&data, vec![2, 3], None);
    //
    //     let reshaped = tensor.reshape(&[3, 2]);
    //     assert_eq!(reshaped.shape(), &[3, 2]);
    //     // assert_eq!(reshaped.shape(), &[3, 2]);
    //     // assert_eq!(reshaped.numel(), 6);
    //
    //     let flattened = tensor.flatten();
    //     assert_eq!(flattened.shape(), &[6]);
    // }
    //
    // #[test]
    // fn test_tensor_transpose() {
    //     let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    //     let tensor = Tensor::from_data(&data, vec![2, 3], None);
    //
    //     let transposed = tensor.transpose(0, 1);
    //     assert_eq!(transposed.shape(), &[3, 2]);
    // }

    // #[test]
    // fn test_add() {
    //     let a = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
    //     let b = Tensor::from_data(&[4.0, 5.0, 6.0], vec![3], None);
    //
    //     let c = a.clone().add(b.clone());
    //
    //     // Vérifier la forme
    //     assert_eq!(c.shape(), &[3]);
    //
    //     // Vérifier le contenu
    //     match c.storage.as_ref() {
    //         StorageType::F32(data) => {
    //             assert_eq!(data, &[5.0, 7.0, 9.0]);
    //         },
    //         StorageType::F64(data) => {
    //             assert_eq!(data, &[5.0, 7.0, 9.0]);
    //         },
    //         _ => panic!("Unexpected storage type"),
    //     }
    // }

    #[test]
    fn test_broadcasting() {
        // Test de broadcasting: scalaire + vecteur
        let scalar = Tensor::from_data(&[5.0], vec![1], None);
        let vector = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);

        let result = match scalar.add_broadcast(&vector) {
            Ok(r) => r,
            Err(e) => panic!("Broadcasting failed: {}", e),
        };

        assert_eq!(result.shape(), &[3]);

        match result.storage.as_ref() {
            StorageType::F32(data) => {
                assert_eq!(data, &[6.0, 7.0, 8.0]);
            }
            StorageType::F64(data) => {
                assert_eq!(data, &[6.0, 7.0, 8.0]);
            }
            _ => panic!("Unexpected storage type"),
        }
    }

    #[test]
    fn test_matrix_multiplication() {
        // Matrice 2x3
        let a = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None);
        // Matrice 3x2
        let b = Tensor::from_data(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], None);

        let result = match a.matmul(&b) {
            Ok(r) => r,
            Err(e) => panic!("Matrix multiplication failed: {}", e),
        };

        // Le résultat devrait être une matrice 2x2
        assert_eq!(result.shape(), &[2, 2]);

        // Vérifier le contenu (multiplication matricielle)
        match result.storage.as_ref() {
            StorageType::F32(data) => {
                // [1, 2, 3] • [7, 8] = 1*7 + 2*9 + 3*11 = 58
                //           • [9, 10]   1*8 + 2*10 + 3*12 = 64
                // [4, 5, 6] • [11, 12] = 4*7 + 5*9 + 6*11 = 139
                //                        4*8 + 5*10 + 6*12 = 154
                assert_eq!(data[0], 58.0);
                assert_eq!(data[1], 64.0);
                assert_eq!(data[2], 139.0);
                assert_eq!(data[3], 154.0);
            }
            StorageType::F64(data) => {
                assert_eq!(data[0], 58.0);
                assert_eq!(data[1], 64.0);
                assert_eq!(data[2], 139.0);
                assert_eq!(data[3], 154.0);
            }
            _ => panic!("Unexpected storage type"),
        }
    }

    // #[test]
    // fn test_reduction_operations() {
    //     let tensor = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], None);
    //
    //     // Test sum
    //     let sum = tensor.sum();
    //     match sum.storage.as_ref() {
    //         StorageType::F32(data) => {
    //             assert_eq!(data[0], 10.0); // 1+2+3+4
    //         },
    //         StorageType::F64(data) => {
    //             assert_eq!(data[0], 10.0);
    //         },
    //         _ => panic!("Unexpected storage type"),
    //     }
    //
    //     // Test mean
    //     let mean = tensor.mean();
    //     match mean.storage.as_ref() {
    //         StorageType::F32(data) => {
    //             assert_eq!(data[0], 2.5); // (1+2+3+4)/4
    //             // assert_eq!(data[0], 2.5);
    //         },
    //         StorageType::F64(data) => {
    //             assert_eq!(data[0], 2.5);
    //         },
    //         _ => panic!("Unexpected storage type"),
    //     }
    //
    //     // Test max
    //     let max = tensor.max();
    //     match max.storage.as_ref() {
    //         StorageType::F32(data) => {
    //             assert_eq!(data[0], 4.0);
    //         },
    //         StorageType::F64(data) => {
    //             assert_eq!(data[0], 4.0);
    //         },
    //         _ => panic!("Unexpected storage type"),
    //     }
    //
    //     // Test min
    //     let min = tensor.min();
    //     match min.storage.as_ref() {
    //         StorageType::F32(data) => {
    //             assert_eq!(data[0], 1.0);
    //         },
    //         StorageType::F64(data) => {
    //             assert_eq!(data[0], 1.0);
    //         },
    //         _ => panic!("Unexpected storage type"),
    //     }
    // }

    // === Weight Initialization Methods ===

    /// Initialize tensor with Xavier/Glorot uniform distribution
    /// Suitable for tanh/sigmoid activations
    pub fn xavier_uniform(
        shape: Vec<usize>,
        gain: Option<f64>,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        initializers::Initializers::xavier_uniform(shape, gain, options)
    }

    /// Initialize tensor with Xavier/Glorot normal distribution
    /// Suitable for tanh/sigmoid activations
    pub fn xavier_normal(
        shape: Vec<usize>,
        gain: Option<f64>,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        initializers::Initializers::xavier_normal(shape, gain, options)
    }

    /// Initialize tensor with Kaiming/He uniform distribution
    /// Suitable for ReLU activations
    pub fn kaiming_uniform(
        shape: Vec<usize>,
        a: Option<f64>,
        mode: initializers::FanMode,
        nonlinearity: initializers::Nonlinearity,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        initializers::Initializers::kaiming_uniform(shape, a, mode, nonlinearity, options)
    }

    /// Initialize tensor with Kaiming/He normal distribution
    /// Suitable for ReLU activations
    pub fn kaiming_normal(
        shape: Vec<usize>,
        a: Option<f64>,
        mode: initializers::FanMode,
        nonlinearity: initializers::Nonlinearity,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        initializers::Initializers::kaiming_normal(shape, a, mode, nonlinearity, options)
    }

    /// Initialize tensor with orthogonal matrix
    /// Maintains orthogonality of linear transformations
    pub fn orthogonal(
        shape: Vec<usize>,
        gain: Option<f64>,
        options: Option<TensorOptions>,
    ) -> Result<Tensor> {
        initializers::Initializers::orthogonal(shape, gain, options)
    }
}
