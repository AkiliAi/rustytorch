//rustytorch_tensor/src/lib.rs


use rustytorch_core::{Dtype, TensorOptions, NumericOps, Reduction, Reshapable, Device};

use std::sync::Arc;
use rand::Rng;

// use std::simd::f32x8;
use rayon::prelude::*;


pub mod storage;
mod tensor_errors;
mod tensor_optims;
mod broadcastings;

use storage::StorageType;
use crate::tensor_errors::TensorError;
use crate::tensor_errors::TensorErrorType::ShapeMismatch;

#[derive(Clone,Debug,PartialEq,)]
pub struct Tensor {
    storage: Arc<StorageType>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    options : TensorOptions,

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
        assert_eq!(data.len(), total_size, "Shape size mismatch with data length");

        // Convertir les données en type approprié et créer le stockage
        let storage = match options.dtype {
            Dtype::Float32 => {
                let float_data: Vec<f32> = data.iter().map(|&v| v.into() as f32).collect();
                StorageType::from_f32(&float_data)
            },
            Dtype::Float64 => {
                let float_data: Vec<f64> = data.iter().map(|&v| v.into()).collect();
                StorageType::from_f64(&float_data)
            },
            // Autres types à implémenter...
            _ => unimplemented!("Type de données non supporté"),
        };

        // Calculer les strides (empreintes)
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len()-1).rev() {
            strides[i] = strides[i+1] * shape[i+1];
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
    pub fn rand(shape: Vec<usize>, options: Option<TensorOptions>) -> Self{

        let mut rng = rand::rng();
        let total_size: usize = shape.iter().product();
        let random_data: Vec<f64> = (0..total_size).map(|_| rng.random()).collect();
        Self::from_data(&random_data, shape, options)

    }

    /// renvoie la forme du tenseur (shape)
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// renvoie la Dimension du tenseur
    pub fn ndim(&self) -> usize{
        self.shape.len()
    }

    /// renvoie le nombre d'éléments du tenseur
    pub fn numel(&self) ->usize{
        self.shape.iter().product()
    }

    /// Renvoie le type de données du tenseur
    pub fn dtype(&self) -> Dtype {
        self.options.dtype
    }

    /// Renvoie le type de stockage du tenseur
    pub fn storage(&self) -> &StorageType {
        &self.storage
    }

    /// Renvoie le device
    pub fn device(&self) -> &Device{
        &self.options.device
    }


}


/// Implémentation NumericOps pour le tenseur

impl NumericOps for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        match self.add_broadcast(&rhs) {
            Ok(result) => result,
            Err(e) => panic!("Error in add Operation: {}", e),
            // Err() =>
        }
    }
    fn sub(self, rhs: Self) -> Self::Output {
        match self.sub_broadcast(&rhs) {
            Ok(result) => result,
            Err(e) => panic!("Error in sub Operation: {}", e),
        }

    }
    fn mul(self, rhs: Self) -> Self::Output {
        match self.mul_broadcast(&rhs) {
            Ok(result) => result,
            Err(e) =>panic!("Error in mul Operation {}",e),
        }

    }
    fn div(self, rhs: Self) -> Self::Output {
        match self.div_broadcast(&rhs) {
            Ok(result) =>result,
            Err(e) =>panic!("Error in div Operation {}",e),
        }
    }
}


impl Reduction for Tensor{
    type Output = Tensor;

    fn sum(&self) -> Self::Output {
        match self.sum_dim(None) {
            Ok(result) => result,
            Err(e) => panic!("Error in sum operation {}",e),
        }
    }
    fn mean(&self) -> Self::Output {
        match self.mean_dim(None) {
            Ok(result) =>result,
            Err(e) => panic!("Error in mean operation {}", e),
        }
    }

    fn max(&self) -> Self::Output {
        match self.max_dim(None) {
            Ok(result) => result,
            Err(e) => panic!("Error in max operation {}", e),
        }
    }

    fn min(&self) -> Self::Output {
        match self.min_dim(None) {
            Ok(result) => result,
            Err(e) => panic!("Error in min operation {}", e),
        }
    }

}



impl Reshapable for Tensor {
    fn reshape(&self, shape: &[usize]) -> Self /*Result<Tensor, TensorError> */{
        //vérifier que le nombre total d'éléments est le même
        let new_size :usize = shape.iter().product();
        assert_eq!(self.numel(),new_size, "Shape size n'est pas compatible avec le nombre d'éléments");


        // if self.numel() != new_size {
        //     return Err(TensorError::new(ShapeMismatch,"Shape size mismatch with data length"));
        //
        // }
        //

        // creer un nouveau tenseur avec la meme memoire mais avec une nouvelle forme
        let mut result = self.clone();
        result.shape = shape.to_vec();

        // Recalculte les strides
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len()-1).rev(){
            strides[i] = strides[i+1] * shape[i+1];
        }
        result.strides = strides;

        result
        // Ok(result)
    }

    // flatten le tenseur
    fn flatten(&self) -> Self {
        self.reshape(&[self.numel()])
    }


    // transpose le tenseur
    fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        assert!(dim0 < self.ndim() && dim1 < self.ndim(), "Dimension out of range");

        // Créer un nouveau tenseur avec la forme transposée
        let mut result = self.clone();
        result.shape.swap(dim0,dim1);
        result.strides.swap(dim0,dim1);

        result

    }

}





// Unit tests
// rustytorch_tensor/src/lib.rs (partie tests)

#[cfg(test)]
mod tests_tensor_operation{
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

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(&data, vec![2, 3], None);

        let reshaped = tensor.reshape(&[3, 2]);
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.numel(), 6);

        let flattened = tensor.flatten();
        assert_eq!(flattened.shape(), &[6]);
    }

    #[test]
    fn test_tensor_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(&data, vec![2, 3], None);

        let transposed = tensor.transpose(0, 1);
        assert_eq!(transposed.shape(), &[3, 2]);
    }


    #[test]
    fn test_add() {
        let a = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
        let b = Tensor::from_data(&[4.0, 5.0, 6.0], vec![3], None);

        let c = a.clone().add(b.clone());

        // Vérifier la forme
        assert_eq!(c.shape(), &[3]);

        // Vérifier le contenu
        match c.storage.as_ref() {
            StorageType::F32(data) => {
                assert_eq!(data, &[5.0, 7.0, 9.0]);
            },
            StorageType::F64(data) => {
                assert_eq!(data, &[5.0, 7.0, 9.0]);
            },
            _ => panic!("Unexpected storage type"),
        }
    }

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
            },
            StorageType::F64(data) => {
                assert_eq!(data, &[6.0, 7.0, 8.0]);
            },
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
            },
            StorageType::F64(data) => {
                assert_eq!(data[0], 58.0);
                assert_eq!(data[1], 64.0);
                assert_eq!(data[2], 139.0);
                assert_eq!(data[3], 154.0);
            },
            _ => panic!("Unexpected storage type"),
        }
    }

    #[test]
    fn test_reduction_operations() {
        let tensor = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], None);

        // Test sum
        let sum = tensor.sum();
        match sum.storage.as_ref() {
            StorageType::F32(data) => {
                assert_eq!(data[0], 10.0); // 1+2+3+4
            },
            StorageType::F64(data) => {
                assert_eq!(data[0], 10.0);
            },
            _ => panic!("Unexpected storage type"),
        }

        // Test mean
        let mean = tensor.mean();
        match mean.storage.as_ref() {
            StorageType::F32(data) => {
                assert_eq!(data[0], 2.5); // (1+2+3+4)/4
            },
            StorageType::F64(data) => {
                assert_eq!(data[0], 2.5);
            },
            _ => panic!("Unexpected storage type"),
        }

        // Test max
        let max = tensor.max();
        match max.storage.as_ref() {
            StorageType::F32(data) => {
                assert_eq!(data[0], 4.0);
            },
            StorageType::F64(data) => {
                assert_eq!(data[0], 4.0);
            },
            _ => panic!("Unexpected storage type"),
        }

        // Test min
        let min = tensor.min();
        match min.storage.as_ref() {
            StorageType::F32(data) => {
                assert_eq!(data[0], 1.0);
            },
            StorageType::F64(data) => {
                assert_eq!(data[0], 1.0);
            },
            _ => panic!("Unexpected storage type"),
        }
    }
}








