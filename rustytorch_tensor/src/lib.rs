//rustytorch_tensor/src/lib.rs


use rustytorch_core::{Dtype, TensorOptions, NumericOps, Reduction, Reshapable, Device};

use std::sync::Arc;
use rand::Rng;

// use std::simd::f32x8;
use rayon::prelude::*;


mod storage;

use storage::StorageType;


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

    /// renvoie le type de données du tenseur
    pub fn dtype(&self) -> Dtype {
        self.options.dtype
    }


    // renvoie le type de stockage du tenseur
    pub fn storage(&self) -> &StorageType {
        &self.storage
    }

    // renvoie le device
    pub fn device(&self) -> &Device{
        &self.options.device
    }


    // Addition optimisée avec SIMD et parallélisation
    // pub fn add_optimized(&self, other: &Self) -> Self {
    //     assert_eq!(self.shape(), other.shape());
    //
    //     let mut result = Self::zeros(self.shape().to_vec(), Some(self.options.clone()));
    //
    //     // Parallélisation avec Rayon
    //     result.data.chunks_mut(1024)
    //         .zip(self.data.chunks(1024))
    //         .zip(other.data.chunks(1024))
    //         .par_bridge()
    //         .for_each(|((result_chunk, self_chunk), other_chunk)| {
    //             // SIMD pour chaque chunk
    //             for (i, (a, b)) in self_chunk.iter().zip(other_chunk.iter()).enumerate() {
    //                 // Utilisation de SIMD quand les données sont alignées
    //                 if i + 8 <= result_chunk.len() {
    //                     let a_simd = f32x8::from_slice(&self_chunk[i..i+8]);
    //                     let b_simd = f32x8::from_slice(&other_chunk[i..i+8]);
    //                     let result_simd = a_simd + b_simd;
    //                     result_simd.write_to_slice_unaligned(&mut result_chunk[i..i+8]);
    //                     i += 7; // On avance de 7 (la boucle incrémentera de 1)
    //                 } else {
    //                     result_chunk[i] = a + b;
    //                 }
    //             }
    //         });
    //
    //     result
    // }

}


/// Implémentation NumericOps pour le tenseur

impl NumericOps for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        unimplemented!()
    }
    fn sub(self, rhs: Self) -> Self::Output {
        unimplemented!()
    }
    fn mul(self, rhs: Self) -> Self::Output {
        unimplemented!()
    }
    fn div(self, rhs: Self) -> Self::Output {
        unimplemented!()
    }
}


impl Reduction for Tensor{
    type Output = Tensor;

    fn sum(&self) -> Self::Output {
        unimplemented!()
    }
    fn mean(&self) -> Self::Output {
        unimplemented!()
    }

    fn max(&self) -> Self::Output {
        unimplemented!()
    }

    fn min(&self) -> Self::Output {
        unimplemented!()
    }

}



impl Reshapable for Tensor {
    fn reshape(&self, shape: &[usize]) -> Self {
        //vérifier que le nombre total d'éléments est le même
        let new_size :usize = shape.iter().product();
        // assert_eq!(self.numel(),new_size,"Shape size mismatch with data length");
        assert_eq!(self.numel(),new_size, "Shape size n'est pas compatible avec le nombre d'éléments");

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
mod tests_tensor{
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
}








