//rustytorch_tensor/src/broadcastings.rs
use rayon::iter::IndexedParallelIterator;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator};
use rayon::prelude::*;
use std::sync::Arc;

use crate::storage::StorageType;
use crate::tensor_errors::TensorErrorType::ShapeMismatch;
use crate::tensor_errors::{TensorError, TensorErrorType};
use crate::Tensor;

impl Tensor {
    /// Compare les formes de deux tenseurs pour la compatibilité avec le broadcasting
    pub fn broadcast_shapes(&self, other: &Self) -> Result<Vec<usize>, TensorError> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Si les formes sont identiques, aucun broadcasting n'est nécessaire
        if a_shape == b_shape {
            return Ok(a_shape.to_vec());
        }

        // Aligner les dimensions à droite (en ajoutant des dimensions de taille 1 à gauche)
        let a_dims = a_shape.len();
        let b_dims = b_shape.len();
        let result_dims = std::cmp::max(a_dims, b_dims);

        let mut result_shape = Vec::with_capacity(result_dims);

        // Parcourir les dimensions de droite à gauche
        for i in 0..result_dims {
            let a_dim = if i < a_dims {
                a_shape[a_dims - 1 - i]
            } else {
                1
            };
            let b_dim = if i < b_dims {
                b_shape[b_dims - 1 - i]
            } else {
                1
            };

            // Les dimensions doivent être égales ou l'une d'elles doit être 1
            if a_dim == b_dim || a_dim == 1 || b_dim == 1 {
                result_shape.push(std::cmp::max(a_dim, b_dim));
            } else {
                return Err(TensorError::new(
                    ShapeMismatch,
                    &format!(
                        "Cannot broadcast shapes {:?} and {:?}: incompatible dimensions {} and {}",
                        a_shape, b_shape, a_dim, b_dim
                    ),
                ));
            }
        }

        // Inverser pour avoir les dimensions dans le bon ordre
        result_shape.reverse();
        Ok(result_shape)
    }

    /// Étend un tenseur pour correspondre à une forme spécifique (pour le broadcasting)
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor, TensorError> {
        let self_shape = self.shape();

        // Vérifier si le broadcast est possible
        if self_shape == shape {
            return Ok(self.clone());
        }

        // La forme cible doit avoir au moins autant de dimensions que la forme source
        if shape.len() < self_shape.len() {
            return Err(TensorError::new(
                ShapeMismatch,
                &format!(
                    "Cannot broadcast shape {:?} to shape {:?}: target shape has fewer dimensions",
                    self_shape, shape
                ),
            ));
        }

        // Vérifier la compatibilité pour chaque dimension
        let offset = shape.len() - self_shape.len();
        for i in 0..self_shape.len() {
            if self_shape[i] != shape[i + offset] && self_shape[i] != 1 {
                return Err(TensorError::new(
                    ShapeMismatch,
                    &format!("Cannot broadcast shape {:?} to shape {:?}: incompatible dimension {} vs {}",
                             self_shape, shape, self_shape[i], shape[i + offset])
                ));
            }
        }

        // Créer un nouveau tenseur avec la forme cible
        let mut result = Self::zeros(shape.to_vec(), Some(self.options.clone()));

        // TODO: Implémentation complète du broadcasting qui copie les données
        // Pour le moment, nous utilisons une implémentation simplifiée

        // Si notre tenseur d'entrée a une seule valeur, nous la diffusons à tous les éléments
        if self.numel() == 1 {
            let value = match self.storage.as_ref() {
                StorageType::F32(data) => data[0] as f64,
                StorageType::F64(data) => data[0],
                _ => unimplemented!("Type de données non supporté"),
            };

            // Remplir le tenseur résultat avec cette valeur
            let total_size = result.numel();
            match result.storage.as_ref() {
                StorageType::F32(_) => {
                    let data = vec![value as f32; total_size];
                    result.storage = Arc::new(StorageType::from_f32(&data));
                }
                StorageType::F64(_) => {
                    let data = vec![value; total_size];
                    result.storage = Arc::new(StorageType::from_f64(&data));
                }
                _ => unimplemented!("Type de données non supporté"),
            }
        } else {
            // Pour les cas plus complexes, une implémentation plus sophistiquée est nécessaire
            // TODO: Implémenter le broadcasting pour les cas multidimensionnels
            unimplemented!("Broadcasting multidimensionnel non implémenté");
        }

        Ok(result)
    }

    fn parallel_binary_op<F32Op, F64Op>(
        &self,
        other: &Self,
        f32_op: F32Op,
        f64_op: F64Op,
    ) -> Result<Self, TensorError>
    where
        F32Op: Fn(f32, f32) -> f32 + Sync + Send,
        F64Op: Fn(f64, f64) -> f64 + Sync + Send,
    {
        // Déterminer la forme résultante après broadcasting
        let result_shape = self.broadcast_shapes(other)?;

        // Broadcast les deux tenseurs à la forme résultante
        let a_broadcast = self.broadcast_to(&result_shape)?;
        let b_broadcast = other.broadcast_to(&result_shape)?;

        // Maintenant que les deux tenseurs ont la même forme, faire l'opération
        let mut result = Self::zeros(result_shape, Some(self.options.clone()));

        match (a_broadcast.storage.as_ref(), b_broadcast.storage.as_ref()) {
            (StorageType::F32(a_data), StorageType::F32(b_data)) => {
                let mut result_data = vec![0.0; a_data.len()];

                if a_data.len() > 10000 {
                    result_data
                        .par_iter_mut()
                        .zip(a_data.par_iter().zip(b_data.par_iter()))
                        .for_each(|(res, (a, b))| {
                            *res = f32_op(*a, *b);
                        });
                } else {
                    for i in 0..a_data.len() {
                        result_data[i] = f32_op(a_data[i], b_data[i]);
                    }
                }

                result.storage = Arc::new(StorageType::from_f32(&result_data));
            }
            (StorageType::F64(a_data), StorageType::F64(b_data)) => {
                let mut result_data = vec![0.0; a_data.len()];

                if a_data.len() > 10000 {
                    result_data
                        .par_iter_mut()
                        .zip(a_data.par_iter().zip(b_data.par_iter()))
                        .for_each(|(res, (a, b))| {
                            *res = f64_op(*a, *b);
                        });
                } else {
                    for i in 0..a_data.len() {
                        result_data[i] = f64_op(a_data[i], b_data[i]);
                    }
                }

                result.storage = Arc::new(StorageType::from_f64(&result_data));
            }
            _ => {
                return Err(TensorError::new(
                    TensorErrorType::TypeError,
                    "Unsupported or mismatched data types for operation",
                ))
            }
        }

        Ok(result)
    }
    /// Addition avec broadcasting
    pub fn add_broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        self.parallel_binary_op(other, |a, b| a + b, |a, b| a + b)
    }
    /// Soustraction avec broadcasting
    pub fn sub_broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        self.parallel_binary_op(other, |a, b| a - b, |a, b| a - b)
    }
    /// Multiplication avec broadcasting
    pub fn mul_broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        self.parallel_binary_op(other, |a, b| a * b, |a, b| a * b)
    }
    /// Division avec broadcasting
    pub fn div_broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        self.parallel_binary_op(
            other,
            |a, b| if b != 0.0 { a / b } else { f32::NAN },
            |a, b| if b != 0.0 { a / b } else { f64::NAN },
        )
    }

    // matmul method moved to linalg.rs with optimized implementation

    /// Opération de réduction - sum
    pub fn sum_dim(&self, dim: Option<usize>) -> Result<Self, TensorError> {
        if self.numel() == 0 {
            return Err(TensorError::new(
                TensorErrorType::InvalidOperation,
                "Cannot compute sum of empty tensor",
            ));
        }

        match dim {
            Some(d) => {
                if d >= self.ndim() {
                    return Err(TensorError::new(
                        TensorErrorType::IndexOutOfBounds,
                        &format!(
                            "Dimension {} out of range for tensor with {} dimensions",
                            d,
                            self.ndim()
                        ),
                    ));
                }

                // Calculer la forme résultante après réduction
                let mut result_shape = self.shape().to_vec();
                result_shape[d] = 1;

                let mut result = Self::zeros(result_shape.clone(), Some(self.options.clone()));

                // Implémentation pour différents types de stockage
                match self.storage.as_ref() {
                    StorageType::F32(data) => {
                        // TODO: Implémentation de sum le long d'une dimension
                        // Cette implémentation est simplifiée et doit être adaptée
                        let result_data = vec![0.0; result.numel()];

                        // Pour chaque élément du résultat, calculer la somme
                        // Cette implémentation est naïve et devrait être optimisée
                        // pour des tenseurs de grande taille

                        result.storage = Arc::new(StorageType::from_f32(&result_data));
                    }
                    StorageType::F64(data) => {
                        // Implémentation similaire pour F64
                        let result_data = vec![0.0; result.numel()];

                        result.storage = Arc::new(StorageType::from_f64(&result_data));
                    }
                    _ => {
                        return Err(TensorError::new(
                            TensorErrorType::TypeError,
                            "Unsupported data type for sum operation",
                        ))
                    }
                }

                Ok(result)
            }
            None => {
                // Sum de tous les éléments
                let result_shape = vec![1]; // Tensor scalaire
                let mut result = Self::zeros(result_shape, Some(self.options.clone()));

                match self.storage.as_ref() {
                    StorageType::F32(data) => {
                        let sum: f32 = data.iter().sum();
                        result.storage = Arc::new(StorageType::from_f32(&[sum]));
                    }
                    StorageType::F64(data) => {
                        let sum: f64 = data.iter().sum();
                        result.storage = Arc::new(StorageType::from_f64(&[sum]));
                    }
                    _ => {
                        return Err(TensorError::new(
                            TensorErrorType::TypeError,
                            "Unsupported data type for sum operation",
                        ))
                    }
                }

                Ok(result)
            }
        }
    }

    /// Opération de réduction - mean
    pub fn mean_dim(&self, dim: Option<usize>) -> Result<Self, TensorError> {
        if self.numel() == 0 {
            return Err(TensorError::new(
                TensorErrorType::InvalidOperation,
                "Cannot compute mean of empty tensor",
            ));
        }

        // Calculer la somme
        let sum_result = self.sum_dim(dim)?;

        // Diviser par le nombre d'éléments
        let num_elements = match dim {
            Some(d) => self.shape()[d] as f64,
            None => self.numel() as f64,
        };

        match sum_result.storage.as_ref() {
            StorageType::F32(data) => {
                let mut result_data = vec![0.0; data.len()];
                for i in 0..data.len() {
                    result_data[i] = data[i] / num_elements as f32;
                }
                let mut result = sum_result.clone();
                result.storage = Arc::new(StorageType::from_f32(&result_data));
                Ok(result)
            }
            StorageType::F64(data) => {
                let mut result_data = vec![0.0; data.len()];
                for i in 0..data.len() {
                    result_data[i] = data[i] / num_elements;
                }
                let mut result = sum_result.clone();
                result.storage = Arc::new(StorageType::from_f64(&result_data));
                Ok(result)
            }
            _ => Err(TensorError::new(
                TensorErrorType::TypeError,
                "Unsupported data type for mean operation",
            )),
        }
    }

    /// Opération de réduction - max
    pub fn max_dim(&self, dim: Option<usize>) -> Result<Self, TensorError> {
        if self.numel() == 0 {
            return Err(TensorError::new(
                TensorErrorType::InvalidOperation,
                "Cannot compute max of empty tensor",
            ));
        }

        match dim {
            Some(d) => {
                if d >= self.ndim() {
                    return Err(TensorError::new(
                        TensorErrorType::IndexOutOfBounds,
                        &format!(
                            "Dimension {} out of range for tensor with {} dimensions",
                            d,
                            self.ndim()
                        ),
                    ));
                }

                // Calculer la forme résultante après réduction
                let mut result_shape = self.shape().to_vec();
                result_shape[d] = 1;

                let result = Self::zeros(result_shape.clone(), Some(self.options.clone()));

                // TODO: Implémentation de max le long d'une dimension
                // (similaire à sum_dim)

                Ok(result)
            }
            None => {
                // Max de tous les éléments
                let result_shape = vec![1]; // Tensor scalaire
                let mut result = Self::zeros(result_shape, Some(self.options.clone()));

                match self.storage.as_ref() {
                    StorageType::F32(data) => {
                        if let Some(max) = data.iter().cloned().reduce(f32::max) {
                            result.storage = Arc::new(StorageType::from_f32(&[max]));
                        } else {
                            return Err(TensorError::new(
                                TensorErrorType::InvalidOperation,
                                "Failed to compute max",
                            ));
                        }
                    }
                    StorageType::F64(data) => {
                        if let Some(max) = data.iter().cloned().reduce(f64::max) {
                            result.storage = Arc::new(StorageType::from_f64(&[max]));
                        } else {
                            return Err(TensorError::new(
                                TensorErrorType::InvalidOperation,
                                "Failed to compute max",
                            ));
                        }
                    }
                    _ => {
                        return Err(TensorError::new(
                            TensorErrorType::TypeError,
                            "Unsupported data type for max operation",
                        ))
                    }
                }

                Ok(result)
            }
        }
    }

    /// Opération de réduction - min
    pub fn min_dim(&self, dim: Option<usize>) -> Result<Self, TensorError> {
        if self.numel() == 0 {
            return Err(TensorError::new(
                TensorErrorType::InvalidOperation,
                "Cannot compute min of empty tensor",
            ));
        }

        match dim {
            Some(d) => {
                if d >= self.ndim() {
                    return Err(TensorError::new(
                        TensorErrorType::IndexOutOfBounds,
                        &format!(
                            "Dimension {} out of range for tensor with {} dimensions",
                            d,
                            self.ndim()
                        ),
                    ));
                }

                // Calculer la forme résultante après réduction
                let mut result_shape = self.shape().to_vec();
                result_shape[d] = 1;

                let result = Self::zeros(result_shape.clone(), Some(self.options.clone()));

                // TODO: Implémentation de min le long d'une dimension
                // (similaire à sum_dim)

                Ok(result)
            }
            None => {
                // Min de tous les éléments
                let result_shape = vec![1]; // Tensor scalaire
                let mut result = Self::zeros(result_shape, Some(self.options.clone()));

                match self.storage.as_ref() {
                    StorageType::F32(data) => {
                        if let Some(min) = data.iter().cloned().reduce(f32::min) {
                            result.storage = Arc::new(StorageType::from_f32(&[min]));
                        } else {
                            return Err(TensorError::new(
                                TensorErrorType::InvalidOperation,
                                "Failed to compute min",
                            ));
                        }
                    }
                    StorageType::F64(data) => {
                        if let Some(min) = data.iter().cloned().reduce(f64::min) {
                            result.storage = Arc::new(StorageType::from_f64(&[min]));
                        } else {
                            return Err(TensorError::new(
                                TensorErrorType::InvalidOperation,
                                "Failed to compute min",
                            ));
                        }
                    }
                    _ => {
                        return Err(TensorError::new(
                            TensorErrorType::TypeError,
                            "Unsupported data type for min operation",
                        ))
                    }
                }

                Ok(result)
            }
        }
    }
}
