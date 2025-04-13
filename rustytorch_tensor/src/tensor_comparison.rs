// rustytorch_tensor/src/tensor_comparison.rs

use crate::Tensor;
use crate::storage::StorageType;
use crate::tensor_errors::{TensorError, TensorErrorType};
use std::sync::Arc;

impl Tensor {
    /// Compare élément par élément si les éléments du tenseur sont inférieurs à ceux d'un autre tenseur
    pub fn lt(&self, other: &Self) -> Result<Self, TensorError> {
        let result_shape = self.broadcast_shapes(other)?;

        // Si les formes ne sont pas identiques, broadcaster
        let self_broadcast = self.broadcast_to(&result_shape)?;
        let other_broadcast = other.broadcast_to(&result_shape)?;

        // Créer un tenseur booléen résultat
        let mut result = Self::zeros(result_shape.clone(), Some(self.options.clone()));

        // Comparer élément par élément
        match (self_broadcast.storage.as_ref(), other_broadcast.storage.as_ref()) {
            (StorageType::F32(a), StorageType::F32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] < b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::F64(a), StorageType::F64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] < b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::I32(a), StorageType::I32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] < b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::I64(a), StorageType::I64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] < b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            _ => return Err(TensorError::new(
                TensorErrorType::TypeError,
                "Incompatible types for comparison"
            )),
        }

        Ok(result)
    }

    /// Compare élément par élément si les éléments du tenseur sont inférieurs ou égaux à ceux d'un autre tenseur
    pub fn le(&self, other: &Self) -> Result<Self, TensorError> {
        let result_shape = self.broadcast_shapes(other)?;

        // Si les formes ne sont pas identiques, broadcaster
        let self_broadcast = self.broadcast_to(&result_shape)?;
        let other_broadcast = other.broadcast_to(&result_shape)?;

        // Créer un tenseur booléen résultat
        // let mut result = Self::zeros(result_shape.clone(), Some(self.options.clone()))?;
        let mut result = Self::zeros(result_shape.clone(), Some(self.options.clone()));
        // Comparer élément par élément
        match (self_broadcast.storage.as_ref(), other_broadcast.storage.as_ref()) {
            (StorageType::F32(a), StorageType::F32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] <= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::F64(a), StorageType::F64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] <= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::I32(a), StorageType::I32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] <= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::I64(a), StorageType::I64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] <= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            _ => return Err(TensorError::new(
                TensorErrorType::TypeError,
                "Incompatible types for comparison"
            )),
        }

        Ok(result)
    }

    /// Compare élément par élément si les éléments du tenseur sont égaux à ceux d'un autre tenseur
    pub fn eq(&self, other: &Self) -> Result<Self, TensorError> {
        let result_shape = self.broadcast_shapes(other)?;

        // Si les formes ne sont pas identiques, broadcaster
        let self_broadcast = self.broadcast_to(&result_shape)?;
        let other_broadcast = other.broadcast_to(&result_shape)?;

        // Créer un tenseur booléen résultat
        let mut result = Self::zeros(result_shape.clone(), Some(self.options.clone()));

        // Comparer élément par élément
        match (self_broadcast.storage.as_ref(), other_broadcast.storage.as_ref()) {
            (StorageType::F32(a), StorageType::F32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = (a[i] - b[i]).abs() < 1e-6;  // Comparaison avec tolérance pour les flottants
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::F64(a), StorageType::F64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = (a[i] - b[i]).abs() < 1e-10;  // Comparaison avec tolérance pour les flottants
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::I32(a), StorageType::I32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] == b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::I64(a), StorageType::I64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] == b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            (StorageType::Bool(a), StorageType::Bool(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] == b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            },
            _ => return Err(TensorError::new(
                TensorErrorType::TypeError,
                "Incompatible types for comparison"
            )),
        }

        Ok(result)
    }

    /// Convertit un tenseur booléen en tenseur f64
    pub fn to_f64(&self) -> Result<Self, TensorError> {
        match self.storage.as_ref() {
            StorageType::Bool(data) => {
                let result_data: Vec<f64> = data.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
                // Ok(Self::from_data(&result_data, self.shape().to_vec(), Some(self.options.clone()))?)
                Ok(Self::from_data(&result_data, self.shape().to_vec(), Some(self.options.clone())))
            }
            _ => Err(TensorError::new(
                TensorErrorType::TypeError,
                "Expected Bool tensor for conversion to f64"
            )),
        }
    }
}