// rustytorch_tensor/src/tensor_comparison.rs

use crate::storage::StorageType;
use crate::tensor_errors::{TensorError, TensorErrorType};
use crate::Tensor;
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
        match (
            self_broadcast.storage.as_ref(),
            other_broadcast.storage.as_ref(),
        ) {
            (StorageType::F32(a), StorageType::F32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] < b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::F64(a), StorageType::F64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] < b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I32(a), StorageType::I32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] < b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I64(a), StorageType::I64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] < b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            _ => {
                return Err(TensorError::new(
                    TensorErrorType::TypeError,
                    "Incompatible types for comparison",
                ))
            }
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
        match (
            self_broadcast.storage.as_ref(),
            other_broadcast.storage.as_ref(),
        ) {
            (StorageType::F32(a), StorageType::F32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] <= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::F64(a), StorageType::F64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] <= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I32(a), StorageType::I32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] <= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I64(a), StorageType::I64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] <= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            _ => {
                return Err(TensorError::new(
                    TensorErrorType::TypeError,
                    "Incompatible types for comparison",
                ))
            }
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
        match (
            self_broadcast.storage.as_ref(),
            other_broadcast.storage.as_ref(),
        ) {
            (StorageType::F32(a), StorageType::F32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = (a[i] - b[i]).abs() < 1e-6; // Comparaison avec tolérance pour les flottants
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::F64(a), StorageType::F64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = (a[i] - b[i]).abs() < 1e-10; // Comparaison avec tolérance pour les flottants
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I32(a), StorageType::I32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] == b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I64(a), StorageType::I64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] == b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::Bool(a), StorageType::Bool(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] == b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            _ => {
                return Err(TensorError::new(
                    TensorErrorType::TypeError,
                    "Incompatible types for comparison",
                ))
            }
        }

        Ok(result)
    }

    /// Compare élément par élément si les éléments du tenseur sont inférieurs ou égaux à ceux d'un autre tenseur
    pub fn ge(&self, other: &Self) -> Result<Self, TensorError> {
        let result_shape = self.broadcast_shapes(other)?;

        // Si les formes ne sont pas identiques, broadcaster
        let self_broadcast = self.broadcast_to(&result_shape)?;
        let other_broadcast = other.broadcast_to(&result_shape)?;

        // Créer un tenseur booléen résultat
        let mut result = Self::zeros(result_shape.clone(), Some(self.options.clone()));

        // Comparer élément par élément
        match (
            self_broadcast.storage.as_ref(),
            other_broadcast.storage.as_ref(),
        ) {
            (StorageType::F32(a), StorageType::F32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] >= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::F64(a), StorageType::F64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] >= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I32(a), StorageType::I32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] >= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I64(a), StorageType::I64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] >= b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::Bool(a), StorageType::Bool(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] >= b[i]; // true >= false
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            _ => {
                return Err(TensorError::new(
                    TensorErrorType::TypeError,
                    "Incompatible types for comparison",
                ))
            }
        }

        Ok(result)
    }

    /// Compare élément par élément si les éléments du tenseur sont supérieurs à ceux d'un autre tenseur
    pub fn gt(&self, other: &Self) -> Result<Self, TensorError> {
        let result_shape = self.broadcast_shapes(other)?;

        // Si les formes ne sont pas identiques, broadcaster
        let self_broadcast = self.broadcast_to(&result_shape)?;
        let other_broadcast = other.broadcast_to(&result_shape)?;

        // Créer un tenseur booléen résultat
        let mut result = Self::zeros(result_shape.clone(), Some(self.options.clone()));

        // Comparer élément par élément
        match (
            self_broadcast.storage.as_ref(),
            other_broadcast.storage.as_ref(),
        ) {
            (StorageType::F32(a), StorageType::F32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] > b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::F64(a), StorageType::F64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] > b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I32(a), StorageType::I32(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] > b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::I64(a), StorageType::I64(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] > b[i];
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            (StorageType::Bool(a), StorageType::Bool(b)) => {
                let mut result_data = vec![false; a.len()];
                for i in 0..a.len() {
                    result_data[i] = a[i] > b[i]; // true > false
                }
                result.storage = Arc::new(StorageType::from_bool(&result_data));
            }
            _ => {
                return Err(TensorError::new(
                    TensorErrorType::TypeError,
                    "Incompatible types for comparison",
                ))
            }
        }

        Ok(result)
    }

    /// Compare élément par élément si les éléments ne sont pas égaux
    pub fn ne(&self, other: &Self) -> Result<Self, TensorError> {
        let eq_result = self.eq(other)?;
        // Inverser le résultat
        match eq_result.storage.as_ref() {
            StorageType::Bool(data) => {
                let inverted_data: Vec<bool> = data.iter().map(|&x| !x).collect();
                let mut result = eq_result.clone();
                result.storage = Arc::new(StorageType::from_bool(&inverted_data));
                Ok(result)
            }
            _ => Err(TensorError::new(
                TensorErrorType::TypeError,
                "Expected boolean tensor from eq operation",
            )),
        }
    }

    /// Vérifie si tous les éléments sont vrais (pour tenseurs booléens)
    pub fn all(&self) -> Result<bool, TensorError> {
        match self.storage.as_ref() {
            StorageType::Bool(data) => Ok(data.iter().all(|&x| x)),
            _ => Err(TensorError::new(
                TensorErrorType::TypeError,
                "all() operation requires boolean tensor",
            )),
        }
    }

    /// Vérifie si au moins un élément est vrai (pour tenseurs booléens)
    pub fn any(&self) -> Result<bool, TensorError> {
        match self.storage.as_ref() {
            StorageType::Bool(data) => Ok(data.iter().any(|&x| x)),
            _ => Err(TensorError::new(
                TensorErrorType::TypeError,
                "any() operation requires boolean tensor",
            )),
        }
    }

    // to_f64 method moved to type_ops.rs for comprehensive type support
}

// Implémentation du trait Comparable
use rustytorch_core::{Comparable, CoreError, Result as CoreResult};

#[cfg(test)]
mod tests {
    use super::*;
    use rustytorch_core::{DType, TensorOptions};

    #[test]
    fn test_comparison_operations() {
        let a = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], None);
        let b = Tensor::from_data(&[2.0f32, 2.0, 1.0], vec![3], None);

        // Test lt
        let lt_result = a.lt(&b).unwrap();
        if let StorageType::Bool(data) = lt_result.storage.as_ref() {
            assert_eq!(data, &[true, false, false]);
        } else {
            panic!("Expected boolean storage");
        }

        // Test gt
        let gt_result = a.gt(&b).unwrap();
        if let StorageType::Bool(data) = gt_result.storage.as_ref() {
            assert_eq!(data, &[false, false, true]);
        } else {
            panic!("Expected boolean storage");
        }

        // Test eq
        let eq_result = a.eq(&b).unwrap();
        if let StorageType::Bool(data) = eq_result.storage.as_ref() {
            assert_eq!(data, &[false, true, false]);
        } else {
            panic!("Expected boolean storage");
        }

        // Test ne
        let ne_result = a.ne(&b).unwrap();
        if let StorageType::Bool(data) = ne_result.storage.as_ref() {
            assert_eq!(data, &[true, false, true]);
        } else {
            panic!("Expected boolean storage");
        }
    }

    #[test]
    fn test_comparable_trait() {
        use rustytorch_core::Comparable;

        let a = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], None);
        let b = Tensor::from_data(&[2.0f32, 2.0, 1.0], vec![3], None);

        // Test trait methods
        let eq_result = Comparable::eq(&a, &b).unwrap();
        if let StorageType::Bool(data) = eq_result.storage.as_ref() {
            assert_eq!(data, &[false, true, false]);
        } else {
            panic!("Expected boolean storage");
        }

        let lt_result = Comparable::lt(&a, &b).unwrap();
        if let StorageType::Bool(data) = lt_result.storage.as_ref() {
            assert_eq!(data, &[true, false, false]);
        } else {
            panic!("Expected boolean storage");
        }
    }

    #[test]
    fn test_all_any_operations() {
        let all_true = Tensor::from_data(
            &[true, true, true],
            vec![3],
            Some(TensorOptions::new().dtype(DType::Bool)),
        );
        let mixed = Tensor::from_data(
            &[true, false, true],
            vec![3],
            Some(TensorOptions::new().dtype(DType::Bool)),
        );
        let all_false = Tensor::from_data(
            &[false, false, false],
            vec![3],
            Some(TensorOptions::new().dtype(DType::Bool)),
        );

        // Test all()
        assert_eq!(all_true.all().unwrap(), true);
        assert_eq!(mixed.all().unwrap(), false);
        assert_eq!(all_false.all().unwrap(), false);

        // Test any()
        assert_eq!(all_true.any().unwrap(), true);
        assert_eq!(mixed.any().unwrap(), true);
        assert_eq!(all_false.any().unwrap(), false);

        // Test trait methods
        use rustytorch_core::Comparable;
        assert_eq!(Comparable::all(&all_true).unwrap(), true);
        assert_eq!(Comparable::any(&mixed).unwrap(), true);
    }
}

impl Comparable for Tensor {
    type Output = Tensor;

    fn eq(&self, other: &Self) -> CoreResult<Self::Output> {
        self.eq(other)
            .map_err(|e| CoreError::invalid_op("eq", &e.to_string()))
    }

    fn ne(&self, other: &Self) -> CoreResult<Self::Output> {
        self.ne(other)
            .map_err(|e| CoreError::invalid_op("ne", &e.to_string()))
    }

    fn lt(&self, other: &Self) -> CoreResult<Self::Output> {
        self.lt(other)
            .map_err(|e| CoreError::invalid_op("lt", &e.to_string()))
    }

    fn le(&self, other: &Self) -> CoreResult<Self::Output> {
        self.le(other)
            .map_err(|e| CoreError::invalid_op("le", &e.to_string()))
    }

    fn gt(&self, other: &Self) -> CoreResult<Self::Output> {
        self.gt(other)
            .map_err(|e| CoreError::invalid_op("gt", &e.to_string()))
    }

    fn ge(&self, other: &Self) -> CoreResult<Self::Output> {
        self.ge(other)
            .map_err(|e| CoreError::invalid_op("ge", &e.to_string()))
    }

    fn all(&self) -> CoreResult<bool> {
        self.all()
            .map_err(|e| CoreError::invalid_op("all", &e.to_string()))
    }

    fn any(&self) -> CoreResult<bool> {
        self.any()
            .map_err(|e| CoreError::invalid_op("any", &e.to_string()))
    }
}

impl Tensor {
    /// Retourne le minimum élément par élément entre deux tenseurs
    pub fn minimum(&self, other: &Self) -> Result<Self, TensorError> {
        self.apply_binary_op(
            other,
            |a, b| if a < b { a } else { b },
            |a, b| if a < b { a } else { b },
        )
    }
    
    /// Retourne le maximum élément par élément entre deux tenseurs
    pub fn maximum(&self, other: &Self) -> Result<Self, TensorError> {
        self.apply_binary_op(
            other,
            |a, b| if a > b { a } else { b },
            |a, b| if a > b { a } else { b },
        )
    }
}
