// rustytorch_tensor/src/tensor_optims.rs

use rayon::prelude::*;
use crate::Tensor;
use crate::storage::StorageType;
use crate::tensor_errors::TensorError;
use crate::tensor_errors::TensorErrorType;
use std::sync::Arc;

/// Module contenant des optimisations pour les opérations tensorielles
impl Tensor {
    /// Applique une opération unaire optimisée élément par élément sur le tenseur
    pub(crate) fn apply_unary_op<F32Op, F64Op>(&self, f32_op: F32Op, f64_op: F64Op) -> Result<Self, TensorError>
    where
        F32Op: Fn(f32) -> f32 + Sync + Send,
        F64Op: Fn(f64) -> f64 + Sync + Send,
    {
        let mut result = self.clone();

        match self.storage.as_ref() {
            StorageType::F32(data) => {
                let mut result_data = vec![0.0; data.len()];

                // Utiliser Rayon pour la parallélisation si le tenseur est assez grand
                if data.len() > 10000 {
                    result_data.par_iter_mut().zip(data.par_iter())
                        .for_each(|(res, &val)| {
                            *res = f32_op(val);
                        });
                } else {
                    // Sinon, utiliser une boucle séquentielle
                    for (res, &val) in result_data.iter_mut().zip(data.iter()) {
                        *res = f32_op(val);
                    }
                }

                result.storage = Arc::new(StorageType::from_f32(&result_data));
            },
            StorageType::F64(data) => {
                let mut result_data = vec![0.0; data.len()];

                if data.len() > 10000 {
                    result_data.par_iter_mut().zip(data.par_iter())
                        .for_each(|(res, &val)| {
                            *res = f64_op(val);
                        });
                } else {
                    for (res, &val) in result_data.iter_mut().zip(data.iter()) {
                        *res = f64_op(val);
                    }
                }

                result.storage = Arc::new(StorageType::from_f64(&result_data));
            },
            _ => return Err(TensorError::new(
                TensorErrorType::UnsupportedOperation,
                "Unsupported storage type for unary operation",
            )),
        }

        Ok(result)
    }

    /// Applique une opération binaire optimisée élément par élément sur deux tenseurs
    pub(crate) fn apply_binary_op<F32Op, F64Op>(&self, other: &Self, f32_op: F32Op, f64_op: F64Op) -> Result<Self, TensorError>
    where
        F32Op: Fn(f32, f32) -> f32 + Sync + Send,
        F64Op: Fn(f64, f64) -> f64 + Sync + Send,
    {
        // Vérifier si les formes sont compatibles
        let self_shape = self.shape();
        let other_shape = other.shape();

        // Si les formes ne sont pas identiques, utiliser broadcasting
        if self_shape != other_shape {
            let result_shape = self.broadcast_shapes(other)?;
            let a_broadcast = self.broadcast_to(&result_shape)?;
            let b_broadcast = other.broadcast_to(&result_shape)?;

            return a_broadcast.apply_binary_op(&b_broadcast, f32_op, f64_op);
        }

        // Créer le tenseur résultat
        let mut result = Self::zeros(self_shape.to_vec(), Some(self.options.clone()));

        // Appliquer l'opération en fonction du type de stockage
        match (self.storage.as_ref(), other.storage.as_ref()) {
            (StorageType::F32(a_data), StorageType::F32(b_data)) => {
                let mut result_data = vec![0.0; a_data.len()];

                // Parallélisation pour les grands tenseurs
                if a_data.len() > 10000 {
                    result_data.par_iter_mut().zip(a_data.par_iter().zip(b_data.par_iter()))
                        .for_each(|(res, (&a, &b))| {
                            *res = f32_op(a, b);
                        });
                } else {
                    for i in 0..a_data.len() {
                        result_data[i] = f32_op(a_data[i], b_data[i]);
                    }
                }

                result.storage = Arc::new(StorageType::from_f32(&result_data));
            },
            (StorageType::F64(a_data), StorageType::F64(b_data)) => {
                let mut result_data = vec![0.0; a_data.len()];

                if a_data.len() > 10000 {
                    result_data.par_iter_mut().zip(a_data.par_iter().zip(b_data.par_iter()))
                        .for_each(|(res, (&a, &b))| {
                            *res = f64_op(a, b);
                        });
                } else {
                    for i in 0..a_data.len() {
                        result_data[i] = f64_op(a_data[i], b_data[i]);
                    }
                }

                result.storage = Arc::new(StorageType::from_f64(&result_data));
            },
            _ => return Err(TensorError::new(
                TensorErrorType::TypeError,
                "Mismatched or unsupported storage types for binary operation",
            )),
        }

        Ok(result)
    }

    /// Version optimisée de add_broadcast utilisant apply_binary_op
    pub fn add_optimized(&self, other: &Self) -> Result<Self, TensorError> {
        self.apply_binary_op(other, |a, b| a + b, |a, b| a + b)
    }

    /// Version optimisée de sub_broadcast utilisant apply_binary_op
    pub fn sub_optimized(&self, other: &Self) -> Result<Self, TensorError> {
        self.apply_binary_op(other, |a, b| a - b, |a, b| a - b)
    }

    /// Version optimisée de mul_broadcast utilisant apply_binary_op
    pub fn mul_optimized(&self, other: &Self) -> Result<Self, TensorError> {
        self.apply_binary_op(other, |a, b| a * b, |a, b| a * b)
    }

    /// Version optimisée de div_broadcast utilisant apply_binary_op
    pub fn div_optimized(&self, other: &Self) -> Result<Self, TensorError> {
        self.apply_binary_op(other,
                             |a, b| if b != 0.0 { a / b } else { f32::NAN },
                             |a, b| if b != 0.0 { a / b } else { f64::NAN })
    }

    /// Applique une fonction d'activation ReLU optimisée
    pub fn relu(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| if x > 0.0 { x } else { 0.0 },
            |x| if x > 0.0 { x } else { 0.0 }
        )
    }

    /// Applique une fonction d'activation sigmoid optimisée
    pub fn sigmoid(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| 1.0 / (1.0 + (-x).exp()),
            |x| 1.0 / (1.0 + (-x).exp())
        )
    }

    /// Applique une fonction d'activation cosh optimisée
    pub fn cosh(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| x.cosh(),
            |x| x.cosh()
        )
    }

    /// Applique une fonction d'activation sinh optimisée
    pub fn sinh(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| x.sinh(),
            |x| x.sinh()
        )
    }

    /// Applique une fonction d'activation tanh optimisée
    pub fn tanh(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| x.tanh(),
            |x| x.tanh()
        )
    }

    /// Élève chaque élément du tenseur à une puissance
    pub fn pow(&self, exponent: f64) -> Result<Self, TensorError> {
        let exp_f32 = exponent as f32;

        self.apply_unary_op(
            |x| x.powf(exp_f32),
            |x| x.powf(exponent)
        )
    }

    /// Calcule l'exponentielle de chaque élément
    pub fn exp(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| x.exp(),
            |x| x.exp()
        )
    }

    /// Calcule le logarithme naturel de chaque élément
    pub fn log(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| if x > 0.0 { x.ln() } else { f32::NAN },
            |x| if x > 0.0 { x.ln() } else { f64::NAN }
        )
    }

    /// Calcule le sinus de chaque élément
    pub fn sin(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| x.sin(),
            |x| x.sin()
        )
    }

    /// Calcule le cosinus de chaque élément
    pub fn cos(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| x.cos(),
            |x| x.cos()
        )
    }

    /// Calcule la tangente de chaque élément
    pub fn tan(&self) -> Result<Self, TensorError> {
        self.apply_unary_op(
            |x| x.tan(),
            |x| x.tan()
        )
    }


    /// Applique une fonction de softmax (pour les problèmes de classification)
    pub fn softmax(&self, dim: Option<usize>) -> Result<Self, TensorError> {
        // Si aucune dimension n'est spécifiée, appliquer softmax sur la dernière dimension
        let dim = dim.unwrap_or(self.ndim() - 1);

        if dim >= self.ndim() {
            return Err(TensorError::new(
                TensorErrorType::IndexOutOfBounds,
                &format!("Dimension {} out of range for tensor with {} dimensions", dim, self.ndim())
            ));
        }

        // Pour simplifier, nous implémentons softmax pour des tenseurs 1D et 2D uniquement
        if self.ndim() > 2 {
            return Err(TensorError::new(
                TensorErrorType::UnsupportedOperation,
                "Softmax for tensors with dimension > 2 not implemented yet"
            ));
        }

        // Cas 1D: appliquer softmax sur le vecteur entier
        if self.ndim() == 1 {
            match self.storage.as_ref() {
                StorageType::F32(data) => {
                    // Trouver le maximum pour la stabilité numérique
                    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    // Calculer exp(x_i - max) pour chaque élément
                    let mut exp_values: Vec<f32> = data.iter()
                        .map(|&x| (x - max_val).exp())
                        .collect();

                    // Calculer la somme des valeurs exp
                    let sum_exp: f32 = exp_values.iter().sum();

                    // Normaliser pour obtenir les probabilités
                    for val in &mut exp_values {
                        *val /= sum_exp;
                    }

                    // Créer le tenseur résultat
                    let mut result = self.clone();
                    result.storage = Arc::new(StorageType::from_f32(&exp_values));

                    Ok(result)
                },
                StorageType::F64(data) => {
                    // Même logique pour f64
                    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    let mut exp_values: Vec<f64> = data.iter()
                        .map(|&x| (x - max_val).exp())
                        .collect();

                    let sum_exp: f64 = exp_values.iter().sum();

                    for val in &mut exp_values {
                        *val /= sum_exp;
                    }

                    let mut result = self.clone();
                    result.storage = Arc::new(StorageType::from_f64(&exp_values));

                    Ok(result)
                },
                _ => Err(TensorError::new(
                    TensorErrorType::TypeError,
                    "Unsupported data type for softmax operation"
                )),
            }
        } else { // Cas 2D
            let shape = self.shape();
            let (rows, cols) = (shape[0], shape[1]);

            match self.storage.as_ref() {
                StorageType::F32(data) => {
                    let mut result_data = vec![0.0; data.len()];

                    // Appliquer softmax séparément sur chaque ligne ou colonne
                    if dim == 0 {
                        // Softmax sur les colonnes
                        for col in 0..cols {
                            // Trouver le maximum dans cette colonne
                            let mut max_val = f32::NEG_INFINITY;
                            for row in 0..rows {
                                max_val = max_val.max(data[row * cols + col]);
                            }

                            // Calculer exp(x - max) pour chaque élément dans cette colonne
                            let mut col_exp_sum = 0.0;
                            for row in 0..rows {
                                let idx = row * cols + col;
                                result_data[idx] = (data[idx] - max_val).exp();
                                col_exp_sum += result_data[idx];
                            }

                            // Normaliser
                            for row in 0..rows {
                                let idx = row * cols + col;
                                result_data[idx] /= col_exp_sum;
                            }
                        }
                    } else if dim == 1 {
                        // Softmax sur les lignes
                        for row in 0..rows {
                            let row_start = row * cols;
                            let row_end = row_start + cols;

                            // Trouver le maximum dans cette ligne
                            let mut max_val = f32::NEG_INFINITY;
                            for i in row_start..row_end {
                                max_val = max_val.max(data[i]);
                            }

                            // Calculer exp(x - max) pour chaque élément dans cette ligne
                            let mut row_exp_sum = 0.0;
                            for i in row_start..row_end {
                                result_data[i] = (data[i] - max_val).exp();
                                row_exp_sum += result_data[i];
                            }

                            // Normaliser
                            for i in row_start..row_end {
                                result_data[i] /= row_exp_sum;
                            }
                        }
                    }

                    let mut result = self.clone();
                    result.storage = Arc::new(StorageType::from_f32(&result_data));

                    Ok(result)
                },
                StorageType::F64(data) => {
                    let mut result_data = vec![0.0; data.len()];

                    // Même logique pour f64
                    if dim == 0 {
                        for col in 0..cols {
                            let mut max_val = f64::NEG_INFINITY;
                            for row in 0..rows {
                                max_val = max_val.max(data[row * cols + col]);
                            }

                            let mut col_exp_sum = 0.0;
                            for row in 0..rows {
                                let idx = row * cols + col;
                                result_data[idx] = (data[idx] - max_val).exp();
                                col_exp_sum += result_data[idx];
                            }

                            for row in 0..rows {
                                let idx = row * cols + col;
                                result_data[idx] /= col_exp_sum;
                            }
                        }
                    } else if dim == 1 {
                        for row in 0..rows {
                            let row_start = row * cols;
                            let row_end = row_start + cols;

                            let mut max_val = f64::NEG_INFINITY;
                            for i in row_start..row_end {
                                max_val = max_val.max(data[i]);
                            }

                            let mut row_exp_sum = 0.0;
                            for i in row_start..row_end {
                                result_data[i] = (data[i] - max_val).exp();
                                row_exp_sum += result_data[i];
                            }

                            for i in row_start..row_end {
                                result_data[i] /= row_exp_sum;
                            }
                        }
                    }

                    let mut result = self.clone();
                    result.storage = Arc::new(StorageType::from_f64(&result_data));

                    Ok(result)
                },
                _ => Err(TensorError::new(
                    TensorErrorType::TypeError,
                    "Unsupported data type for softmax operation"
                )),
            }
        }
    }

    /// Calcule le gradient de la fonction ReLU
    pub fn relu_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
        // Le gradient de ReLU est 1 si l'entrée est > 0, sinon 0
        self.apply_binary_op(grad_output,
                             |x, grad| if x > 0.0 { grad } else { 0.0 },
                             |x, grad| if x > 0.0 { grad } else { 0.0 }
        )
    }

    /// Calcule le gradient de la fonction sigmoid
    pub fn sigmoid_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
        // Le gradient de sigmoid est sigmoid(x) * (1 - sigmoid(x)) * grad_output
        let sigmoid_result = self.sigmoid()?;

        sigmoid_result.apply_binary_op(grad_output,
                                       |sig_x, grad| sig_x * (1.0 - sig_x) * grad,
                                       |sig_x, grad| sig_x * (1.0 - sig_x) * grad
        )
    }

    /// Calcule le gradient de la fonction tanh
    pub fn tanh_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
        // Le gradient de tanh est (1 - tanh(x)^2) * grad_output
        let tanh_result = self.tanh()?;

        tanh_result.apply_binary_op(grad_output,
                                    |tanh_x, grad| (1.0 - tanh_x * tanh_x) * grad,
                                    |tanh_x, grad| (1.0 - tanh_x * tanh_x) * grad
        )
    }
}

// Tests pour les optimisations du module tensor
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        let tensor = Tensor::from_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], None);

        // Test ReLU
        let relu_result = tensor.relu().unwrap();
        match relu_result.storage.as_ref() {
            StorageType::F32(data) => {
                assert_eq!(data, &[0.0, 0.0, 0.0, 1.0, 2.0]);
            },
            StorageType::F64(data) => {
                assert_eq!(data, &[0.0, 0.0, 0.0, 1.0, 2.0]);
            },
            _ => panic!("Unexpected storage type"),
        }

        // Test sigmoid
        let sigmoid_result = tensor.sigmoid().unwrap();
        match sigmoid_result.storage.as_ref() {
            StorageType::F32(data) => {
                assert!((data[0] - 0.1192).abs() < 0.0001); // sigmoid(-2) ≈ 0.1192
                assert!((data[2] - 0.5000).abs() < 0.0001); // sigmoid(0) = 0.5
                assert!((data[4] - 0.8808).abs() < 0.0001); // sigmoid(2) ≈ 0.8808
            },
            StorageType::F64(data) => {
                assert!((data[0] - 0.1192).abs() < 0.0001);
                assert!((data[2] - 0.5000).abs() < 0.0001);
                assert!((data[4] - 0.8808).abs() < 0.0001);
            },
            _ => panic!("Unexpected storage type"),
        }
    }

    #[test]
    fn test_softmax() {
        // Test softmax sur un vecteur 1D
        let tensor = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
        let softmax_result = tensor.softmax(None).unwrap();

        match softmax_result.storage.as_ref() {
            StorageType::F32(data) => {
                // La somme devrait être 1
                let sum: f32 = data.iter().sum();
                assert!((sum - 1.0).abs() < 0.0001);

                // Les valeurs devraient être croissantes
                assert!(data[0] < data[1]);
                assert!(data[1] < data[2]);
            },
            StorageType::F64(data) => {
                let sum: f64 = data.iter().sum();
                assert!((sum - 1.0).abs() < 0.0001);

                assert!(data[0] < data[1]);
                assert!(data[1] < data[2]);
            },
            _ => panic!("Unexpected storage type"),
        }

        // Test softmax sur un tenseur 2D
        let tensor_2d = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None);

        // Softmax le long des lignes (dim=1)
        let softmax_rows = tensor_2d.softmax(Some(1)).unwrap();

        match softmax_rows.storage.as_ref() {
            StorageType::F32(data) => {
                // La somme de chaque ligne devrait être 1
                let sum_row1: f32 = data[0..3].iter().sum();
                let sum_row2: f32 = data[3..6].iter().sum();

                assert!((sum_row1 - 1.0).abs() < 0.0001);
                assert!((sum_row2 - 1.0).abs() < 0.0001);
            },
            StorageType::F64(data) => {
                let sum_row1: f64 = data[0..3].iter().sum();
                let sum_row2: f64 = data[3..6].iter().sum();

                assert!((sum_row1 - 1.0).abs() < 0.0001);
                assert!((sum_row2 - 1.0).abs() < 0.0001);
            },
            _ => panic!("Unexpected storage type"),
        }
    }
}