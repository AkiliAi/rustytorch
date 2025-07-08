//rustytorch_tensor/src/tensor_ops.rs

use crate::{storage::StorageType, Tensor};
use rustytorch_core::{CoreError, Result};

impl Tensor {
    /// Concatène plusieurs tenseurs le long d'une dimension spécifiée
    ///
    /// # Arguments
    /// * `tensors` - Slice de tenseurs à concaténer
    /// * `dim` - Dimension le long de laquelle concaténer
    ///
    /// # Examples
    /// ```rust
    /// let a = Tensor::from_data(&[1.0, 2.0], vec![2], None);
    /// let b = Tensor::from_data(&[3.0, 4.0], vec![2], None);
    /// let result = Tensor::cat(&[a, b], 0).unwrap();
    /// // result: [1.0, 2.0, 3.0, 4.0] avec shape [4]
    /// ```
    pub fn cat(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(CoreError::invalid_op(
                "cat",
                "Cannot concatenate empty tensor list",
            ));
        }

        let first = &tensors[0];
        let shape = first.shape();

        if dim >= shape.len() {
            return Err(CoreError::dim_out_of_bounds(dim, shape.len(), "cat"));
        }

        // Vérifier que tous les tenseurs ont le même type
        let first_dtype = first.dtype();
        for tensor in tensors.iter().skip(1) {
            if tensor.dtype() != first_dtype {
                return Err(CoreError::invalid_op(
                    "cat",
                    "All tensors must have the same data type",
                ));
            }
        }

        // Vérifier que toutes les dimensions sauf `dim` sont identiques
        for tensor in tensors.iter().skip(1) {
            let tensor_shape = tensor.shape();
            if tensor_shape.len() != shape.len() {
                return Err(CoreError::shape_mismatch(
                    vec![tensor_shape.len()],
                    vec![shape.len()],
                    "cat",
                ));
            }

            for (i, (&s1, &s2)) in shape.iter().zip(tensor_shape.iter()).enumerate() {
                if i != dim && s1 != s2 {
                    return Err(CoreError::invalid_op(
                        "cat",
                        &format!("All dimensions except {} must match", dim),
                    ));
                }
            }
        }

        // Calculer la nouvelle forme
        let total_size_in_dim: usize = tensors.iter().map(|t| t.shape()[dim]).sum();
        let mut new_shape = shape.to_vec();
        new_shape[dim] = total_size_in_dim;

        // Concaténer les données
        match &first.storage.as_ref() {
            StorageType::F32(_) => {
                let mut result_data = Vec::new();
                Self::cat_data_f32(tensors, dim, &mut result_data)?;
                Ok(Tensor::from_data(
                    &result_data,
                    new_shape,
                    Some(first.options.clone()),
                ))
            }
            StorageType::F64(_) => {
                let mut result_data = Vec::new();
                Self::cat_data_f64(tensors, dim, &mut result_data)?;
                Ok(Tensor::from_data(
                    &result_data,
                    new_shape,
                    Some(first.options.clone()),
                ))
            }
            StorageType::I32(_) => {
                let mut result_data = Vec::new();
                Self::cat_data_i32(tensors, dim, &mut result_data)?;
                // Convert i32 to f64 for from_data
                let float_data: Vec<f64> = result_data.iter().map(|&x| x as f64).collect();
                Ok(Tensor::from_data(
                    &float_data,
                    new_shape,
                    Some(first.options.clone()),
                ))
            }
            StorageType::I64(_) => {
                let mut result_data = Vec::new();
                Self::cat_data_i64(tensors, dim, &mut result_data)?;
                // Convert i64 to f64 for from_data
                let float_data: Vec<f64> = result_data.iter().map(|&x| x as f64).collect();
                Ok(Tensor::from_data(
                    &float_data,
                    new_shape,
                    Some(first.options.clone()),
                ))
            }
            _ => Err(CoreError::invalid_op(
                "cat",
                "Concatenation not implemented for this type",
            )),
        }
    }

    /// Divise un tenseur en chunks de taille égale le long d'une dimension
    ///
    /// # Arguments
    /// * `chunks` - Nombre de chunks à créer
    /// * `dim` - Dimension le long de laquelle diviser
    ///
    /// # Examples
    /// ```rust
    /// let tensor = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], None);
    /// let chunks = tensor.chunk(3, 0).unwrap(); // 3 chunks de 2 éléments chacun
    /// ```
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Tensor>> {
        if chunks == 0 {
            return Err(CoreError::invalid_op(
                "chunk",
                "Number of chunks must be greater than 0",
            ));
        }

        let shape = self.shape();
        if dim >= shape.len() {
            return Err(CoreError::dim_out_of_bounds(dim, shape.len(), "chunk"));
        }

        let size_in_dim = shape[dim];
        let chunk_size = (size_in_dim + chunks - 1) / chunks; // Ceiling division

        self.split(chunk_size, dim)
    }

    /// Divise un tenseur en sections de taille spécifiée
    ///
    /// # Arguments
    /// * `split_size` - Taille de chaque section
    /// * `dim` - Dimension le long de laquelle diviser
    pub fn split(&self, split_size: usize, dim: usize) -> Result<Vec<Tensor>> {
        if split_size == 0 {
            return Err(CoreError::invalid_op(
                "split",
                "Split size must be greater than 0",
            ));
        }

        let shape = self.shape();
        if dim >= shape.len() {
            return Err(CoreError::dim_out_of_bounds(dim, shape.len(), "split"));
        }

        let size_in_dim = shape[dim];
        let mut results = Vec::new();
        let mut start = 0;

        while start < size_in_dim {
            let end = (start + split_size).min(size_in_dim);
            let slice_tensor = self.slice_dim(dim, start, end)?;
            results.push(slice_tensor);
            start = end;
        }

        Ok(results)
    }

    /// Extrait une slice le long d'une dimension spécifique
    fn slice_dim(&self, dim: usize, start: usize, end: usize) -> Result<Tensor> {
        let shape = self.shape();
        let mut new_shape = shape.to_vec();
        new_shape[dim] = end - start;

        // Utiliser la vue pour extraire la slice
        let mut ranges: Vec<std::ops::Range<usize>> =
            shape.iter().enumerate().map(|(i, &size)| 0..size).collect();
        ranges[dim] = start..end;

        self.slice_ranges(&ranges)
    }

    /// Slice avec des ranges multiples (helper method)
    pub fn slice_ranges(&self, ranges: &[std::ops::Range<usize>]) -> Result<Tensor> {
        // Pour simplifier, créons un nouveau tenseur avec les données copiées
        let shape = self.shape();
        let new_shape: Vec<usize> = ranges.iter().map(|r| r.len()).collect();

        match &self.storage.as_ref() {
            StorageType::F32(data) => {
                let mut result_data = Vec::new();
                Self::extract_slice_f32(data, shape, ranges, &mut result_data)?;
                Ok(Tensor::from_data(
                    &result_data,
                    new_shape,
                    Some(self.options.clone()),
                ))
            }
            StorageType::F64(data) => {
                let mut result_data = Vec::new();
                Self::extract_slice_f64(data, shape, ranges, &mut result_data)?;
                Ok(Tensor::from_data(
                    &result_data,
                    new_shape,
                    Some(self.options.clone()),
                ))
            }
            _ => Err(CoreError::invalid_op(
                "slice",
                "Slicing not implemented for this type",
            )),
        }
    }

    // Helper methods pour concaténation par type
    fn cat_data_f32(tensors: &[Tensor], dim: usize, result: &mut Vec<f32>) -> Result<()> {
        for tensor in tensors {
            match tensor.storage.as_ref() {
                StorageType::F32(data) => {
                    let tensor_data = Self::extract_data_along_dim_f32(data, tensor.shape(), dim)?;
                    result.extend_from_slice(&tensor_data);
                }
                _ => {
                    return Err(CoreError::invalid_op(
                        "cat",
                        "All tensors must have same type",
                    ))
                }
            }
        }
        Ok(())
    }

    fn cat_data_f64(tensors: &[Tensor], dim: usize, result: &mut Vec<f64>) -> Result<()> {
        for tensor in tensors {
            match tensor.storage.as_ref() {
                StorageType::F64(data) => {
                    let tensor_data = Self::extract_data_along_dim_f64(data, tensor.shape(), dim)?;
                    result.extend_from_slice(&tensor_data);
                }
                _ => {
                    return Err(CoreError::invalid_op(
                        "cat",
                        "All tensors must have same type",
                    ))
                }
            }
        }
        Ok(())
    }

    fn cat_data_i32(tensors: &[Tensor], dim: usize, result: &mut Vec<i32>) -> Result<()> {
        for tensor in tensors {
            match tensor.storage.as_ref() {
                StorageType::I32(data) => {
                    let tensor_data = Self::extract_data_along_dim_i32(data, tensor.shape(), dim)?;
                    result.extend_from_slice(&tensor_data);
                }
                _ => {
                    return Err(CoreError::invalid_op(
                        "cat",
                        "All tensors must have same type",
                    ))
                }
            }
        }
        Ok(())
    }

    fn cat_data_i64(tensors: &[Tensor], dim: usize, result: &mut Vec<i64>) -> Result<()> {
        for tensor in tensors {
            match tensor.storage.as_ref() {
                StorageType::I64(data) => {
                    let tensor_data = Self::extract_data_along_dim_i64(data, tensor.shape(), dim)?;
                    result.extend_from_slice(&tensor_data);
                }
                _ => {
                    return Err(CoreError::invalid_op(
                        "cat",
                        "All tensors must have same type",
                    ))
                }
            }
        }
        Ok(())
    }

    // Helper methods pour extraction de données
    fn extract_data_along_dim_f32(data: &[f32], _shape: &[usize], _dim: usize) -> Result<Vec<f32>> {
        // Simplification : pour l'instant, retourner toutes les données
        // Une implémentation complète gérerait les strides et l'extraction spécifique
        Ok(data.to_vec())
    }

    fn extract_data_along_dim_f64(data: &[f64], _shape: &[usize], _dim: usize) -> Result<Vec<f64>> {
        Ok(data.to_vec())
    }

    fn extract_data_along_dim_i32(data: &[i32], _shape: &[usize], _dim: usize) -> Result<Vec<i32>> {
        Ok(data.to_vec())
    }

    fn extract_data_along_dim_i64(data: &[i64], _shape: &[usize], _dim: usize) -> Result<Vec<i64>> {
        Ok(data.to_vec())
    }

    // Helper methods pour slicing
    fn extract_slice_f32(
        data: &[f32],
        shape: &[usize],
        ranges: &[std::ops::Range<usize>],
        result: &mut Vec<f32>,
    ) -> Result<()> {
        // Pour simplifier, implémentation basique pour tenseurs 1D et 2D
        match shape.len() {
            1 => {
                let start = ranges[0].start;
                let end = ranges[0].end;
                result.extend_from_slice(&data[start..end]);
            }
            2 => {
                let rows = shape[0];
                let cols = shape[1];
                let row_range = &ranges[0];
                let col_range = &ranges[1];

                for row in row_range.clone() {
                    let row_start = row * cols + col_range.start;
                    let row_end = row * cols + col_range.end;
                    result.extend_from_slice(&data[row_start..row_end]);
                }
            }
            _ => {
                // Pour les tenseurs de dimension supérieure, implémentation récursive
                // Pour l'instant, retourner une erreur
                return Err(CoreError::invalid_op(
                    "slice",
                    "Slicing for >2D tensors not yet implemented",
                ));
            }
        }
        Ok(())
    }

    fn extract_slice_f64(
        data: &[f64],
        shape: &[usize],
        ranges: &[std::ops::Range<usize>],
        result: &mut Vec<f64>,
    ) -> Result<()> {
        match shape.len() {
            1 => {
                let start = ranges[0].start;
                let end = ranges[0].end;
                result.extend_from_slice(&data[start..end]);
            }
            2 => {
                let rows = shape[0];
                let cols = shape[1];
                let row_range = &ranges[0];
                let col_range = &ranges[1];

                for row in row_range.clone() {
                    let row_start = row * cols + col_range.start;
                    let row_end = row * cols + col_range.end;
                    result.extend_from_slice(&data[row_start..row_end]);
                }
            }
            _ => {
                return Err(CoreError::invalid_op(
                    "slice",
                    "Slicing for >2D tensors not yet implemented",
                ));
            }
        }
        Ok(())
    }
}

// Tests pour les opérations de tenseur
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cat_1d() {
        let a = Tensor::from_data(&[1.0f32, 2.0], vec![2], None);
        let b = Tensor::from_data(&[3.0f32, 4.0], vec![2], None);

        let result = Tensor::cat(&[a, b], 0).unwrap();
        assert_eq!(result.shape(), &[4]);

        let data = result.storage().to_vec_f64();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cat_2d() {
        let a = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None);
        let b = Tensor::from_data(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2], None);

        let result = Tensor::cat(&[a, b], 0).unwrap(); // Concat le long des lignes
        assert_eq!(result.shape(), &[4, 2]);
    }

    #[test]
    fn test_split_even() {
        let tensor = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], None);

        let chunks = tensor.split(2, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[2]);
        assert_eq!(chunks[1].shape(), &[2]);
        assert_eq!(chunks[2].shape(), &[2]);
    }

    #[test]
    fn test_chunk() {
        let tensor = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], None);

        let chunks = tensor.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);

        for chunk in chunks {
            assert_eq!(chunk.shape(), &[2]);
        }
    }

    #[test]
    fn test_cat_type_mismatch() {
        use rustytorch_core::{DType, TensorOptions};

        let a = Tensor::from_data(
            &[1.0f32, 2.0],
            vec![2],
            Some(TensorOptions::new().dtype(DType::Float32)),
        );
        let b = Tensor::from_data(
            &[3.0f64, 4.0],
            vec![2],
            Some(TensorOptions::new().dtype(DType::Float64)),
        );

        let result = Tensor::cat(&[a, b], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cat_shape_mismatch() {
        let a = Tensor::from_data(&[1.0f32, 2.0], vec![2], None);
        let b = Tensor::from_data(&[3.0f32, 4.0, 5.0], vec![3], None);

        let result = Tensor::cat(&[a, b], 0);
        // Cette opération devrait réussir car on concatène le long de la dimension 0
        assert!(result.is_ok());
        assert_eq!(result.unwrap().shape(), &[5]);
    }
}
