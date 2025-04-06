// rustytorch_tensor/src/storage.rs


use std::fmt::{Debug, Display, Formatter};

#[derive(Clone,Debug,PartialEq,)]
pub enum StorageType {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}


// impl PartialEq for StorageType {
//     fn eq(&self, other: &Self) -> bool {
//         match (self, other) {
//             (StorageType::F32(a), StorageType::F32(b)) => a == b,
//             (StorageType::F64(a), StorageType::F64(b)) => a == b,
//             (StorageType::I32(a), StorageType::I32(b)) => a == b,
//             (StorageType::I64(a), StorageType::I64(b)) => a == b,
//             (StorageType::Bool(a), StorageType::Bool(b)) => a == b,
//             _ => false,
//         }
//     }
// }


impl Display for StorageType{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageType::F32(v) => write!(f, "F32({} elements)", v.len()),
            StorageType::F64(v) => write!(f, "F64({} elements)", v.len()),
            StorageType::I32(v) => write!(f, "I32({} elements)", v.len()),
            StorageType::I64(v) => write!(f, "I64({} elements)", v.len()),
            StorageType::Bool(v) => write!(f, "Bool({} elements)", v.len()),

        }
    }
}


impl StorageType {
    pub fn from_f32(data: &[f32]) -> Self{
        StorageType::F32(data.to_vec())
    }

    pub fn from_f64(data: &[f64]) -> Self{
        StorageType::F64(data.to_vec())
    }

    pub fn from_i32(data: &[i32]) -> Self{
        StorageType::I32(data.to_vec())
    }

    pub fn from_i64(data: &[i64]) -> Self{
        StorageType::I64(data.to_vec())
    }


    pub fn from_bool(data: &[bool]) -> Self{
        StorageType::Bool(data.to_vec())
    }


    pub fn size(&self) -> usize {
        match self {
            StorageType::F32(data) => data.len(),
            StorageType::F64(data) => data.len(),
            StorageType::I32(data) => data.len(),
            StorageType::I64(data) => data.len(),
            StorageType::Bool(data) => data.len(),
        }
    }

    /// Acceder a un element a l'index specifie (pour le debogage et les tests)
    pub fn get_f64(&self, index: usize) -> Option<f64> {
        match self {
            StorageType::F32(data) => data.get(index).map(|&v| v as f64),
            StorageType::F64(data) => data.get(index).map(|&v| v),
            StorageType::I32(data) => data.get(index).map(|&v| v as f64),
            StorageType::I64(data) => data.get(index).map(|&v| v as f64),
            StorageType::Bool(data) => data.get(index).map(|&v| if v { 1.0 } else { 0.0 }),
        }
    }

    /// Convertit tout le stockage en Vec<f64>
    pub fn to_vec_f64(&self) -> Vec<f64> {
        match self {
            StorageType::F32(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::F64(data) => data.clone(),
            StorageType::I32(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::I64(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::Bool(data) => data.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect(),
        }
    }

    /// Crée un nouveau storage rempli de zéros du même type et de la même taille
    pub fn zeros_like(&self) -> Self {
        match self {
            StorageType::F32(data) => StorageType::F32(vec![0.0; data.len()]),
            StorageType::F64(data) => StorageType::F64(vec![0.0; data.len()]),
            StorageType::I32(data) => StorageType::I32(vec![0; data.len()]),
            StorageType::I64(data) => StorageType::I64(vec![0; data.len()]),
            StorageType::Bool(data) => StorageType::Bool(vec![false; data.len()]),
        }
    }

    /// Crée un nouveau storage rempli de uns du même type et de la même taille
    pub fn ones_like(&self) -> Self {
        match self {
            StorageType::F32(data) => StorageType::F32(vec![1.0; data.len()]),
            StorageType::F64(data) => StorageType::F64(vec![1.0; data.len()]),
            StorageType::I32(data) => StorageType::I32(vec![1; data.len()]),
            StorageType::I64(data) => StorageType::I64(vec![1; data.len()]),
            StorageType::Bool(data) => StorageType::Bool(vec![true; data.len()]),
        }
    }

}

// Tests pour le module storage
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_creation() {
        let storage_f32 = StorageType::from_f32(&[1.0, 2.0, 3.0]);
        let storage_f64 = StorageType::from_f64(&[1.0, 2.0, 3.0]);

        assert_eq!(storage_f32.size(), 3);
        assert_eq!(storage_f64.size(), 3);
    }

    #[test]
    fn test_storage_get() {
        let storage = StorageType::from_f32(&[1.0, 2.0, 3.0]);

        assert_eq!(storage.get_f64(0), Some(1.0));
        assert_eq!(storage.get_f64(1), Some(2.0));
        assert_eq!(storage.get_f64(2), Some(3.0));
        assert_eq!(storage.get_f64(3), None);
    }

    #[test]
    fn test_storage_to_vec_f64() {
        let storage = StorageType::from_f32(&[1.0, 2.0, 3.0]);
        let vec_f64 = storage.to_vec_f64();

        assert_eq!(vec_f64, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_storage_zeros_ones_like() {
        let storage = StorageType::from_f32(&[1.0, 2.0, 3.0]);

        let zeros = storage.zeros_like();
        let ones = storage.ones_like();

        match zeros {
            StorageType::F32(data) => {
                assert_eq!(data, vec![0.0, 0.0, 0.0]);
            },
            _ => panic!("Expected F32 storage"),
        }

        match ones {
            StorageType::F32(data) => {
                assert_eq!(data, vec![1.0, 1.0, 1.0]);
            },
            _ => panic!("Expected F32 storage"),
        }
    }
}