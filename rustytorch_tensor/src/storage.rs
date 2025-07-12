// rustytorch_tensor/src/storage.rs

use num_complex::Complex;
use std::fmt;

/// Enum pour représenter différents types de stockage
#[derive(Debug, Clone)]
pub enum StorageType {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
    Complex64(Vec<Complex<f32>>),
    Complex128(Vec<Complex<f64>>),
}

impl PartialEq for StorageType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (StorageType::F32(a), StorageType::F32(b)) => a == b,
            (StorageType::F64(a), StorageType::F64(b)) => a == b,
            (StorageType::I8(a), StorageType::I8(b)) => a == b,
            (StorageType::I16(a), StorageType::I16(b)) => a == b,
            (StorageType::I32(a), StorageType::I32(b)) => a == b,
            (StorageType::I64(a), StorageType::I64(b)) => a == b,
            (StorageType::U8(a), StorageType::U8(b)) => a == b,
            (StorageType::U16(a), StorageType::U16(b)) => a == b,
            (StorageType::U32(a), StorageType::U32(b)) => a == b,
            (StorageType::U64(a), StorageType::U64(b)) => a == b,
            (StorageType::Bool(a), StorageType::Bool(b)) => a == b,
            (StorageType::Complex64(a), StorageType::Complex64(b)) => a == b,
            (StorageType::Complex128(a), StorageType::Complex128(b)) => a == b,
            _ => false,
        }
    }
}

impl fmt::Display for StorageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageType::F32(v) => write!(f, "F32({} elements)", v.len()),
            StorageType::F64(v) => write!(f, "F64({} elements)", v.len()),
            StorageType::I8(v) => write!(f, "I8({} elements)", v.len()),
            StorageType::I16(v) => write!(f, "I16({} elements)", v.len()),
            StorageType::I32(v) => write!(f, "I32({} elements)", v.len()),
            StorageType::I64(v) => write!(f, "I64({} elements)", v.len()),
            StorageType::U8(v) => write!(f, "U8({} elements)", v.len()),
            StorageType::U16(v) => write!(f, "U16({} elements)", v.len()),
            StorageType::U32(v) => write!(f, "U32({} elements)", v.len()),
            StorageType::U64(v) => write!(f, "U64({} elements)", v.len()),
            StorageType::Bool(v) => write!(f, "Bool({} elements)", v.len()),
            StorageType::Complex64(v) => write!(f, "Complex64({} elements)", v.len()),
            StorageType::Complex128(v) => write!(f, "Complex128({} elements)", v.len()),
        }
    }
}

impl StorageType {
    /// Crée un storage à partir de données f32
    pub fn from_f32(data: &[f32]) -> Self {
        StorageType::F32(data.to_vec())
    }

    /// Crée un storage à partir de données f64
    pub fn from_f64(data: &[f64]) -> Self {
        StorageType::F64(data.to_vec())
    }

    /// Crée un storage à partir de données i8
    pub fn from_i8(data: &[i8]) -> Self {
        StorageType::I8(data.to_vec())
    }

    /// Crée un storage à partir de données i16
    pub fn from_i16(data: &[i16]) -> Self {
        StorageType::I16(data.to_vec())
    }

    /// Crée un storage à partir de données i32
    pub fn from_i32(data: &[i32]) -> Self {
        StorageType::I32(data.to_vec())
    }

    /// Crée un storage à partir de données i64
    pub fn from_i64(data: &[i64]) -> Self {
        StorageType::I64(data.to_vec())
    }

    /// Crée un storage à partir de données u8
    pub fn from_u8(data: &[u8]) -> Self {
        StorageType::U8(data.to_vec())
    }

    /// Crée un storage à partir de données u16
    pub fn from_u16(data: &[u16]) -> Self {
        StorageType::U16(data.to_vec())
    }

    /// Crée un storage à partir de données u32
    pub fn from_u32(data: &[u32]) -> Self {
        StorageType::U32(data.to_vec())
    }

    /// Crée un storage à partir de données u64
    pub fn from_u64(data: &[u64]) -> Self {
        StorageType::U64(data.to_vec())
    }

    /// Crée un storage à partir de données bool
    pub fn from_bool(data: &[bool]) -> Self {
        StorageType::Bool(data.to_vec())
    }

    /// Crée un storage à partir de données complex64
    pub fn from_complex64(data: &[Complex<f32>]) -> Self {
        StorageType::Complex64(data.to_vec())
    }

    /// Crée un storage à partir de données complex128
    pub fn from_complex128(data: &[Complex<f64>]) -> Self {
        StorageType::Complex128(data.to_vec())
    }

    /// Renvoie la taille du stockage
    pub fn size(&self) -> usize {
        match self {
            StorageType::F32(data) => data.len(),
            StorageType::F64(data) => data.len(),
            StorageType::I8(data) => data.len(),
            StorageType::I16(data) => data.len(),
            StorageType::I32(data) => data.len(),
            StorageType::I64(data) => data.len(),
            StorageType::U8(data) => data.len(),
            StorageType::U16(data) => data.len(),
            StorageType::U32(data) => data.len(),
            StorageType::U64(data) => data.len(),
            StorageType::Bool(data) => data.len(),
            StorageType::Complex64(data) => data.len(),
            StorageType::Complex128(data) => data.len(),
        }
    }

    /// Alias for size() for compatibility
    pub fn numel(&self) -> usize {
        self.size()
    }

    /// Accède à un élément à l'index spécifié (pour le débogage et les tests)
    pub fn get_f64(&self, index: usize) -> Option<f64> {
        match self {
            StorageType::F32(data) => data.get(index).map(|&v| v as f64),
            StorageType::F64(data) => data.get(index).map(|&v| v),
            StorageType::I8(data) => data.get(index).map(|&v| v as f64),
            StorageType::I16(data) => data.get(index).map(|&v| v as f64),
            StorageType::I32(data) => data.get(index).map(|&v| v as f64),
            StorageType::I64(data) => data.get(index).map(|&v| v as f64),
            StorageType::U8(data) => data.get(index).map(|&v| v as f64),
            StorageType::U16(data) => data.get(index).map(|&v| v as f64),
            StorageType::U32(data) => data.get(index).map(|&v| v as f64),
            StorageType::U64(data) => data.get(index).map(|&v| v as f64),
            StorageType::Bool(data) => data.get(index).map(|&v| if v { 1.0 } else { 0.0 }),
            StorageType::Complex64(data) => data.get(index).map(|&v| v.norm() as f64),
            StorageType::Complex128(data) => data.get(index).map(|&v| v.norm()),
        }
    }

    /// Convertit tout le stockage en Vec<f64>
    pub fn to_vec_f64(&self) -> Vec<f64> {
        match self {
            StorageType::F32(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::F64(data) => data.clone(),
            StorageType::I8(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::I16(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::I32(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::I64(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::U8(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::U16(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::U32(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::U64(data) => data.iter().map(|&v| v as f64).collect(),
            StorageType::Bool(data) => data.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect(),
            StorageType::Complex64(data) => data.iter().map(|&v| v.norm() as f64).collect(),
            StorageType::Complex128(data) => data.iter().map(|&v| v.norm()).collect(),
        }
    }

    /// Crée un nouveau storage rempli de zéros du même type et de la même taille
    pub fn zeros_like(&self) -> Self {
        match self {
            StorageType::F32(data) => StorageType::F32(vec![0.0; data.len()]),
            StorageType::F64(data) => StorageType::F64(vec![0.0; data.len()]),
            StorageType::I8(data) => StorageType::I8(vec![0; data.len()]),
            StorageType::I16(data) => StorageType::I16(vec![0; data.len()]),
            StorageType::I32(data) => StorageType::I32(vec![0; data.len()]),
            StorageType::I64(data) => StorageType::I64(vec![0; data.len()]),
            StorageType::U8(data) => StorageType::U8(vec![0; data.len()]),
            StorageType::U16(data) => StorageType::U16(vec![0; data.len()]),
            StorageType::U32(data) => StorageType::U32(vec![0; data.len()]),
            StorageType::U64(data) => StorageType::U64(vec![0; data.len()]),
            StorageType::Bool(data) => StorageType::Bool(vec![false; data.len()]),
            StorageType::Complex64(data) => {
                StorageType::Complex64(vec![Complex::new(0.0, 0.0); data.len()])
            }
            StorageType::Complex128(data) => {
                StorageType::Complex128(vec![Complex::new(0.0, 0.0); data.len()])
            }
        }
    }

    /// Crée un nouveau storage rempli de uns du même type et de la même taille
    pub fn ones_like(&self) -> Self {
        match self {
            StorageType::F32(data) => StorageType::F32(vec![1.0; data.len()]),
            StorageType::F64(data) => StorageType::F64(vec![1.0; data.len()]),
            StorageType::I8(data) => StorageType::I8(vec![1; data.len()]),
            StorageType::I16(data) => StorageType::I16(vec![1; data.len()]),
            StorageType::I32(data) => StorageType::I32(vec![1; data.len()]),
            StorageType::I64(data) => StorageType::I64(vec![1; data.len()]),
            StorageType::U8(data) => StorageType::U8(vec![1; data.len()]),
            StorageType::U16(data) => StorageType::U16(vec![1; data.len()]),
            StorageType::U32(data) => StorageType::U32(vec![1; data.len()]),
            StorageType::U64(data) => StorageType::U64(vec![1; data.len()]),
            StorageType::Bool(data) => StorageType::Bool(vec![true; data.len()]),
            StorageType::Complex64(data) => {
                StorageType::Complex64(vec![Complex::new(1.0, 0.0); data.len()])
            }
            StorageType::Complex128(data) => {
                StorageType::Complex128(vec![Complex::new(1.0, 0.0); data.len()])
            }
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
            }
            _ => panic!("Expected F32 storage"),
        }

        match ones {
            StorageType::F32(data) => {
                assert_eq!(data, vec![1.0, 1.0, 1.0]);
            }
            _ => panic!("Expected F32 storage"),
        }
    }
}
