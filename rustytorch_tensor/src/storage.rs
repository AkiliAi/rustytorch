// rustytorch_tensor/src/storage.rs


#[derive(Clone,Debug,PartialEq,)]
pub enum StorageType {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
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

}

