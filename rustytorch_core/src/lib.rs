//rustytorch_core/src/lib.rs


/// Trait pour les types prenant en charge les opérations mathematiques de base
pub trait NumericOps<Rhs =Self> {
    type Output;

    fn add(self,rhs:Rhs) -> Self::Output;
    fn sub(self,rhs:Rhs) -> Self::Output;
    fn mul(self,rhs:Rhs) -> Self::Output;
    fn div(self,rhs:Rhs) -> Self::Output;

}

/// Trait pour les types de supportant les opérations de reduction
pub trait Reduction {
    type Output;

    fn sum(&self) -> Self::Output;
    fn mean(&self) -> Self::Output;
    fn max(&self) -> Self::Output;
    fn min(&self) -> Self::Output;
}


/// Trait pour les types pouvant etre convertis en differents formes
pub trait Reshapable<TensorError> {
    // fn reshape(&self,shape: &[usize]) -> Self;
    // fn flatten(&self) -> Self;
    // fn transpose(&self,dim0:usize,dim1:usize) -> Self;
    fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError> where Self: Sized;
    fn flatten(&self) -> Result<Self, TensorError> where Self: Sized;
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, TensorError> where Self: Sized;
}


/// Trait pour le Broadcasting

pub trait Differentiable {
    type Gradient;

    fn backward(&self);
    fn grad(&self) -> Option<Self::Gradient>;
    fn requires_grad(&self) -> bool;

    fn set_requires_grad(&mut self, requires_grad: bool);
    fn detach(&self) -> Self;
}


/// Trait pour les types supportant la serialisation/deserialisation
pub trait Serialization{
    fn save(&self,path:&str) -> std::io::Result<()>;
    fn load(path:&str) -> std::io::Result<Self> where Self: Sized;
}


/// Type de données pour les tenseurs
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum Dtype {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}


#[derive(Clone, Debug, PartialEq)]
pub struct TensorOptions{
    pub dtype: Dtype,
    pub requires_grad: bool,
    pub device: Device,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device{
    CPU,
    // CUDA(u32),
    CUDA(usize),

}



impl Default for TensorOptions{
    fn default() -> Self {
        Self {
            dtype: Dtype::Float32,
            requires_grad: false,
            device: Device::CPU,
            // device: Device,
        }
    }
}





























