// pub trait Backend {
//     fn add<T: NumericOps>(a: &[T], b: &[T], result: &mut [T]);
//     fn mul<T: Numeric>(a: &[T], b: &[T], result: &mut [T]);
//     // Autres opérations...
// }
//
// pub struct CPUBackend;
// pub struct CUDABackend;
//
// impl Backend for CPUBackend {
//     fn add<T: Numeric>(a: &[T], b: &[T], result: &mut [T]) {
//         // Implémentation CPU
//     }
//     // ...
// }
//
// impl Backend for CUDABackend {
//     fn add<T: Numeric>(a: &[T], b: &[T], result: &mut [T]) {
//         // Implémentation CUDA
//     }
//     // ...
// }
//
// // Tenseur générique sur le backend
// pub struct GenericTensor<B: Backend> {
//     data: Vec<f32>,
//     shape: Vec<usize>,
//     _backend: std::marker::PhantomData<B>,
// }
//
// impl<B: Backend> GenericTensor<B> {
//     pub fn add(&self, other: &Self) -> Self {
//         let mut result = Self::zeros(self.shape.clone());
//         B::add(&self.data, &other.data, &mut result.data);
//         result
//     }
// }
//
// // Aliases pour plus de lisibilité
// pub type CPUTensor = GenericTensor<CPUBackend>;
// pub type CUDATensor = GenericTensor<CUDABackend>;
