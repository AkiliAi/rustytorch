// use rustytorch_autograd::Variable;
// use rustytorch_tensor::Tensor;
// use crate::{Module, ModuleBase, ModuleState};
//
// /// Module d'activation ReLU (Rectified Linear Unit)
// pub struct ReLU{
//     base: ModuleBase,
//     /// Pente pour les valeurs négatives Leaky ReLU si > 0
//     negative_slope: f64,
// }
//
// impl ReLU {
//     pub fn new() -> Self{
//         Self{
//             base: ModuleBase::new(),
//             negative_slope: 0.0,
//         }
//     }
//
//     pub fn leaky_relu(negative_slope:f64) -> Self{
//         Self{
//             base: ModuleBase::new(),
//             negative_slope,
//         }
//     }
// }
//
// impl Module for ReLU {
//     fn forward(&self, input: &Variable) -> Variable {
//         if self.negative_slope == 0.0 {
//             // ReLU standard: max(0, x)
//             input.relu()
//         } else {
//             // LeakyReLU: max(0, x) + negative_slope * min(0, x)
//             let pos = input.relu();
//             let neg = input.mul(&Variable::from_tensor(
//                 // Tensor::from_data(&[self.negative_slope], vec![1], None).expect("Failed to create tensor"),
//                 Tensor::from_data(&[self.negative_slope], vec![1], None),
//
//                 false
//             ));
//
//             let zeros = Variable::from_tensor(
//                 // Tensor::zeros(input.tensor.shape().to_vec(), None).expect("Failed to create zeros tensor"),
//                 Tensor::zeros(input.tensor.shape().to_vec(), None),
//                 false
//             );
//
//             let mask = input.tensor.lt(&zeros.tensor);
//             // let mask_var = Variable::from_tensor(
//             //     mask.to_f64().expect("Failed to convert mask to f64"),
//             //     false
//             // );
//             let mask_var = Variable::from_tensor(
//                 mask.to_f64().expect("Failed to convert mask to f64"),
//                 false
//             );
//
//             pos.add(&neg.mul(&mask_var))
//         }
//     }
//
//     fn parameters(&self) -> Vec<Variable> {
//         // ReLU n'a pas de paramètres entraînables
//         vec![]
//     }
//
//     fn train(&mut self) {
//         self.base.state = ModuleState::Train;
//     }
//
//     fn eval(&mut self) {
//         self.base.state = ModuleState::Eval;
//     }
//
//     fn is_training(&self) -> bool {
//         self.base.state == ModuleState::Train
//     }
// }
//
// /// Module d'activation Sigmoid
// pub struct Sigmoid {
//     base: ModuleBase,
// }
//
// impl Sigmoid {
//     /// Crée un nouveau module Sigmoid
//     pub fn new() -> Self {
//         Self {
//             base: ModuleBase::new(),
//         }
//     }
// }
//
// impl Module for Sigmoid {
//     fn forward(&self, input: &Variable) -> Variable {
//         input.sigmoid()
//     }
//
//     fn parameters(&self) -> Vec<Variable> {
//         // Sigmoid n'a pas de paramètres entraînables
//         vec![]
//     }
//
//     fn train(&mut self) {
//         self.base.state = ModuleState::Train;
//     }
//
//     fn eval(&mut self) {
//         self.base.state = ModuleState::Eval;
//     }
//
//     fn is_training(&self) -> bool {
//         self.base.state == ModuleState::Train
//     }
// }
//
// /// Module conteneur pour une séquence de modules
// pub struct Sequential {
//     base: ModuleBase,
//     modules: Vec<Box<dyn Module>>,
// }
//
// impl Sequential {
//     /// Crée un nouveau module Sequential vide
//     pub fn new() -> Self {
//         Self {
//             base: ModuleBase::new(),
//             modules: Vec::new(),
//         }
//     }
//
//     /// Ajoute un module à la séquence
//     pub fn add(&mut self, module: Box<dyn Module>) {
//         self.modules.push(module);
//     }
// }
//
// impl Module for Sequential {
//     fn forward(&self, input: &Variable) -> Variable {
//         let mut current = input.clone();
//
//         for module in &self.modules {
//             current = module.forward(&current);
//         }
//
//         current
//     }
//
//     fn parameters(&self) -> Vec<Variable> {
//         let mut params = Vec::new();
//
//         for module in &self.modules {
//             params.extend(module.parameters());
//         }
//
//         params
//     }
//
//     fn train(&mut self) {
//         self.base.state = ModuleState::Train;
//
//         for module in &mut self.modules {
//             module.train();
//         }
//     }
//
//     fn eval(&mut self) {
//         self.base.state = ModuleState::Eval;
//
//         for module in &mut self.modules {
//             module.eval();
//         }
//     }
//
//     fn is_training(&self) -> bool {
//         self.base.state == ModuleState::Train
//     }
// }
