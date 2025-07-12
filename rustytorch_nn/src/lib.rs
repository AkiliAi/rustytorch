// //rustytorch_nn/src/lib.rs
//
// mod activations;
// mod nn_errors;
//
// use crate::nn_errors::NNError;
// use crate::InitMethod::Normal;
// use rand::Rng;
// use rustytorch_autograd::Variable;
// use rustytorch_core::Reshapable;
// use rustytorch_tensor::Tensor;
// use std::collections::HashMap;
// use std::error::Error;
// use std::fmt;
// use std::sync::Arc;
//
// /// Trait fondamental pour les modules de réseau de neurones
// pub trait Module {
//     /// Effecture une passe avant (forward) du module
//     // fn forward(&self,input:&Variable) -> Result<Variable, Box<dyn Error>>;
//     fn forward(&self, input: &Variable) -> Variable;
//
//     /// Renvoie les paramètres du module
//     fn parameters(&self) -> Vec<Variable>;
//
//     /// Renvoie le nom du module
//     fn train(&mut self);
//
//     /// Met le Module en mode évaluation
//     fn eval(&mut self);
//
//     /// Indique si le module est en mode d'entraînement
//     fn is_training(&self) -> bool;
// }
//
// /// Trait pour les modules qui peuvent être initialisés avec des poids spécifiques
// pub trait Initializable {
//     /// Initialise les poids du module selon une méthode donnée
//     fn init_weights(&mut self, method: InitMethod);
// }
//
// /// Enumération des méthodes d'initialisation disponibles
// pub enum InitMethod {
//     /// Initialisation uniforme dans l'intervalle [-scale, scale]
//     Uniform { scale: f64 },
//     /// Initialisation normale avec moyenne et écart-type donnés
//     Normal { mean: f64, std: f64 },
//     /// Initialisation Xavier/Glorot
//     Xavier,
//     /// Initialisation Kaiming/He
//     Kaiming,
//     /// Initialisation avec des valeurs constantes
//     Constant { value: f64 },
// }
//
// /// Etat de formation pour les modules
// #[derive(Clone, Copy, Debug, PartialEq)]
// pub enum ModuleState {
//     Train, //mode entraînement
//     Eval,  //mode évaluation
// }
//
// /// MOdule de base avec état partage
// pub struct ModuleBase {
//     pub state: ModuleState,
//     // pub parameters: Vec<Variable>,
// }
//
// impl ModuleBase {
//     pub fn new() -> Self {
//         Self {
//             state: ModuleState::Train,
//         }
//     }
// }
//
// /// Couche linéaire (ou "Fully Connected Layer")
// pub struct Linear {
//     base: ModuleBase,
//     in_features: usize,
//     out_features: usize,
//     weight: Variable,       // Poids (matrice de taille out_features x in_features)
//     bias: Option<Variable>, // Biais (vecteur de taille out_features)
// }
//
// impl Linear {
//     /// Crée une nouvelle couche linéaire
//     pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
//         // Initialiser les poids avec Xavier/Glorot par défaut
//         let weight_scale = (6.0 / (in_features + out_features) as f64).sqrt();
//
//         let weight_data: Vec<f64> = (0..in_features * out_features)
//             .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * weight_scale)
//             .collect();
//
//         let weight_tensor = Tensor::from_data(&weight_data, vec![out_features, in_features], None);
//
//         let bias_var = if bias {
//             let bias_data: Vec<f64> = vec![0.0; out_features];
//             let bias_tensor = Tensor::from_data(&bias_data, vec![out_features], None);
//
//             Some(Variable::from_tensor(bias_tensor, true))
//         } else {
//             None
//         };
//
//         Self {
//             base: ModuleBase::new(),
//             in_features,
//             out_features,
//             weight: Variable::from_tensor(weight_tensor, true),
//             bias: bias_var,
//         }
//     }
// }
//
// impl Module for Linear {
//     fn forward(&self, input: &Variable) -> Variable {
//         // Transposer les poids pour multiplication matricielle
//         let weight_t = Variable::from_tensor(
//             self.weight
//                 .tensor
//                 .transpose(0, 1)
//                 .expect("Failed to transpose weights"),
//             self.weight.requires_grad,
//         );
//
//         // Multiplication matricielle: x @ W^T
//         let output = input.matmul(&weight_t);
//
//         // Ajouter le biais si présent
//         if let Some(ref bias) = self.bias {
//             output.add(bias)
//         } else {
//             output
//         }
//     }
//
//     fn parameters(&self) -> Vec<Variable> {
//         let mut params = vec![self.weight.clone()];
//         if let Some(ref bias) = self.bias {
//             params.push(bias.clone());
//         }
//         params
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
// impl Initializable for Linear {
//     fn init_weights(&mut self, method: InitMethod) {
//         let (out_features, in_features) = (self.out_features, self.in_features);
//
//         // Créer un nouveau tenseur de poids selon la méthode d'initialisation
//         let weight_data: Vec<f64> = match method {
//             InitMethod::Uniform { scale } => (0..in_features * out_features)
//                 .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * scale)
//                 .collect(),
//             InitMethod::Normal { mean, std } => {
//                 // Simple implementation using Box-Muller transform
//                 let mut values = Vec::new();
//                 for _ in 0..in_features * out_features {
//                     if values.len() % 2 == 0 {
//                         // Generate two normal random numbers using Box-Muller
//                         let u1: f64 = rand::random();
//                         let u2: f64 = rand::random();
//                         let mag = std * (-2.0 * u1.ln()).sqrt();
//                         let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos() + mean;
//                         let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin() + mean;
//                         values.push(z0);
//                         if values.len() < in_features * out_features {
//                             values.push(z1);
//                         }
//                     }
//                 }
//                 values.truncate(in_features * out_features);
//                 values
//             }
//             InitMethod::Xavier => {
//                 let scale = (6.0 / (in_features + out_features) as f64).sqrt();
//
//                 (0..in_features * out_features)
//                     .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * scale)
//                     .collect()
//             }
//             InitMethod::Kaiming => {
//                 let scale = (2.0 / in_features as f64).sqrt();
//
//                 (0..in_features * out_features)
//                     .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * scale)
//                     .collect()
//             }
//             InitMethod::Constant { value } => {
//                 vec![value; in_features * out_features]
//             }
//         };
//
//         // Mettre à jour le tenseur de poids
//         let weight_tensor = Tensor::from_data(&weight_data, vec![out_features, in_features], None);
//
//         self.weight = Variable::from_tensor(weight_tensor, true);
//
//         // Initialiser le biais si présent
//         if let Some(ref mut bias) = self.bias {
//             match method {
//                 InitMethod::Constant { value } => {
//                     let bias_data = vec![value; out_features];
//                     let bias_tensor = Tensor::from_data(&bias_data, vec![out_features], None);
//
//                     *bias = Variable::from_tensor(bias_tensor, true);
//                 }
//                 _ => {
//                     // Pour les autres méthodes, initialiser le biais à zéro
//                     let bias_data = vec![0.0; out_features];
//                     let bias_tensor = Tensor::from_data(&bias_data, vec![out_features], None);
//
//                     *bias = Variable::from_tensor(bias_tensor, true);
//                 }
//             }
//         }
//     }
// }
//
// // Tests pour le module nn
// #[cfg(test)]
// mod tests {
//     use super::*;
//     // use rustytorch_autograd::no_grad;
//     // use crate::activations::{ReLU, Sequential, Sigmoid};
//
//     #[test]
//     #[ignore] // Broadcasting multidimensionnel non encore implémenté
//     fn test_linear_layer() {
//         // Créer une couche linéaire simple: 2 entrées, 3 sorties
//         let linear = Linear::new(2, 3, true);
//
//         // Créer un tenseur d'entrée (batch de 4 exemples)
//         let input_tensor =
//             Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2], None);
//
//         let input = Variable::from_tensor(input_tensor, true);
//
//         // Forward pass
//         let output = linear.forward(&input);
//
//         // Vérifier la forme de sortie
//         assert_eq!(output.tensor.shape(), &[4, 3]);
//
//         // Vérifier que les paramètres sont récupérables
//         let params = linear.parameters();
//         assert_eq!(params.len(), 2); // Poids + biais
//     }
//
//     // #[test]
//     // fn test_sequential() {
//     //     // Créer un petit réseau séquentiel
//     //     let mut sequential = Sequential::new();
//     //
//     //     // Ajouter des couches
//     //     sequential.add(Box::new(Linear::new(2, 4, true)));
//     //     sequential.add(Box::new(ReLU::new()));
//     //     sequential.add(Box::new(Linear::new(4, 1, true)));
//     //     sequential.add(Box::new(Sigmoid::new()));
//     //
//     //     // Créer un tenseur d'entrée
//     //     let input_tensor = Tensor::from_data(&[1.0, 2.0], vec![1, 2], None);
//     //
//     //     let input = Variable::from_tensor(input_tensor, true);
//     //
//     //     // Forward pass
//     //     let output = sequential.forward(&input);
//     //
//     //     // Vérifier la forme de sortie
//     //     assert_eq!(output.tensor.shape(), &[1, 1]);
//     //
//     //     // Vérifier que la sortie est entre 0 et 1 (sigmoid)
//     //     let value = output.tensor.storage().as_ref().to_vec_f64()[0];
//     //     assert!(value >= 0.0 && value <= 1.0);
//     //
//     //     // Vérifier le nombre de paramètres
//     //     let params = sequential.parameters();
//     //     assert_eq!(params.len(), 4); // 2 couches linéaires × (poids + biais)
//     // }
//
//     #[test]
//     fn test_initialization() {
//         // Tester différentes méthodes d'initialisation
//         let mut linear = Linear::new(10, 5, true);
//
//         // Initialisation constante
//         linear.init_weights(InitMethod::Constant { value: 0.5 });
//
//         // Vérifier que tous les poids sont à 0.5
//         // let weight_data = linear.weight.tensor.storage().as_ref().to_vec_f64();
//         let weight_data = linear.weight.tensor.storage().to_vec_f64();
//         for &w in &weight_data {
//             assert!((w - 0.5).abs() < 1e-6);
//         }
//
//         // Initialisation normale
//         linear.init_weights(InitMethod::Normal {
//             mean: 0.0,
//             std: 0.01,
//         });
//
//         // Pour une initialisation stochastique, on vérifie juste que les poids ont changé
//         // let weight_data_new = linear.weight.tensor.storage().as_ref().to_vec_f64();
//         let weight_data_new = linear.weight.tensor.storage().to_vec_f64();
//         assert!(weight_data != weight_data_new);
//     }
// }
