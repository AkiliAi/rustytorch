// // examples/linear_network.rs
//
// use rustytorch_tensor::Tensor;
// use rustytorch_autograd::{Variable, no_grad};
// use rustytorch_nn::{Linear, Module, MSELoss};
// use rustytorch_optim::SGD;
//
// // Un simple modèle linéaire
// struct LinearModel {
//     linear: Linear,
// }
//
// impl LinearModel {
//     fn new(in_features: usize, out_features: usize) -> Self {
//         Self {
//             linear: Linear::new(in_features, out_features, true),
//         }
//     }
// }
//
// impl Module for LinearModel {
//     fn forward(&self, x: &Variable) -> Variable {
//         self.linear.forward(x)
//     }
//
//     fn parameters(&self) -> Vec<Variable> {
//         self.linear.parameters()
//     }
// }
//
// fn main() {
//     println!("RustyTorch - Exemple d'un réseau linéaire");
//
//     // Création des données
//     let x_data = Tensor::rand(vec![100, 5], None); // 100 échantillons, 5 caractéristiques
//     let w_true = Tensor::rand(vec![5, 1], None);   // Poids vrai
//     let b_true = Tensor::rand(vec![1], None);      // Biais vrai
//
//     // Générer les étiquettes : y = X·w + b + bruit
//     let noise = Tensor::rand(vec![100, 1], None).mul(0.1); // Bruit aléatoire
//     let y_data = x_data.matmul(&w_true).add(b_true).add(noise);
//
//     // Créer le modèle, la fonction de perte et l'optimiseur
//     let model = LinearModel::new(5, 1);
//     let criterion = MSELoss::new();
//     let mut optimizer = SGD::new(model.parameters(), 0.01);
//
//     println!("Début de l'entraînement...");
//
//     // Boucle d'entraînement
//     for epoch in 0..100 {
//         // Convertir les tenseurs en variables pour l'autograd
//         let x = Variable::from_tensor(x_data.clone(), false);
//         let y = Variable::from_tensor(y_data.clone(), false);
//
//         // Forward pass
//         let y_pred = model.forward(&x);
//         let loss = criterion.forward(&y_pred, &y);
//
//         // Backward pass et optimisation
//         optimizer.zero_grad();
//         loss.backward();
//         optimizer.step();
//
//         if epoch % 10 == 0 {
//             // Désactiver le calcul de gradient pour l'évaluation
//             let _guard = no_grad();
//             println!("Époque {} - Perte: {:.6}", epoch, loss.tensor.to_f64());
//         }
//     }
//
//     println!("Entraînement terminé!");
//
//     // Évaluer le modèle final
//     {
//         let _guard = no_grad();
//         let x = Variable::from_tensor(x_data.clone(), false);
//         let y_pred = model.forward(&x);
//
//         println!("Poids appris:");
//         println!("w: {:?}", model.linear.weight.tensor);
//         println!("b: {:?}", model.linear.bias.as_ref().unwrap().tensor);
//
//         println!("Poids vrais:");
//         println!("w: {:?}", w_true);
//         println!("b: {:?}", b_true);
//     }
// }