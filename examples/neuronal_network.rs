// rustytorch_examples/src/bin/neural_network.rs

use rustytorch_tensor::Tensor;
use rustytorch_autograd::{Variable, no_grad, Operation};
use std::error::Error;

// Structure pour une couche linéaire (Dense/Fully Connected)
struct Linear {
    weight: Variable,
    bias: Variable,
}

impl Linear {
    // Initialisation de la couche avec des poids aléatoires
    fn new(in_features: usize, out_features: usize) -> Self {
        // Initialisation Xavier/Glorot pour les poids
        let weight_scale = (6.0 / (in_features + out_features) as f64).sqrt();

        let weight_data: Vec<f64> = (0..in_features * out_features)
            .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * weight_scale)
            .collect();

        let bias_data: Vec<f64> = (0..out_features)
            .map(|_| 0.0) // Initialiser les biais à zéro
            .collect();

        let weight_tensor = Tensor::from_data(&weight_data, vec![out_features, in_features], None);
        let bias_tensor = Tensor::from_data(&bias_data, vec![out_features], None);

        Self {
            weight: Variable::from_tensor(weight_tensor, true),
            bias: Variable::from_tensor(bias_tensor, true),
        }
    }

    // Forward pass de la couche linéaire: y = x @ W^T + b
    fn forward(&self, x: &Variable) -> Variable {
        // Transposer les poids pour multiplication matricielle
        let weight_t = Variable::from_tensor(
            self.weight.tensor.transpose(0, 1),
            self.weight.requires_grad
        );

        // Multiplication matricielle: x @ W^T
        let output = x.matmul(&weight_t);

        // Ajouter le biais (avec broadcasting automatique)
        output.add(&self.bias)
    }

    // Récupérer les paramètres de la couche
    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

// Structure pour un réseau neuronal simple à deux couches
struct SimpleNetwork {
    layer1: Linear,
    layer2: Linear,
}

impl SimpleNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            layer1: Linear::new(input_size, hidden_size),
            layer2: Linear::new(hidden_size, output_size),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        // Première couche avec activation ReLU
        let h1 = self.layer1.forward(x);
        let h1_relu = h1.relu();

        // Couche de sortie
        self.layer2.forward(&h1_relu)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.layer1.parameters();
        params.extend(self.layer2.parameters());
        params
    }
}

// Fonction d'activation ReLU pour Variable
impl Variable {
    fn relu(&self) -> Self {
        let result_tensor = match self.tensor.relu() {
            Ok(t) => t,
            Err(e) => panic!("Error in ReLU: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner simplement le résultat
        if !self.requires_grad {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour ReLU
        let self_clone = self.clone();
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            let grad_input = match self_clone.tensor.relu_backward(grad_output) {
                Ok(g) => g,
                Err(e) => panic!("Error in ReLU backward: {}", e),
            };
            vec![grad_input]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;

        Self::from_operation(
            result_tensor,
            Operation::Relu,
            vec![self.clone()],
            Some(grad_fn),
        )
    }
}

// Fonction de perte MSE (Mean Squared Error)
fn mse_loss(predictions: &Variable, targets: &Variable) -> Variable {
    // Calculer (predictions - targets)^2
    let diff = predictions.sub(targets);
    let squared = diff.mul(&diff);

    // Calculer la moyenne
    squared.mean()
}

// Optimiseur SGD (Stochastic Gradient Descent)
struct SGD {
    parameters: Vec<Variable>,
    learning_rate: f64,
}

impl SGD {
    fn new(parameters: Vec<Variable>, learning_rate: f64) -> Self {
        Self {
            parameters,
            learning_rate,
        }
    }

    fn zero_grad(&self) {
        // Réinitialiser les gradients à zéro
        for param in &self.parameters {
            // Nous n'avons pas d'accès direct pour modifier les gradients
            // Dans une implémentation complète, ce serait quelque chose comme:
            // param.grad = None; ou param.grad.fill_(0);
        }
    }

    fn step(&self) {
        // Mise à jour des paramètres avec la descente de gradient
        for param in &self.parameters {
            if let Some(grad) = &param.grad {
                // Calculer: param = param - learning_rate * grad
                // Dans cette version simplifiée, nous ne pouvons pas modifier le tenseur directement
                // Une approche complète utiliserait des opérations in-place ou des références mutables
            }
        }
    }
}

// Fonction principale
fn main() -> Result<(), Box<dyn Error>> {
    println!("RustyTorch - Exemple de réseau neuronal simple\n");

    // Créer un jeu de données synthétique simple (XOR)
    let x_data = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    ];
    let y_data = vec![
        0.0,
        1.0,
        1.0,
        0.0
    ];

    let x_tensor = Tensor::from_data(&x_data, vec![4, 2], None);
    let y_tensor = Tensor::from_data(&y_data, vec![4, 1], None);

    let x = Variable::from_tensor(x_tensor, false);
    let y = Variable::from_tensor(y_tensor, false);

    // Créer le réseau neuronal
    let model = SimpleNetwork::new(2, 8, 1);

    // Optimiseur
    let optimizer = SGD::new(model.parameters(), 0.1);

    println!("Démarrage de l'entraînement...");

    // Entraînement
    let epochs = 1000;
    for epoch in 0..epochs {
        // Forward pass
        let predictions = model.forward(&x);

        // Calculer la perte
        let loss = mse_loss(&predictions, &y);

        // Afficher la progression
        if epoch % 100 == 0 {
            println!("Époque {}: Perte = {}", epoch, extract_scalar(&loss.tensor));
        }

        // Backward pass
        optimizer.zero_grad();
        loss.backward();

        // Mise à jour des poids
        optimizer.step();
    }

    // Évaluation du modèle
    println!("\nÉvaluation du modèle:");
    {
        let _guard = no_grad(); // Désactiver le calcul de gradient pour l'évaluation

        let predictions = model.forward(&x);

        println!("Entrées XOR:");
        println!("  [0, 0] -> {:.4}", get_prediction(&model, &[0.0, 0.0]));
        println!("  [0, 1] -> {:.4}", get_prediction(&model, &[0.0, 1.0]));
        println!("  [1, 0] -> {:.4}", get_prediction(&model, &[1.0, 0.0]));
        println!("  [1, 1] -> {:.4}", get_prediction(&model, &[1.0, 1.0]));
    }

    println!("\nExemple terminé!");

    Ok(())
}

// Fonction utilitaire pour obtenir une prédiction sur un exemple unique
fn get_prediction(model: &SimpleNetwork, input: &[f64]) -> f64 {
    let input_tensor = Tensor::from_data(input, vec![1, 2], None);
    let input_var = Variable::from_tensor(input_tensor, false);
    let prediction = model.forward(&input_var);
    extract_scalar(&prediction.tensor)
}

// Fonction utilitaire pour extraire un scalaire d'un tenseur
fn extract_scalar(tensor: &Tensor) -> f64 {
    let storage = tensor.storage();
    match storage {
        rustytorch_tensor::storage::StorageType::F32(data) => {
            if data.len() >= 1 {
                data[0] as f64
            } else {
                f64::NAN
            }
        },
        rustytorch_tensor::storage::StorageType::F64(data) => {
            if data.len() >= 1 {
                data[0]
            } else {
                f64::NAN
            }
        },
        _ => f64::NAN,
    }
}

// Extension de Variable avec la méthode mean() pour l'exemple
impl Variable {
    fn mean(&self) -> Self {
        let result_tensor = match self.tensor.mean() {
            Ok(t) => t,
            Err(e) => panic!("Error in mean: {}", e),
        };

        if !self.requires_grad {
            return Self::from_tensor(result_tensor, false);
        }

        let self_clone = self.clone();
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            // Le gradient pour mean est grad_output / n pour chaque élément
            let n = self_clone.tensor.numel() as f64;
            let n_tensor = Tensor::from_data(&[1.0 / n], vec![1], None);

            let grad_input = match grad_output.mul(&n_tensor) {
                Ok(g) => {
                    // Broadcast le gradient à la forme originale
                    match g.broadcast_to(&self_clone.tensor.shape()) {
                        Ok(broadcasted) => broadcasted,
                        Err(e) => panic!("Error broadcasting gradient: {}", e),
                    }
                },
                Err(e) => panic!("Error scaling gradient: {}", e),
            };

            vec![grad_input]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;

        Self::from_operation(
            result_tensor,
            Operation::None, // Nous pourrions ajouter une opération Mean
            vec![self.clone()],
            Some(grad_fn),
        )
    }
}