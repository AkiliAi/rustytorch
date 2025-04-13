// examples/nn_example.rs

use rustytorch_tensor::Tensor;
use rustytorch_autograd::Variable;
use rustytorch_autograd::no_grad;
use rustytorch_nn::{Module, Linear, ReLU, Sigmoid, Sequential, InitMethod, Initializable};
use std::error::Error;

// Structure pour un optimiseur SGD simple
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

    fn zero_grad(&mut self) {
        // Cette fonction devrait réinitialiser les gradients
        // Dans une implémentation complète, chaque Variable aurait une méthode pour cela
    }

    fn step(&mut self) {
        for param in &self.parameters {
            if let Some(ref grad) = param.grad {
                // Dans une implémentation complète, on mettrait à jour les paramètres ici
                // param.tensor = param.tensor - learning_rate * grad
            }
        }
    }
}

// Fonction de perte MSE
fn mse_loss(predictions: &Variable, targets: &Variable) -> Variable {
    // Calculer (predictions - targets)^2
    let diff = predictions.sub(targets);
    let squared = diff.mul(&diff);

    // Moyenne pour obtenir MSE
    squared.mean()
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("RustyTorch - Exemple d'utilisation du module nn\n");

    // Créer un jeu de données simple pour XOR
    let x_data = vec![
        0.0, 0.0,  // Entrée [0, 0] -> Sortie 0
        0.0, 1.0,  // Entrée [0, 1] -> Sortie 1
        1.0, 0.0,  // Entrée [1, 0] -> Sortie 1
        1.0, 1.0   // Entrée [1, 1] -> Sortie 0
    ];

    let y_data = vec![
        0.0,  // XOR: 0 ^ 0 = 0
        1.0,  // XOR: 0 ^ 1 = 1
        1.0,  // XOR: 1 ^ 0 = 1
        0.0   // XOR: 1 ^ 1 = 0
    ];

    // Créer les tenseurs pour les entrées et les sorties
    let x_tensor = Tensor::from_data(&x_data, vec![4, 2], None)?;
    let y_tensor = Tensor::from_data(&y_data, vec![4, 1], None)?;

    // Convertir en Variables pour l'autograd
    let x = Variable::from_tensor(x_tensor, false);
    let y = Variable::from_tensor(y_tensor, false);

    // Créer un réseau neuronal simple pour XOR
    // Architecture: 2 -> 4 -> 1 (avec ReLU en couche cachée et Sigmoid en sortie)
    let mut model = Sequential::new();

    // Couche d'entrée vers couche cachée
    let mut hidden_layer = Linear::new(2, 4, true);
    // Initialiser avec Kaiming pour ReLU
    hidden_layer.init_weights(InitMethod::Kaiming);
    model.add(Box::new(hidden_layer));
    model.add(Box::new(ReLU::new()));

    // Couche cachée vers sortie
    let mut output_layer = Linear::new(4, 1, true);
    // Initialiser avec Xavier
    output_layer.init_weights(InitMethod::Xavier);
    model.add(Box::new(output_layer));
    model.add(Box::new(Sigmoid::new()));

    // Configurer l'optimiseur
    let optimizer = SGD::new(model.parameters(), 0.1);

    println!("Début de l'entraînement du modèle XOR...");

    // Entraînement du modèle
    let epochs = 1000;
    for epoch in 0..epochs {
        // Forward pass
        let predictions = model.forward(&x);

        // Calculer la perte
        let loss = mse_loss(&predictions, &y);

        if epoch % 100 == 0 {
            // Extraire la valeur scalaire de la perte
            let loss_value = loss.tensor.storage().as_ref().to_vec_f64()[0];
            println!("Époque {}: Perte = {:.6}", epoch, loss_value);
        }

        // Backward pass et mise à jour des poids
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    println!("\nÉvaluation du modèle:");

    // Évaluer le modèle (sans calcul de gradient)
    {
        let _no_grad_guard = no_grad();

        // Créer des entrées pour tester individuellement
        let inputs = [
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0]
        ];

        for input in &inputs {
            // Créer un tenseur pour cette entrée
            let tensor = Tensor::from_data(input, vec![1, 2], None)?;
            let var = Variable::from_tensor(tensor, false);

            // Faire une prédiction
            let prediction = model.forward(&var);

            // Extraire la valeur de prédiction
            let value = prediction.tensor.storage().as_ref().to_vec_f64()[0];

            println!("Entrée: [{}, {}] -> Prédiction: {:.4} (Attendu: {})",
                     input[0], input[1], value,
                     if (input[0] == 0.0 && input[1] == 0.0) || (input[0] == 1.0 && input[1] == 1.0) {
                         0.0
                     } else {
                         1.0
                     });
        }
    }

    println!("\nExemple terminé!");

    Ok(())
}