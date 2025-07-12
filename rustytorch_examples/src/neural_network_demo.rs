//! Démonstration d'un mini réseau de neurones avec autograd

use rustytorch_autograd::Variable;
use rustytorch_tensor::Tensor;

pub fn run_neural_network_demo() {
    println!("=== Démonstration: Mini Réseau de Neurones ===\n");

    // === Configuration du réseau ===
    println!("1. Configuration du réseau de neurones");
    println!("   Architecture: 2 → 3 → 1 (perceptron multicouche)");
    println!("   Fonction d'activation: ReLU (couche cachée), Sigmoid (sortie)\n");

    // === Initialisation des poids ===
    // Couche 1: 2 → 3 (W1: 3x2, b1: 3x1)
    let w1_data = vec![0.5, -0.3, 0.2, 0.8, -0.1, 0.4]; // 3x2
    let b1_data = vec![0.1, -0.2, 0.3];                   // 3x1
    
    // Couche 2: 3 → 1 (W2: 1x3, b2: 1x1)  
    let w2_data = vec![0.6, -0.4, 0.7];                   // 1x3
    let b2_data = vec![0.05];                             // 1x1
    
    let mut w1 = Variable::from_tensor(Tensor::from_data(&w1_data, vec![3, 2], None), true);
    let mut b1 = Variable::from_tensor(Tensor::from_data(&b1_data, vec![3], None), true);
    let mut w2 = Variable::from_tensor(Tensor::from_data(&w2_data, vec![1, 3], None), true);
    let mut b2 = Variable::from_tensor(Tensor::from_data(&b2_data, vec![1], None), true);

    println!("2. Poids initialisés:");
    println!("   W1 (3x2): {:?}", w1.tensor().storage().to_vec_f64());
    println!("   b1 (3x1): {:?}", b1.tensor().storage().to_vec_f64());
    println!("   W2 (1x3): {:?}", w2.tensor().storage().to_vec_f64());
    println!("   b2 (1x1): {:?}\n", b2.tensor().storage().to_vec_f64());

    // === Données d'entraînement ===
    let train_data = vec![
        (vec![1.0, 0.0], 1.0),  // XOR-like data
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 1.0], 0.0),
        (vec![0.0, 0.0], 0.0),
    ];

    println!("3. Données d'entraînement (XOR-like):");
    for (i, (input, target)) in train_data.iter().enumerate() {
        println!("   Exemple {}: {:?} → {:.1}", i+1, input, target);
    }
    println!();

    // === Boucle d'entraînement ===
    let learning_rate = 0.1;
    let epochs = 5;

    println!("4. Entraînement (learning_rate = {}, epochs = {}):\n", learning_rate, epochs);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        
        println!("   Époque {}:", epoch + 1);
        
        for (example_idx, (input_data, target)) in train_data.iter().enumerate() {
            // === Forward Pass ===
            
            // Input
            let x = Variable::from_tensor(Tensor::from_data(input_data, vec![2], None), false);
            
            // Couche 1: z1 = W1 @ x + b1
            let z1_linear = simulate_linear(&w1, &x, &b1);
            let z1 = relu_activation(&z1_linear);
            
            // Couche 2: z2 = W2 @ z1 + b2  
            let z2_linear = simulate_linear_1d(&w2, &z1, &b2);
            let output = sigmoid_activation(&z2_linear);
            
            // Loss: MSE = (output - target)²
            let target_var = Variable::from_tensor(Tensor::from_data(&[*target], vec![1], None), false);
            let diff = output.sub(&target_var);
            let loss = diff.mul(&diff);
            
            let loss_val = loss.tensor().storage().to_vec_f64()[0];
            total_loss += loss_val;
            
            // === Backward Pass ===
            let grads = Variable::compute_grad(
                &[loss], 
                &[w1.clone(), b1.clone(), w2.clone(), b2.clone()], 
                None, false, false
            ).unwrap();
            
            // === Mise à jour des poids ===
            if let (Some(dw1), Some(db1), Some(dw2), Some(db2)) = 
                (&grads[0], &grads[1], &grads[2], &grads[3]) {
                
                // w1 = w1 - lr * dw1
                let w1_update = element_wise_update(&w1, dw1, learning_rate);
                let b1_update = element_wise_update(&b1, db1, learning_rate);
                let w2_update = element_wise_update(&w2, dw2, learning_rate);
                let b2_update = element_wise_update(&b2, db2, learning_rate);
                
                w1 = w1_update;
                b1 = b1_update;
                w2 = w2_update;
                b2 = b2_update;
            }
            
            println!("     Ex {}: Input={:?}, Target={:.1}, Output={:.3}, Loss={:.4}", 
                     example_idx + 1, input_data, target, 
                     output.tensor().storage().to_vec_f64()[0], loss_val);
        }
        
        println!("     Loss moyenne: {:.4}\n", total_loss / train_data.len() as f64);
    }

    // === Test final ===
    println!("5. Test après entraînement:");
    for (input_data, target) in &train_data {
        let x = Variable::from_tensor(Tensor::from_data(input_data, vec![2], None), false);
        let z1 = relu_activation(&simulate_linear(&w1, &x, &b1));
        let output = sigmoid_activation(&simulate_linear_1d(&w2, &z1, &b2));
        let output_val = output.tensor().storage().to_vec_f64()[0];
        
        println!("   {:?} → {:.3} (target: {:.1})", input_data, output_val, target);
    }

    println!("\n=== Fin de la démonstration Réseau de Neurones ===\n");
}

// Fonctions utilitaires pour le réseau de neurones

fn simulate_linear(weight: &Variable, input: &Variable, bias: &Variable) -> Variable {
    // Simulation de W @ x + b pour un cas simplifié
    // Note: Dans une vraie implémentation, on utiliserait matmul
    let w_data = weight.tensor().storage().to_vec_f64();
    let x_data = input.tensor().storage().to_vec_f64();
    let b_data = bias.tensor().storage().to_vec_f64();
    
    let mut result = Vec::new();
    for i in 0..3 {  // 3 neurones dans la couche cachée
        let mut sum = b_data[i];
        for j in 0..2 {  // 2 inputs
            sum += w_data[i * 2 + j] * x_data[j];
        }
        result.push(sum);
    }
    
    Variable::from_tensor(Tensor::from_data(&result, vec![3], None), true)
}

fn simulate_linear_1d(weight: &Variable, input: &Variable, bias: &Variable) -> Variable {
    // W @ x + b pour la couche de sortie (1 neurone)
    let w_data = weight.tensor().storage().to_vec_f64();
    let x_data = input.tensor().storage().to_vec_f64();
    let b_data = bias.tensor().storage().to_vec_f64();
    
    let mut sum = b_data[0];
    for i in 0..3 {
        sum += w_data[i] * x_data[i];
    }
    
    Variable::from_tensor(Tensor::from_data(&[sum], vec![1], None), true)
}

fn relu_activation(x: &Variable) -> Variable {
    x.relu()
}

fn sigmoid_activation(x: &Variable) -> Variable {
    x.sigmoid()
}

fn element_wise_update(param: &Variable, grad: &Variable, lr: f64) -> Variable {
    // param = param - lr * grad (élément par élément)
    let param_data = param.tensor().storage().to_vec_f64();
    let grad_data = grad.tensor().storage().to_vec_f64();
    
    let updated: Vec<f64> = param_data.iter().zip(grad_data.iter())
        .map(|(p, g)| p - lr * g)
        .collect();
    
    Variable::from_tensor(
        Tensor::from_data(&updated, param.shape(), None), 
        true
    )
}