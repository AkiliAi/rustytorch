//! Démonstration d'algorithmes d'optimisation avec autograd

use rustytorch_autograd::Variable;
use rustytorch_tensor::Tensor;

pub fn run_optimization_demo() {
    println!("=== Démonstration: Algorithmes d'Optimisation ===\n");

    // === Fonction objectif ===
    println!("Fonction à optimiser: f(x,y) = (x-2)² + (y+1)² + sin(x*y)");
    println!("Minimum théorique proche de (2, -1)\n");

    // === 1. Gradient Descent classique ===
    println!("1. Gradient Descent classique");
    gradient_descent_demo();
    
    // === 2. Momentum ===
    println!("\n2. Gradient Descent avec Momentum");
    momentum_demo();
    
    // === 3. Adam-like optimizer ===
    println!("\n3. Optimiseur adaptatif (Adam-like)");
    adaptive_demo();

    println!("\n=== Fin de la démonstration Optimisation ===\n");
}

fn gradient_descent_demo() {
    let mut x = Variable::variable_with_grad(&[0.0], vec![1]);
    let mut y = Variable::variable_with_grad(&[0.0], vec![1]);
    let lr = 0.1;
    
    println!("   Point de départ: (0.0, 0.0)");
    println!("   Learning rate: {}", lr);
    println!("   Itér.  |    x     |    y     |   f(x,y)  | ||grad||");
    println!("   -------|----------|----------|-----------|----------");
    
    for iter in 0..10 {
        // Calculer f(x,y)
        let f = objective_function(&x, &y);
        let f_val = f.tensor().storage().to_vec_f64()[0];
        
        // Calculer les gradients
        let grads = Variable::compute_grad(&[f], &[x.clone(), y.clone()], None, false, false).unwrap();
        
        if let (Some(dx), Some(dy)) = (&grads[0], &grads[1]) {
            let dx_val = dx.tensor().storage().to_vec_f64()[0];
            let dy_val = dy.tensor().storage().to_vec_f64()[0];
            let grad_norm = (dx_val * dx_val + dy_val * dy_val).sqrt();
            
            let x_val = x.tensor().storage().to_vec_f64()[0];
            let y_val = y.tensor().storage().to_vec_f64()[0];
            
            println!("   {:6} | {:8.3} | {:8.3} | {:9.3} | {:8.3}", 
                     iter, x_val, y_val, f_val, grad_norm);
            
            // Mise à jour
            let new_x = x_val - lr * dx_val;
            let new_y = y_val - lr * dy_val;
            
            x = Variable::variable_with_grad(&[new_x], vec![1]);
            y = Variable::variable_with_grad(&[new_y], vec![1]);
        }
    }
}

fn momentum_demo() {
    let mut x = Variable::variable_with_grad(&[0.0], vec![1]);
    let mut y = Variable::variable_with_grad(&[0.0], vec![1]);
    let lr = 0.1;
    let momentum = 0.9;
    let mut vx = 0.0;  // Vélocité pour x
    let mut vy = 0.0;  // Vélocité pour y
    
    println!("   Point de départ: (0.0, 0.0)");
    println!("   Learning rate: {}, Momentum: {}", lr, momentum);
    println!("   Itér.  |    x     |    y     |   f(x,y)  | ||grad||");
    println!("   -------|----------|----------|-----------|----------");
    
    for iter in 0..10 {
        let f = objective_function(&x, &y);
        let f_val = f.tensor().storage().to_vec_f64()[0];
        
        let grads = Variable::compute_grad(&[f], &[x.clone(), y.clone()], None, false, false).unwrap();
        
        if let (Some(dx), Some(dy)) = (&grads[0], &grads[1]) {
            let dx_val = dx.tensor().storage().to_vec_f64()[0];
            let dy_val = dy.tensor().storage().to_vec_f64()[0];
            let grad_norm = (dx_val * dx_val + dy_val * dy_val).sqrt();
            
            let x_val = x.tensor().storage().to_vec_f64()[0];
            let y_val = y.tensor().storage().to_vec_f64()[0];
            
            println!("   {:6} | {:8.3} | {:8.3} | {:9.3} | {:8.3}", 
                     iter, x_val, y_val, f_val, grad_norm);
            
            // Mise à jour avec momentum
            vx = momentum * vx + lr * dx_val;
            vy = momentum * vy + lr * dy_val;
            
            let new_x = x_val - vx;
            let new_y = y_val - vy;
            
            x = Variable::variable_with_grad(&[new_x], vec![1]);
            y = Variable::variable_with_grad(&[new_y], vec![1]);
        }
    }
}

fn adaptive_demo() {
    let mut x = Variable::variable_with_grad(&[0.0], vec![1]);
    let mut y = Variable::variable_with_grad(&[0.0], vec![1]);
    let lr = 0.3;
    let beta1 = 0.9;   // Momentum exponential decay
    let beta2 = 0.999; // RMSprop exponential decay
    let eps = 1e-8;
    
    let mut mx = 0.0;  // First moment estimate for x
    let mut my = 0.0;  // First moment estimate for y
    let mut vx = 0.0;  // Second moment estimate for x  
    let mut vy = 0.0;  // Second moment estimate for y
    
    println!("   Point de départ: (0.0, 0.0)");
    println!("   Learning rate: {}, β₁: {}, β₂: {}", lr, beta1, beta2);
    println!("   Itér.  |    x     |    y     |   f(x,y)  | ||grad||");
    println!("   -------|----------|----------|-----------|----------");
    
    for iter in 0..10 {
        let f = objective_function(&x, &y);
        let f_val = f.tensor().storage().to_vec_f64()[0];
        
        let grads = Variable::compute_grad(&[f], &[x.clone(), y.clone()], None, false, false).unwrap();
        
        if let (Some(dx), Some(dy)) = (&grads[0], &grads[1]) {
            let dx_val = dx.tensor().storage().to_vec_f64()[0];
            let dy_val = dy.tensor().storage().to_vec_f64()[0];
            let grad_norm = (dx_val * dx_val + dy_val * dy_val).sqrt();
            
            let x_val = x.tensor().storage().to_vec_f64()[0];
            let y_val = y.tensor().storage().to_vec_f64()[0];
            
            println!("   {:6} | {:8.3} | {:8.3} | {:9.3} | {:8.3}", 
                     iter, x_val, y_val, f_val, grad_norm);
            
            // Adam-like update
            mx = beta1 * mx + (1.0 - beta1) * dx_val;
            my = beta1 * my + (1.0 - beta1) * dy_val;
            
            vx = beta2 * vx + (1.0 - beta2) * dx_val * dx_val;
            vy = beta2 * vy + (1.0 - beta2) * dy_val * dy_val;
            
            // Bias correction
            let t = (iter + 1) as f64;
            let mx_hat = mx / (1.0 - beta1.powf(t));
            let my_hat = my / (1.0 - beta1.powf(t));
            let vx_hat = vx / (1.0 - beta2.powf(t));
            let vy_hat = vy / (1.0 - beta2.powf(t));
            
            // Parameter update
            let new_x = x_val - lr * mx_hat / (vx_hat.sqrt() + eps);
            let new_y = y_val - lr * my_hat / (vy_hat.sqrt() + eps);
            
            x = Variable::variable_with_grad(&[new_x], vec![1]);
            y = Variable::variable_with_grad(&[new_y], vec![1]);
        }
    }
}

fn objective_function(x: &Variable, y: &Variable) -> Variable {
    // f(x,y) = (x-2)² + (y+1)² + sin(x*y)
    
    // (x-2)²
    let two = Variable::from_tensor(Tensor::from_data(&[2.0], vec![1], None), false);
    let x_minus_2 = x.sub(&two);
    let term1 = x_minus_2.mul(&x_minus_2);
    
    // (y+1)²
    let one = Variable::from_tensor(Tensor::from_data(&[1.0], vec![1], None), false);
    let y_plus_1 = y.add(&one);
    let term2 = y_plus_1.mul(&y_plus_1);
    
    // sin(x*y)
    let xy = x.mul(y);
    let term3 = xy.sin();
    
    // Somme totale
    term1.add(&term2).add(&term3)
}