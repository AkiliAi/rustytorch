//! Démonstration des fonctionnalités de base de l'autograd

use rustytorch_autograd::Variable;
use rustytorch_tensor::Tensor;

pub fn run_autograd_basic_demo() {
    println!("=== Démonstration: Autograd de Base ===\n");

    // === Exemple 1: Gradient simple ===
    println!("1. Gradient simple: f(x) = x²");
    let x = Variable::variable_with_grad(&[3.0], vec![1]);
    let y = x.mul(&x); // y = x²
    
    println!("   x = 3.0");
    println!("   y = x² = {:.2}", y.tensor().storage().to_vec_f64()[0]);
    
    // Calculer le gradient
    let grads = Variable::compute_grad(&[y], &[x.clone()], None, false, false).unwrap();
    if let Some(grad) = &grads[0] {
        let grad_val = grad.tensor().storage().to_vec_f64()[0];
        println!("   dy/dx = 2x = {:.2} (attendu: 6.0)\n", grad_val);
    }

    // === Exemple 2: Gradients multiples ===
    println!("2. Gradients multiples: f(x,y) = x*y + x²");
    let x = Variable::variable_with_grad(&[2.0], vec![1]);
    let y = Variable::variable_with_grad(&[3.0], vec![1]);
    
    let xy = x.mul(&y);           // x*y
    let x_squared = x.mul(&x);    // x²
    let f = xy.add(&x_squared);   // f = x*y + x²
    
    println!("   x = 2.0, y = 3.0");
    println!("   f = x*y + x² = {:.2}", f.tensor().storage().to_vec_f64()[0]);
    
    let grads = Variable::compute_grad(&[f], &[x.clone(), y.clone()], None, false, false).unwrap();
    
    if let Some(dx_grad) = &grads[0] {
        let dx_val = dx_grad.tensor().storage().to_vec_f64()[0];
        println!("   ∂f/∂x = y + 2x = {:.2} (attendu: 7.0)", dx_val);
    }
    
    if let Some(dy_grad) = &grads[1] {
        let dy_val = dy_grad.tensor().storage().to_vec_f64()[0];
        println!("   ∂f/∂y = x = {:.2} (attendu: 2.0)\n", dy_val);
    }

    // === Exemple 3: Fonctions transcendantes ===
    println!("3. Fonctions transcendantes: f(x) = sin(x) + exp(x/2)");
    let x = Variable::variable_with_grad(&[1.0], vec![1]);
    let x_half = x.mul(&Variable::from_tensor(Tensor::from_data(&[0.5], vec![1], None), false));
    let sin_x = x.sin();
    let exp_x_half = x_half.exp();
    let f = sin_x.add(&exp_x_half);
    
    println!("   x = 1.0");
    println!("   f = sin(x) + exp(x/2) = {:.4}", f.tensor().storage().to_vec_f64()[0]);
    
    let grads = Variable::compute_grad(&[f], &[x.clone()], None, false, false).unwrap();
    if let Some(grad) = &grads[0] {
        let grad_val = grad.tensor().storage().to_vec_f64()[0];
        println!("   df/dx = cos(x) + 0.5*exp(x/2) = {:.4}\n", grad_val);
    }

    // === Exemple 4: Graphe de calcul complexe ===
    println!("4. Graphe complexe: f(x,y,z) = (x + y) * z² + log(x*y)");
    let x = Variable::variable_with_grad(&[2.0], vec![1]);
    let y = Variable::variable_with_grad(&[3.0], vec![1]);
    let z = Variable::variable_with_grad(&[1.5], vec![1]);
    
    let x_plus_y = x.add(&y);           // x + y
    let z_squared = z.mul(&z);          // z²
    let first_term = x_plus_y.mul(&z_squared); // (x + y) * z²
    
    let xy = x.mul(&y);                 // x * y
    let log_xy = xy.log();              // log(x*y)
    
    let f = first_term.add(&log_xy);    // f = (x + y) * z² + log(x*y)
    
    println!("   x = 2.0, y = 3.0, z = 1.5");
    println!("   f = (x + y) * z² + log(x*y) = {:.4}", f.tensor().storage().to_vec_f64()[0]);
    
    let grads = Variable::compute_grad(&[f], &[x.clone(), y.clone(), z.clone()], None, false, false).unwrap();
    
    if let Some(dx_grad) = &grads[0] {
        println!("   ∂f/∂x = z² + 1/(x*y) * y = {:.4}", dx_grad.tensor().storage().to_vec_f64()[0]);
    }
    if let Some(dy_grad) = &grads[1] {
        println!("   ∂f/∂y = z² + 1/(x*y) * x = {:.4}", dy_grad.tensor().storage().to_vec_f64()[0]);
    }
    if let Some(dz_grad) = &grads[2] {
        println!("   ∂f/∂z = 2z * (x + y) = {:.4}", dz_grad.tensor().storage().to_vec_f64()[0]);
    }

    println!("\n=== Fin de la démonstration Autograd de Base ===\n");
}