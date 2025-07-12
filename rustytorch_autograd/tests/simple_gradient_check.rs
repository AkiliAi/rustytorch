//! Test simple pour déboguer les gradients
//! 
//! Test minimal pour comprendre le problème avec notre validation

use rustytorch_autograd::{Variable, enable_grad};
use rustytorch_tensor::Tensor;

#[test]
fn test_simple_addition() {
    let _guard = enable_grad();
    
    // Test très simple: f(x) = x + 2
    let x = Variable::variable_with_grad(&[3.0], vec![1]);
    let constant = Variable::variable_with_grad(&[2.0], vec![1]);
    
    let result = x.add(&constant);
    
    // Calculer le gradient analytique
    let analytical_grads = Variable::compute_grad(&[result.clone()], &[x.clone()], None, false, false).unwrap();
    
    println!("🧪 Test simple: f(x) = x + 2");
    println!("   x = 3.0, constant = 2.0");
    println!("   f(3.0) = {:.6}", result.tensor().storage().to_vec_f64()[0]);
    
    if let Some(analytical_grad) = &analytical_grads[0] {
        let analytical_value = analytical_grad.tensor().storage().to_vec_f64()[0];
        println!("   df/dx (analytical) = {:.6}", analytical_value);
        
        // Calculer le gradient numérique manuellement
        let eps = 1e-4;
        let x_plus = Variable::variable_with_grad(&[3.0 + eps], vec![1]);
        let x_minus = Variable::variable_with_grad(&[3.0 - eps], vec![1]);
        
        let f_plus = x_plus.add(&constant).tensor().storage().to_vec_f64()[0];
        let f_minus = x_minus.add(&constant).tensor().storage().to_vec_f64()[0];
        
        let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
        
        println!("   f(x+h) = {:.6}, f(x-h) = {:.6}", f_plus, f_minus);
        println!("   df/dx (numerical) = {:.6}", numerical_grad);
        println!("   Error = {:.2e}", (analytical_value - numerical_grad).abs());
        
        // Pour l'addition, le gradient devrait être 1
        assert!((analytical_value - 1.0).abs() < 1e-6, "Analytical gradient should be 1.0");
        assert!((numerical_grad - 1.0).abs() < 1e-2, "Numerical gradient should be close to 1.0");
    } else {
        panic!("No analytical gradient computed");
    }
}

#[test]
fn test_simple_multiplication() {
    let _guard = enable_grad();
    
    // Test simple: f(x) = x * 5
    let x = Variable::variable_with_grad(&[2.0], vec![1]);
    let constant = Variable::variable_with_grad(&[5.0], vec![1]);
    
    let result = x.mul(&constant);
    
    // Calculer le gradient analytique
    let analytical_grads = Variable::compute_grad(&[result.clone()], &[x.clone()], None, false, false).unwrap();
    
    println!("🧪 Test simple: f(x) = x * 5");
    println!("   x = 2.0, constant = 5.0");
    println!("   f(2.0) = {:.6}", result.tensor().storage().to_vec_f64()[0]);
    
    if let Some(analytical_grad) = &analytical_grads[0] {
        let analytical_value = analytical_grad.tensor().storage().to_vec_f64()[0];
        println!("   df/dx (analytical) = {:.6}", analytical_value);
        
        // Calculer le gradient numérique manuellement
        let eps = 1e-4;
        let x_plus = Variable::variable_with_grad(&[2.0 + eps], vec![1]);
        let x_minus = Variable::variable_with_grad(&[2.0 - eps], vec![1]);
        
        let f_plus = x_plus.mul(&constant).tensor().storage().to_vec_f64()[0];
        let f_minus = x_minus.mul(&constant).tensor().storage().to_vec_f64()[0];
        
        let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
        
        println!("   f(x+h) = {:.6}, f(x-h) = {:.6}", f_plus, f_minus);
        println!("   df/dx (numerical) = {:.6}", numerical_grad);
        println!("   Error = {:.2e}", (analytical_value - numerical_grad).abs());
        
        // Pour la multiplication par 5, le gradient devrait être 5
        assert!((analytical_value - 5.0).abs() < 1e-6, "Analytical gradient should be 5.0");
        assert!((numerical_grad - 5.0).abs() < 1e-2, "Numerical gradient should be close to 5.0");
    } else {
        panic!("No analytical gradient computed");
    }
}

#[test]
fn test_vector_sum() {
    let _guard = enable_grad();
    
    // Test avec vecteur: f(x) = sum(x) où x = [1, 2, 3]
    let x = Variable::variable_with_grad(&[1.0, 2.0, 3.0], vec![3]);
    let result = x.sum();
    
    // Calculer le gradient analytique
    let analytical_grads = Variable::compute_grad(&[result.clone()], &[x.clone()], None, false, false).unwrap();
    
    println!("🧪 Test vecteur: f(x) = sum(x)");
    println!("   x = [1.0, 2.0, 3.0]");
    println!("   f(x) = {:.6}", result.tensor().storage().to_vec_f64()[0]);
    
    if let Some(analytical_grad) = &analytical_grads[0] {
        let analytical_values = analytical_grad.tensor().storage().to_vec_f64();
        println!("   df/dx (analytical) = {:?}", analytical_values);
        
        // Pour sum(), le gradient par rapport à chaque élément devrait être 1
        for (i, &grad_val) in analytical_values.iter().enumerate() {
            println!("   df/dx[{}] = {:.6}", i, grad_val);
            assert!((grad_val - 1.0).abs() < 1e-6, "Gradient for sum should be 1.0 for each element");
        }
    } else {
        panic!("No analytical gradient computed");
    }
}