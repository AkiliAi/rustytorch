use rustytorch_autograd::{Variable, enable_grad};

fn main() {
    println!("Testing basic gradient functionality...");
    
    let _guard = enable_grad();
    
    // Test très simple: f(x) = x + 2
    let x = Variable::variable_with_grad(&[3.0], vec![1]);
    let constant = Variable::variable_with_grad(&[2.0], vec![1]);
    
    let result = x.add(&constant);
    
    println!("x = {:?}", x.tensor().storage().to_vec_f64());
    println!("constant = {:?}", constant.tensor().storage().to_vec_f64());
    println!("result = {:?}", result.tensor().storage().to_vec_f64());
    
    // Calculer le gradient analytique
    match Variable::compute_grad(&[result.clone()], &[x.clone()], None, false, false) {
        Ok(analytical_grads) => {
            if let Some(analytical_grad) = &analytical_grads[0] {
                let analytical_value = analytical_grad.tensor().storage().to_vec_f64()[0];
                println!("Gradient analytique: {:.6}", analytical_value);
                
                if (analytical_value - 1.0).abs() < 1e-6 {
                    println!("✅ Test gradient addition PASSED");
                } else {
                    println!("❌ Test gradient addition FAILED - expected 1.0, got {}", analytical_value);
                }
            } else {
                println!("❌ No analytical gradient computed");
            }
        },
        Err(e) => {
            println!("❌ Erreur calcul gradient: {}", e);
        }
    }
    
    // Test de Hessienne simple
    println!("\nTesting second-order gradients...");
    let x = Variable::variable_with_grad(&[2.0], vec![1]);
    let y = x.mul(&x).mul(&x); // x³
    
    println!("x = {:?}", x.tensor().storage().to_vec_f64());
    println!("y = x³ = {:?}", y.tensor().storage().to_vec_f64());
    
    match y.hessian(&[x.clone()]) {
        Ok(hessian) => {
            if !hessian.is_empty() && !hessian[0].is_empty() {
                if let Some(second_grad) = &hessian[0][0] {
                    let second_grad_value = second_grad.tensor().storage().to_vec_f64()[0];
                    println!("Second-order gradient: {:.6}", second_grad_value);
                    println!("Expected for x³ at x=2: 12.0");
                    
                    if (second_grad_value - 12.0).abs() < 1e-3 {
                        println!("✅ Test Hessian PASSED");
                    } else {
                        println!("❌ Test Hessian FAILED - expected 12.0, got {}", second_grad_value);
                    }
                } else {
                    println!("❌ Second order gradient is None");
                }
            } else {
                println!("❌ Hessian matrix is empty");
            }
        },
        Err(e) => {
            println!("❌ Erreur calcul Hessienne: {}", e);
        }
    }
}