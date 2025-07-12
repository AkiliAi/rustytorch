use rustytorch_autograd::{Variable, enable_grad};

pub fn run_debug_gradient() {
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
    
    // Test first-order gradient
    println!("Testing first-order gradients with create_graph=true...");
    match Variable::compute_grad(&[y.clone()], &[x.clone()], None, true, true) {
        Ok(first_grads) => {
            if let Some(first_grad) = &first_grads[0] {
                let first_value = first_grad.tensor().storage().to_vec_f64()[0];
                println!("First-order gradient computed: {:.6}", first_value);
                println!("First-order gradient requires_grad: {}", first_grad.requires_grad());
                println!("First-order gradient is_leaf: {}", first_grad.is_leaf());
                // println!("First-order gradient has grad_fn: {}", first_grad.has_grad_fn());
                
                // Now compute second-order gradient
                println!("Computing second-order from first gradient...");
                match Variable::compute_grad(&[first_grad.clone()], &[x.clone()], None, false, false) {
                    Ok(second_grads) => {
                        if let Some(second_grad) = &second_grads[0] {
                            let second_value = second_grad.tensor().storage().to_vec_f64()[0];
                            println!("✅ Second-order gradient: {:.6}", second_value);
                        } else {
                            println!("❌ Second gradient is None");
                        }
                    }
                    Err(e) => {
                        println!("❌ Error computing second-order gradient: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Error computing first-order gradient: {}", e);
        }
    }
    
    match y.hessian(&[x.clone()]) {
        Ok(hessian) => {
            println!("Hessian calculation succeeded");
            println!("Hessian dimensions: {}x{}", hessian.len(), hessian[0].len());
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