use rustytorch_autograd::{Variable, enable_grad};

pub fn test_pow_operation() {
    println!("🧪 Testing pow operation...");
    
    let _guard = enable_grad();
    
    // Test simple pow operation
    let x = Variable::variable_with_grad(&[2.0], vec![1]);
    let result = x.pow(3.0);
    
    println!("x = {:?}", x.tensor().storage().to_vec_f64());
    println!("x^3 = {:?}", result.tensor().storage().to_vec_f64());
    
    // Test gradient
    let grad = Variable::compute_grad(&[result.clone()], &[x.clone()], None, false, false);
    
    match grad {
        Ok(grads) => {
            if let Some(grad_x) = &grads[0] {
                println!("d/dx(x^3) = {:?}", grad_x.tensor().storage().to_vec_f64());
                println!("Expected: [12.0] (3 * 2^2)");
                println!("✅ Pow operation working correctly!");
            } else {
                println!("❌ No gradient computed");
            }
        }
        Err(e) => {
            println!("❌ Error computing gradient: {}", e);
        }
    }
    
    // Test with different values
    println!("\n🧪 Testing with multiple values...");
    let x2 = Variable::variable_with_grad(&[2.0, 3.0], vec![2]);
    let result2 = x2.pow(2.5);
    
    println!("x = {:?}", x2.tensor().storage().to_vec_f64());
    println!("x^2.5 = {:?}", result2.tensor().storage().to_vec_f64());
    
    let grad2 = Variable::compute_grad(&[result2.clone()], &[x2.clone()], None, false, false);
    
    match grad2 {
        Ok(grads) => {
            if let Some(grad_x) = &grads[0] {
                println!("d/dx(x^2.5) = {:?}", grad_x.tensor().storage().to_vec_f64());
                println!("Expected: [7.07, 12.99] (2.5 * x^1.5)");
                println!("✅ Pow operation working for multiple values!");
            }
        }
        Err(e) => {
            println!("❌ Error computing gradient: {}", e);
        }
    }
    
    // Test Hessian with pow
    println!("\n🧪 Testing Hessian with pow...");
    let x3 = Variable::variable_with_grad(&[2.0], vec![1]);
    let y3 = x3.pow(3.0); // x³
    
    match y3.hessian(&[x3.clone()]) {
        Ok(hessian) => {
            if !hessian.is_empty() && !hessian[0].is_empty() {
                if let Some(second_grad) = &hessian[0][0] {
                    let second_grad_value = second_grad.tensor().storage().to_vec_f64()[0];
                    println!("Second-order gradient (Hessian): {:.6}", second_grad_value);
                    println!("Expected for x³ at x=2: 12.0");
                    
                    if (second_grad_value - 12.0).abs() < 1e-3 {
                        println!("✅ Test Hessian with pow PASSED");
                    } else {
                        println!("❌ Test Hessian with pow FAILED - expected 12.0, got {}", second_grad_value);
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Error computing Hessian: {}", e);
        }
    }
}