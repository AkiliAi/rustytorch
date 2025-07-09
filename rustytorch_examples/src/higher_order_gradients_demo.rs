//! Démonstration des gradients d'ordre supérieur

use rustytorch_autograd::Variable;

pub fn run_higher_order_gradients_demo() {
    println!("=== Démonstration: Gradients d'Ordre Supérieur ===\n");

    // === Exemple 1: Hessienne d'une fonction quadratique ===
    println!("1. Matrice Hessienne: f(x,y) = x² + xy + y²");
    let x = Variable::variable_with_grad(&[1.0], vec![1]);
    let y = Variable::variable_with_grad(&[2.0], vec![1]);
    
    let x_squared = x.mul(&x);
    let y_squared = y.mul(&y);
    let xy = x.mul(&y);
    let f = x_squared.add(&xy).add(&y_squared);
    
    println!("   x = 1.0, y = 2.0");
    println!("   f = x² + xy + y² = {:.2}", f.tensor().storage().to_vec_f64()[0]);
    
    // Calculer la Hessienne
    match f.hessian(&[x.clone(), y.clone()]) {
        Ok(hessian) => {
            println!("   Matrice Hessienne H =");
            println!("   ┌                    ┐");
            
            let h00 = hessian[0][0].as_ref().map(|h| h.tensor().storage().to_vec_f64()[0]).unwrap_or(0.0);
            let h01 = hessian[0][1].as_ref().map(|h| h.tensor().storage().to_vec_f64()[0]).unwrap_or(0.0);
            let h10 = hessian[1][0].as_ref().map(|h| h.tensor().storage().to_vec_f64()[0]).unwrap_or(0.0);
            let h11 = hessian[1][1].as_ref().map(|h| h.tensor().storage().to_vec_f64()[0]).unwrap_or(0.0);
            
            println!("   │ {:6.1}   {:6.1} │", h00, h01);
            println!("   │ {:6.1}   {:6.1} │", h10, h11);
            println!("   └                    ┘");
            println!("   (Attendu: [[2, 1], [1, 2]])\n");
        }
        Err(e) => println!("   Erreur de calcul Hessienne: {}\n", e),
    }

    // === Exemple 2: Gradients d'ordre n ===
    println!("2. Gradients d'ordre n: f(x) = x⁴");
    let x = Variable::variable_with_grad(&[2.0], vec![1]);
    let x2 = x.mul(&x);
    let x4 = x2.mul(&x2); // x⁴
    
    println!("   x = 2.0");
    println!("   f = x⁴ = {:.2}", x4.tensor().storage().to_vec_f64()[0]);
    
    // Calculer les gradients successifs
    for order in 1..=4 {
        match x4.nth_order_grad(&[x.clone()], order) {
            Ok(grad) => {
                if let Some(grad_val) = &grad[0] {
                    let val = grad_val.tensor().storage().to_vec_f64()[0];
                    match order {
                        1 => println!("   f'(x) = 4x³ = {:.2} (attendu: 32.0)", val),
                        2 => println!("   f''(x) = 12x² = {:.2} (attendu: 48.0)", val),
                        3 => println!("   f'''(x) = 24x = {:.2} (attendu: 48.0)", val),
                        4 => println!("   f''''(x) = 24 = {:.2} (attendu: 24.0)", val),
                        _ => {},
                    }
                }
            }
            Err(e) => println!("   Erreur gradient ordre {}: {}", order, e),
        }
    }
    println!();

    // === Exemple 3: Jacobien d'une fonction vectorielle ===
    println!("3. Matrice Jacobienne: F(x,y) = [x + y, x*y, x² - y²]");
    let x = Variable::variable_with_grad(&[3.0], vec![1]);
    let y = Variable::variable_with_grad(&[2.0], vec![1]);
    
    let f1 = x.add(&y);                    // f1 = x + y
    let f2 = x.mul(&y);                    // f2 = x * y  
    let x_sq = x.mul(&x);
    let y_sq = y.mul(&y);
    let f3 = x_sq.sub(&y_sq);             // f3 = x² - y²
    
    println!("   x = 3.0, y = 2.0");
    println!("   F = [{:.1}, {:.1}, {:.1}]", 
             f1.tensor().storage().to_vec_f64()[0],
             f2.tensor().storage().to_vec_f64()[0],
             f3.tensor().storage().to_vec_f64()[0]);
    
    match Variable::jacobian(&[f1, f2, f3], &[x.clone(), y.clone()]) {
        Ok(jacobian) => {
            println!("   Matrice Jacobienne J =");
            println!("   ┌                    ┐");
            for i in 0..3 {
                let j_i0 = jacobian[i][0].as_ref().map(|j| j.tensor().storage().to_vec_f64()[0]).unwrap_or(0.0);
                let j_i1 = jacobian[i][1].as_ref().map(|j| j.tensor().storage().to_vec_f64()[0]).unwrap_or(0.0);
                println!("   │ {:6.1}   {:6.1} │", j_i0, j_i1);
            }
            println!("   └                    ┘");
            println!("   (Attendu: [[1, 1], [2, 3], [6, -4]])\n");
        }
        Err(e) => println!("   Erreur de calcul Jacobienne: {}\n", e),
    }

    // === Exemple 4: Optimisation de second ordre ===
    println!("4. Simulation d'optimisation Newton: f(x) = x² - 4x + 3");
    let mut x = Variable::variable_with_grad(&[0.5], vec![1]); // Point de départ
    
    println!("   Méthode de Newton: x_new = x - f'(x)/f''(x)");
    println!("   f(x) = x² - 4x + 3");
    println!("   Point de départ: x₀ = 0.5\n");
    
    for iter in 0..3 {
        // f(x) = x² - 4x + 3
        let x_sq = x.mul(&x);
        let four = Variable::from_tensor(rustytorch_tensor::Tensor::from_data(&[4.0], vec![1], None), false);
        let three = Variable::from_tensor(rustytorch_tensor::Tensor::from_data(&[3.0], vec![1], None), false);
        let four_x = four.mul(&x);
        let f = x_sq.sub(&four_x).add(&three);
        
        let x_val = x.tensor().storage().to_vec_f64()[0];
        let f_val = f.tensor().storage().to_vec_f64()[0];
        
        // Calculer f'(x) et f''(x)
        let first_order = f.nth_order_grad(&[x.clone()], 1).unwrap();
        let second_order = f.nth_order_grad(&[x.clone()], 2).unwrap();
        
        if let (Some(fp), Some(fpp)) = (&first_order[0], &second_order[0]) {
            let fp_val = fp.tensor().storage().to_vec_f64()[0];
            let fpp_val = fpp.tensor().storage().to_vec_f64()[0];
            
            println!("   Iter {}: x = {:.4}, f(x) = {:.4}, f'(x) = {:.4}, f''(x) = {:.4}", 
                     iter, x_val, f_val, fp_val, fpp_val);
            
            // Mise à jour Newton: x_new = x - f'(x)/f''(x)
            let newton_step = fp_val / fpp_val;
            let new_x_val = x_val - newton_step;
            x = Variable::variable_with_grad(&[new_x_val], vec![1]);
            
            println!("           x_new = {:.4} - {:.4} = {:.4}", x_val, newton_step, new_x_val);
        }
    }
    
    println!("   Minimum théorique: x = 2.0, f(2) = -1.0");

    println!("\n=== Fin de la démonstration Gradients d'Ordre Supérieur ===\n");
}