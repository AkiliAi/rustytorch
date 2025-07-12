//! Tests pour les gradients d'ordre supérieur
//! 
//! Tests pour Hessienne, gradients de n-ième ordre, etc.

use rustytorch_autograd::{Variable, enable_grad};

#[test]
fn test_second_order_simple() {
    // Test gradient d'ordre 2 pour f(x) = x³
    // f'(x) = 3x², f''(x) = 6x
    let _guard = enable_grad();
    
    let x = Variable::variable_with_grad(&[2.0], vec![1]);
    let y = x.mul(&x).mul(&x); // x³
    
    // Calculer la Hessienne
    let hessian = y.hessian(&[x.clone()]).unwrap();
    
    assert!(!hessian.is_empty());
    assert!(!hessian[0].is_empty());
    
    if let Some(second_grad) = &hessian[0][0] {
        let second_grad_value = second_grad.tensor().storage().to_vec_f64()[0];
        // f''(2) = 6 * 2 = 12
        let expected = 12.0;
        let error = (second_grad_value - expected).abs() / expected;
        
        println!("🧪 Test: Second order gradient of x³");
        println!("   Expected: {:.6}, Got: {:.6}, Error: {:.2e}", 
                 expected, second_grad_value, error);
        
        assert!(error < 1e-5, "Second order gradient test failed with error: {:.2e}", error);
    } else {
        panic!("Second order gradient is None");
    }
}

#[test]
fn test_hessian_quadratic() {
    // Test Hessienne pour f(x,y) = x² + xy + y²
    // Hessienne = [[2, 1], [1, 2]]
    let _guard = enable_grad();
    
    let x = Variable::variable_with_grad(&[1.0], vec![1]);
    let y = Variable::variable_with_grad(&[2.0], vec![1]);
    
    let x_squared = x.mul(&x);
    let y_squared = y.mul(&y);
    let xy = x.mul(&y);
    let f = x_squared.add(&xy).add(&y_squared);
    
    // Calculer la Hessienne
    let hessian = f.hessian(&[x.clone(), y.clone()]).unwrap();
    
    assert_eq!(hessian.len(), 2);
    assert_eq!(hessian[0].len(), 2);
    
    let mut hessian_values = [[0.0; 2]; 2];
    let expected = [[2.0, 1.0], [1.0, 2.0]];
    
    for i in 0..2 {
        for j in 0..2 {
            if let Some(h_ij) = &hessian[i][j] {
                hessian_values[i][j] = h_ij.tensor().storage().to_vec_f64()[0];
            }
        }
    }
    
    println!("🧪 Test: Hessian of quadratic function");
    println!("   Expected: [[{:.1}, {:.1}], [{:.1}, {:.1}]]", 
             expected[0][0], expected[0][1], expected[1][0], expected[1][1]);
    println!("   Got:      [[{:.1}, {:.1}], [{:.1}, {:.1}]]", 
             hessian_values[0][0], hessian_values[0][1], 
             hessian_values[1][0], hessian_values[1][1]);
    
    for i in 0..2 {
        for j in 0..2 {
            let error = (hessian_values[i][j] - expected[i][j]).abs();
            assert!(error < 1e-5, "Hessian element ({},{}) error: {:.2e}", i, j, error);
        }
    }
}

#[test]
fn test_third_order_gradients() {
    // Test gradient d'ordre 3 pour f(x) = x⁴
    // f'(x) = 4x³, f''(x) = 12x², f'''(x) = 24x
    let _guard = enable_grad();
    
    let x = Variable::variable_with_grad(&[2.0], vec![1]);
    let x2 = x.mul(&x);
    let x4 = x2.mul(&x2); // x⁴
    
    // Gradient d'ordre 3
    let third_order = x4.nth_order_grad(&[x.clone()], 3).unwrap();
    
    assert!(!third_order.is_empty());
    
    if let Some(grad3) = &third_order[0] {
        let grad3_value = grad3.tensor().storage().to_vec_f64()[0];
        // f'''(2) = 24 * 2 = 48
        let expected = 48.0;
        let error = (grad3_value - expected).abs() / expected;
        
        println!("🧪 Test: Third order gradient of x⁴");
        println!("   Expected: {:.1}, Got: {:.1}, Error: {:.2e}", 
                 expected, grad3_value, error);
        
        assert!(error < 1e-4, "Third order gradient test failed with error: {:.2e}", error);
    } else {
        panic!("Third order gradient is None");
    }
}

#[test]
fn test_jacobian_vector_function() {
    // Test Jacobien pour fonction vectorielle
    // f1(x,y) = x + y, f2(x,y) = x * y
    // J = [[1, 1], [y, x]]
    let _guard = enable_grad();
    
    let x = Variable::variable_with_grad(&[2.0], vec![1]);
    let y = Variable::variable_with_grad(&[3.0], vec![1]);
    
    let f1 = x.add(&y);      // f1 = x + y
    let f2 = x.mul(&y);      // f2 = x * y
    
    let jacobian = Variable::jacobian(&[f1, f2], &[x.clone(), y.clone()]).unwrap();
    
    let expected = [[1.0, 1.0], [3.0, 2.0]]; // [[df1/dx, df1/dy], [df2/dx, df2/dy]]
    let mut jacobian_values = [[0.0; 2]; 2];
    
    for i in 0..2 {
        for j in 0..2 {
            if let Some(j_ij) = &jacobian[i][j] {
                jacobian_values[i][j] = j_ij.tensor().storage().to_vec_f64()[0];
            }
        }
    }
    
    println!("🧪 Test: Jacobian of vector function");
    println!("   Expected: [[{:.1}, {:.1}], [{:.1}, {:.1}]]", 
             expected[0][0], expected[0][1], expected[1][0], expected[1][1]);
    println!("   Got:      [[{:.1}, {:.1}], [{:.1}, {:.1}]]", 
             jacobian_values[0][0], jacobian_values[0][1], 
             jacobian_values[1][0], jacobian_values[1][1]);
    
    for i in 0..2 {
        for j in 0..2 {
            let error = (jacobian_values[i][j] - expected[i][j]).abs();
            assert!(error < 1e-6, "Jacobian element ({},{}) error: {:.2e}", i, j, error);
        }
    }
}

#[test]
fn test_mixed_partial_derivatives() {
    // Test dérivées partielles mixtes pour f(x,y) = x²y + xy²
    // ∂²f/∂x∂y = 2x + 2y
    let _guard = enable_grad();
    
    let x = Variable::variable_with_grad(&[1.0], vec![1]);
    let y = Variable::variable_with_grad(&[2.0], vec![1]);
    
    let x2 = x.mul(&x);
    let y2 = y.mul(&y);
    let x2y = x2.mul(&y);
    let xy2 = x.mul(&y2);
    let f = x2y.add(&xy2); // f = x²y + xy²
    
    let hessian = f.hessian(&[x.clone(), y.clone()]).unwrap();
    
    // ∂²f/∂x∂y = ∂²f/∂y∂x = 2x + 2y = 2*1 + 2*2 = 6
    let expected_mixed = 6.0;
    
    if let (Some(h_xy), Some(h_yx)) = (&hessian[0][1], &hessian[1][0]) {
        let h_xy_val = h_xy.tensor().storage().to_vec_f64()[0];
        let h_yx_val = h_yx.tensor().storage().to_vec_f64()[0];
        
        println!("🧪 Test: Mixed partial derivatives");
        println!("   Expected: {:.1}", expected_mixed);
        println!("   ∂²f/∂x∂y = {:.1}", h_xy_val);
        println!("   ∂²f/∂y∂x = {:.1}", h_yx_val);
        
        let error_xy = (h_xy_val - expected_mixed).abs();
        let error_yx = (h_yx_val - expected_mixed).abs();
        let symmetry_error = (h_xy_val - h_yx_val).abs();
        
        assert!(error_xy < 1e-5, "Mixed derivative ∂²f/∂x∂y error: {:.2e}", error_xy);
        assert!(error_yx < 1e-5, "Mixed derivative ∂²f/∂y∂x error: {:.2e}", error_yx);
        assert!(symmetry_error < 1e-10, "Hessian symmetry error: {:.2e}", symmetry_error);
    } else {
        panic!("Mixed partial derivatives are None");
    }
}

#[test]
fn test_fourth_order_constant() {
    // Test gradient d'ordre 4 pour f(x) = x⁴
    // f''''(x) = 24 (constant)
    let _guard = enable_grad();
    
    let x = Variable::variable_with_grad(&[1.5], vec![1]);
    let x2 = x.mul(&x);
    let x4 = x2.mul(&x2); // x⁴
    
    // Gradient d'ordre 4
    let fourth_order = x4.nth_order_grad(&[x.clone()], 4).unwrap();
    
    assert!(!fourth_order.is_empty());
    
    if let Some(grad4) = &fourth_order[0] {
        let grad4_value = grad4.tensor().storage().to_vec_f64()[0];
        // f''''(x) = 24 pour tout x
        let expected = 24.0;
        let error = (grad4_value - expected).abs();
        
        println!("🧪 Test: Fourth order gradient of x⁴");
        println!("   Expected: {:.1}, Got: {:.1}, Error: {:.2e}", 
                 expected, grad4_value, error);
        
        assert!(error < 1e-3, "Fourth order gradient test failed with error: {:.2e}", error);
    } else {
        panic!("Fourth order gradient is None");
    }
}