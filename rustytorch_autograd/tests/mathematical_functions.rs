//! Tests pour les fonctions mathématiques
//! 
//! Tests pour exp, log, sin, cos, etc.

use rustytorch_autograd::{Variable, enable_grad};

use crate::gradient_validation::{gradient_check, DEFAULT_TOLERANCE};

#[test]
fn test_exponential() {
    // Test exp(x) avec des valeurs modérées pour éviter overflow
    let x = Variable::variable_with_grad(&[-1.0, -0.5, 0.0, 0.5, 1.0], vec![5]);
    
    let result = gradient_check(
        &[x],
        |inputs| inputs[0].exp(),
        DEFAULT_TOLERANCE,
        "Exponential exp(x)",
    );
    
    assert!(result.passed);
}

#[test]
fn test_natural_logarithm() {
    // Test log(x) avec des valeurs strictement positives
    let x = Variable::variable_with_grad(&[0.1, 0.5, 1.0, 2.0, 5.0], vec![5]);
    
    let result = gradient_check(
        &[x],
        |inputs| inputs[0].log(),
        DEFAULT_TOLERANCE,
        "Natural logarithm log(x)",
    );
    
    assert!(result.passed);
}

#[test]
fn test_sine() {
    // Test sin(x) sur différents quadrants
    let x = Variable::variable_with_grad(&[-1.57, -0.78, 0.0, 0.78, 1.57], vec![5]); // ~[-π/2, π/2]
    
    let result = gradient_check(
        &[x],
        |inputs| inputs[0].sin(),
        DEFAULT_TOLERANCE,
        "Sine sin(x)",
    );
    
    assert!(result.passed);
}

#[test]
fn test_cosine() {
    // Test cos(x) sur différents quadrants
    let x = Variable::variable_with_grad(&[-1.57, -0.78, 0.0, 0.78, 1.57], vec![5]); // ~[-π/2, π/2]
    
    let result = gradient_check(
        &[x],
        |inputs| inputs[0].cos(),
        DEFAULT_TOLERANCE,
        "Cosine cos(x)",
    );
    
    assert!(result.passed);
}

#[test]
fn test_tangent() {
    // Test tan(x) en évitant les asymptotes
    let x = Variable::variable_with_grad(&[-1.0, -0.5, 0.0, 0.5, 1.0], vec![5]);
    
    let result = gradient_check(
        &[x],
        |inputs| inputs[0].tan(),
        DEFAULT_TOLERANCE,
        "Tangent tan(x)",
    );
    
    assert!(result.passed);
}

#[test]
fn test_power_with_positive_base() {
    // Test x^n avec base positive
    let x = Variable::variable_with_grad(&[0.5, 1.0, 1.5, 2.0], vec![4]);
    
    let result = gradient_check(
        &[x],
        |inputs| inputs[0].pow(3.0),
        DEFAULT_TOLERANCE,
        "Power x^3",
    );
    
    assert!(result.passed);
}

#[test]
fn test_power_fractional() {
    // Test x^(1/2) = sqrt(x) avec x > 0
    let x = Variable::variable_with_grad(&[0.25, 1.0, 4.0, 9.0], vec![4]);
    
    let result = gradient_check(
        &[x],
        |inputs| inputs[0].pow(0.5),
        DEFAULT_TOLERANCE,
        "Power x^0.5 (square root)",
    );
    
    assert!(result.passed);
}

#[test]
fn test_exp_log_composition() {
    // Test exp(log(x)) = x pour x > 0
    let x = Variable::variable_with_grad(&[0.5, 1.0, 2.0], vec![3]);
    
    let result = gradient_check(
        &[x],
        |inputs| inputs[0].log().exp(),
        DEFAULT_TOLERANCE,
        "Composition: exp(log(x))",
    );
    
    assert!(result.passed);
}

#[test]
fn test_trigonometric_identity() {
    // Test sin²(x) + cos²(x) = 1 via dérivation
    let x = Variable::variable_with_grad(&[0.5], vec![1]);
    
    let result = gradient_check(
        &[x],
        |inputs| {
            let sin_x = inputs[0].sin();
            let cos_x = inputs[0].cos();
            let sin_squared = sin_x.mul(&sin_x);
            let cos_squared = cos_x.mul(&cos_x);
            sin_squared.add(&cos_squared)
        },
        DEFAULT_TOLERANCE,
        "Trigonometric identity: sin²(x) + cos²(x)",
    );
    
    assert!(result.passed);
}

#[test]
fn test_complex_mathematical_function() {
    // f(x) = exp(sin(x)) * log(1 + x²)
    let x = Variable::variable_with_grad(&[0.5], vec![1]);
    
    let result = gradient_check(
        &[x],
        |inputs| {
            let x = &inputs[0];
            let sin_x = x.sin();
            let exp_sin_x = sin_x.exp();
            
            let x_squared = x.mul(x);
            let one = Variable::variable_with_grad(&[1.0], vec![1]);
            let one_plus_x_squared = one.add(&x_squared);
            let log_term = one_plus_x_squared.log();
            
            exp_sin_x.mul(&log_term)
        },
        DEFAULT_TOLERANCE,
        "Complex function: exp(sin(x)) * log(1 + x²)",
    );
    
    assert!(result.passed);
}