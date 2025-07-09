//! Tests de validation numérique pour les gradients
//! 
//! Ce module contient des tests exhaustifs pour valider que nos gradients analytiques
//! correspondent aux gradients numériques calculés par différences finies.

use rustytorch_autograd::{Variable, enable_grad};
use rustytorch_tensor::Tensor;

/// Tolérance par défaut pour les comparaisons de gradients
pub const DEFAULT_TOLERANCE: f64 = 1e-3;

/// Epsilon pour les différences finies
const FINITE_DIFF_EPS: f64 = 1e-6;

/// Structure pour encapsuler les résultats de validation de gradients
#[derive(Debug)]
pub struct GradientCheckResult {
    pub passed: bool,
    pub max_error: f64,
    pub mean_error: f64,
    pub num_elements: usize,
    pub details: Vec<(usize, f64, f64, f64)>, // (index, analytical, numerical, error)
}

impl GradientCheckResult {
    pub fn print_summary(&self, test_name: &str) {
        println!("🧪 Test: {}", test_name);
        if self.passed {
            println!("   ✅ PASSED - Max error: {:.2e}, Mean error: {:.2e}", 
                     self.max_error, self.mean_error);
        } else {
            println!("   ❌ FAILED - Max error: {:.2e}, Mean error: {:.2e}", 
                     self.max_error, self.mean_error);
            
            // Afficher les 3 pires erreurs
            let mut sorted_details = self.details.clone();
            sorted_details.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
            
            println!("   Worst errors:");
            for (i, (idx, analytical, numerical, error)) in sorted_details.iter().take(3).enumerate() {
                println!("     {}. Index {}: analytical={:.6e}, numerical={:.6e}, error={:.6e}", 
                         i+1, idx, analytical, numerical, error);
            }
        }
        println!();
    }
}

/// Fonction générique pour vérifier les gradients par différences finies
pub fn gradient_check<F>(
    inputs: &[Variable],
    output_fn: F,
    tolerance: f64,
    test_name: &str,
) -> GradientCheckResult
where
    F: Fn(&[Variable]) -> Variable,
{
    let _guard = enable_grad();
    
    // Calculer la sortie et les gradients analytiques
    let output = output_fn(inputs);
    let analytical_grads = Variable::compute_grad(
        &[output.clone()], 
        inputs, 
        None, 
        false, 
        false
    ).expect("Failed to compute analytical gradients");
    
    let mut all_errors = Vec::new();
    let mut max_error: f64 = 0.0;
    let mut total_error: f64 = 0.0;
    let mut num_elements = 0;
    let mut passed = true;
    
    // Pour chaque input, calculer le gradient numérique
    for (input_idx, input) in inputs.iter().enumerate() {
        if let Some(analytical_grad) = &analytical_grads[input_idx] {
            let analytical_values = analytical_grad.tensor().storage().to_vec_f64();
            let input_values = input.tensor().storage().to_vec_f64();
            let input_shape = input.shape();
            
            // Calculer le gradient numérique pour chaque élément
            for (elem_idx, _) in input_values.iter().enumerate() {
                // f(x + h)
                let mut perturbed_up = input_values.clone();
                perturbed_up[elem_idx] += FINITE_DIFF_EPS;
                let input_up = Variable::from_tensor(
                    Tensor::from_data(&perturbed_up, input_shape.clone(), None),
                    false,
                );
                
                // f(x - h)
                let mut perturbed_down = input_values.clone();
                perturbed_down[elem_idx] -= FINITE_DIFF_EPS;
                let input_down = Variable::from_tensor(
                    Tensor::from_data(&perturbed_down, input_shape.clone(), None),
                    false,
                );
                
                // Créer les nouveaux inputs avec la perturbation
                let mut inputs_up = inputs.to_vec();
                let mut inputs_down = inputs.to_vec();
                inputs_up[input_idx] = input_up;
                inputs_down[input_idx] = input_down;
                
                // Calculer f(x+h) et f(x-h)
                let output_up = output_fn(&inputs_up);
                let output_down = output_fn(&inputs_down);
                
                let f_plus = output_up.tensor().storage().to_vec_f64()[0];
                let f_minus = output_down.tensor().storage().to_vec_f64()[0];
                
                // Debug: imprimer les valeurs pour comprendre le problème (commenté)
                // if elem_idx == 0 && input_idx == 0 {
                //     println!("DEBUG: f_plus = {}, f_minus = {}", f_plus, f_minus);
                //     println!("DEBUG: input_val = {}, eps = {}", input_values[elem_idx], FINITE_DIFF_EPS);
                //     println!("DEBUG: perturbed_up = {:?}", perturbed_up);
                //     println!("DEBUG: perturbed_down = {:?}", perturbed_down);
                // }
                
                // Gradient numérique: (f(x+h) - f(x-h)) / (2*h)
                let numerical_grad = (f_plus - f_minus) / (2.0 * FINITE_DIFF_EPS);
                let analytical_val = analytical_values[elem_idx];
                
                // Calculer l'erreur relative
                let error = if analytical_val.abs() > 1e-10 {
                    ((analytical_val - numerical_grad) / analytical_val).abs()
                } else {
                    (analytical_val - numerical_grad).abs()
                };
                
                all_errors.push((
                    input_idx * input_values.len() + elem_idx,
                    analytical_val,
                    numerical_grad,
                    error,
                ));
                
                max_error = max_error.max(error);
                total_error += error;
                num_elements += 1;
                
                if error > tolerance {
                    passed = false;
                }
            }
        }
    }
    
    let mean_error = if num_elements > 0 { total_error / num_elements as f64 } else { 0.0 };
    
    let result = GradientCheckResult {
        passed,
        max_error,
        mean_error,
        num_elements,
        details: all_errors,
    };
    
    result.print_summary(test_name);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition_gradients() {
        let x = Variable::variable_with_grad(&[2.0, 3.0], vec![2]);
        let y = Variable::variable_with_grad(&[1.0, 4.0], vec![2]);
        
        let result = gradient_check(
            &[x, y],
            |inputs| inputs[0].add(&inputs[1]),
            DEFAULT_TOLERANCE,
            "Addition (x + y)",
        );
        
        assert!(result.passed, "Addition gradient check failed with max error: {:.2e}", result.max_error);
    }
    
    #[test]
    fn test_multiplication_gradients() {
        let x = Variable::variable_with_grad(&[2.0, 3.0], vec![2]);
        let y = Variable::variable_with_grad(&[1.5, 0.5], vec![2]);
        
        let result = gradient_check(
            &[x, y],
            |inputs| inputs[0].mul(&inputs[1]),
            DEFAULT_TOLERANCE,
            "Multiplication (x * y)",
        );
        
        assert!(result.passed, "Multiplication gradient check failed with max error: {:.2e}", result.max_error);
    }
    
    #[test]
    fn test_quadratic_function() {
        // f(x, y) = x² + xy + y²
        let x = Variable::variable_with_grad(&[1.0], vec![1]);
        let y = Variable::variable_with_grad(&[2.0], vec![1]);
        
        let result = gradient_check(
            &[x.clone(), y.clone()],
            |inputs| {
                let x = &inputs[0];
                let y = &inputs[1];
                let x_squared = x.mul(x);
                let y_squared = y.mul(y);
                let xy = x.mul(y);
                x_squared.add(&xy).add(&y_squared)
            },
            DEFAULT_TOLERANCE,
            "Quadratic function f(x,y) = x² + xy + y²",
        );
        
        assert!(result.passed, "Quadratic function gradient check failed with max error: {:.2e}", result.max_error);
    }
}