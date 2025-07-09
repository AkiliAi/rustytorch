//rustytorch_examples/src/main.rs

mod advanced_linalg;
mod autograd_basic_demo;
mod decompositions_demo;
mod device_demo;
mod f16_demo;
mod higher_order_gradients_demo;
mod initializers_demo;
mod memory_pool_demo;
mod neural_network_demo;
mod new_reductions;
mod optimization_demo;
mod padding_demo;
mod pow_test;
mod random_generators_demo;

use rustytorch_autograd::{enable_grad, Variable};
use rustytorch_core::{Reduction, Reshapable};
use rustytorch_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RustyTorch - Démonstrations Complètes\n");
    
    // Activer le calcul de gradient par défaut
    let _grad_guard = enable_grad();
    
    // === NOUVELLES DÉMONSTRATIONS AUTOGRAD ===
    println!("🧠 DÉMONSTRATIONS AUTOGRAD - NOUVELLES FONCTIONNALITÉS\n");
    
    // Démonstration autograd de base avec la nouvelle API
    autograd_basic_demo::run_autograd_basic_demo();
    
    // Démonstration gradients d'ordre supérieur
    higher_order_gradients_demo::run_higher_order_gradients_demo();
    
    // Démonstration réseau de neurones mini
    neural_network_demo::run_neural_network_demo();
    
    // Démonstration algorithmes d'optimisation
    optimization_demo::run_optimization_demo();
    
    // === DEBUG: Test gradient simple ===
    println!("=== DEBUG: Test gradient simple ===");
    debug_simple_gradients();
    
    // === Exemple rapide de la nouvelle API ===
    println!("=== Exemple Rapide: Nouvelle API Autograd ===");
    quick_autograd_example();
    
    // === DÉMONSTRATIONS DES AUTRES MODULES ===
    println!("\n📊 DÉMONSTRATIONS DES AUTRES MODULES\n");

    // Lancer la démonstration des nouvelles réductions
    println!("{}", "=".repeat(60));
    new_reductions::run_new_reductions_demo();

    // Lancer la démonstration du padding et cropping
    println!("{}", "=".repeat(60));
    padding_demo::run_padding_demo();

    // Lancer la démonstration d'algèbre linéaire avancée
    println!("{}", "=".repeat(60));
    advanced_linalg::run_advanced_linalg_demo();

    // Lancer la démonstration des générateurs aléatoires
    println!("{}", "=".repeat(60));
    random_generators_demo::run_random_generators_demo();

    // Lancer la démonstration des initialiseurs
    println!("{}", "=".repeat(60));
    initializers_demo::run_initializers_demo();

    // Lancer la démonstration des décompositions
    println!("{}", "=".repeat(60));
    decompositions_demo::run_decompositions_demo();

    // Lancer la démonstration des devices
    println!("{}", "=".repeat(60));
    device_demo::run_device_demo();

    // Lancer la démonstration F16
    println!("{}", "=".repeat(60));
    f16_demo::run_f16_demo();

    // Lancer la démonstration Memory Pool
    println!("{}", "=".repeat(60));
    memory_pool_demo::run_memory_pool_demo();

    println!("\n🎉 Toutes les démonstrations terminées avec succès!");
    println!("📈 Statistiques du graphe: {:?}", Variable::graph_stats());

    Ok(())
}

/// Exemple rapide montrant les nouvelles fonctionnalités d'autograd
fn quick_autograd_example() {
    println!("   Calcul: f(x,y) = x²y + sin(xy)");
    
    let x = Variable::variable_with_grad(&[1.5], vec![1]);
    let y = Variable::variable_with_grad(&[2.0], vec![1]);
    
    // f(x,y) = x²y + sin(xy)
    let x_squared = x.mul(&x);
    let x2y = x_squared.mul(&y);
    let xy = x.mul(&y);
    let sin_xy = xy.sin();
    let f = x2y.add(&sin_xy);
    
    println!("   x = 1.5, y = 2.0");
    println!("   f(1.5, 2.0) = {:.4}", f.tensor().storage().to_vec_f64()[0]);
    
    // Gradients de premier ordre
    let grads = Variable::compute_grad(&[f.clone()], &[x.clone(), y.clone()], None, false, true).unwrap();
    if let (Some(dx), Some(dy)) = (&grads[0], &grads[1]) {
        println!("   ∂f/∂x = {:.4}", dx.tensor().storage().to_vec_f64()[0]);
        println!("   ∂f/∂y = {:.4}", dy.tensor().storage().to_vec_f64()[0]);
    }
    
    // Hessienne (gradients de second ordre)
    match f.hessian(&[x.clone(), y.clone()]) {
        Ok(hessian) => {
            if let (Some(h_xx), Some(h_xy), Some(h_yx), Some(h_yy)) = 
                (&hessian[0][0], &hessian[0][1], &hessian[1][0], &hessian[1][1]) {
                println!("   Hessienne:");
                println!("   [[{:.3}, {:.3}],", 
                         h_xx.tensor().storage().to_vec_f64()[0],
                         h_xy.tensor().storage().to_vec_f64()[0]);
                println!("    [{:.3}, {:.3}]]", 
                         h_yx.tensor().storage().to_vec_f64()[0],
                         h_yy.tensor().storage().to_vec_f64()[0]);
            }
        }
        Err(_) => println!("   Erreur calcul Hessienne"),
    }
    
    println!();
}

/// Debug fonction pour tester les gradients de base
fn debug_simple_gradients() {
    println!("Testing basic gradient functionality...");
    
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
    
    // First, let's test if first-order gradients work with create_graph=true
    println!("Testing first-order gradients with create_graph=true...");
    match Variable::compute_grad(&[y.clone()], &[x.clone()], None, true, true) {
        Ok(first_grads) => {
            if let Some(first_grad) = &first_grads[0] {
                println!("First-order gradient computed: {:.6}", first_grad.tensor().storage().to_vec_f64()[0]);
                println!("First-order gradient requires_grad: {}", first_grad.requires_grad());
                println!("First-order gradient is_leaf: {}", first_grad.is_leaf());
                println!("First-order gradient has grad_fn: {}", first_grad.grad_fn());
                
                // Now try second-order  
                println!("Computing second-order from first gradient...");
                match Variable::compute_grad(&[first_grad.clone()], &[x.clone()], None, false, false) {
                    Ok(second_grads) => {
                        if let Some(second_grad) = &second_grads[0] {
                            println!("✅ Second-order gradient: {:.6}", second_grad.tensor().storage().to_vec_f64()[0]);
                        } else {
                            println!("❌ Second-order gradient is None");
                        }
                    },
                    Err(e) => {
                        println!("❌ Error computing second-order gradient: {}", e);
                    }
                }
            } else {
                println!("❌ First-order gradient is None");
            }
        },
        Err(e) => {
            println!("❌ Error computing first-order gradient: {}", e);
        }
    }
    
    match y.hessian(&[x.clone()]) {
        Ok(hessian) => {
            println!("Hessian calculation succeeded");
            println!("Hessian dimensions: {}x{}", hessian.len(), if hessian.is_empty() { 0 } else { hessian[0].len() });
            
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
    
    // Test pow operation
    println!("🧪 Testing pow operation...");
    pow_test::test_pow_operation();
    
    println!();
}

