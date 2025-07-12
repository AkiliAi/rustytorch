//! Module de tests exhaustifs pour rustytorch_autograd
//! 
//! Ce module organise tous les tests de validation des gradients et des fonctionnalités autograd.

pub mod gradient_validation;
pub mod basic_operations;
pub mod activations;
pub mod mathematical_functions;
pub mod higher_order_gradients;

// Re-export des utilitaires de test
pub use gradient_validation::{gradient_check, GradientCheckResult};

/// Fonction utilitaire pour exécuter une suite complète de tests
pub fn run_comprehensive_gradient_tests() {
    println!("🚀 Exécution des tests exhaustifs de gradients...\n");
    
    // Les tests individuels seront exécutés par cargo test
    println!("ℹ️  Utilisez 'cargo test' pour exécuter tous les tests de validation des gradients.");
    println!("ℹ️  Utilisez 'cargo test --test gradient_validation' pour des tests spécifiques.");
    
    println!("\n📋 Tests disponibles:");
    println!("   • basic_operations: Tests des opérations arithmétiques de base");
    println!("   • activations: Tests des fonctions d'activation (ReLU, Sigmoid, Tanh)");
    println!("   • mathematical_functions: Tests des fonctions mathématiques (exp, log, sin, cos)");
    println!("   • higher_order_gradients: Tests des gradients d'ordre supérieur (Hessienne, etc.)");
    println!("   • gradient_validation: Tests génériques de validation numérique");
}