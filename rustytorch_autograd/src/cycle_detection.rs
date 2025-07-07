// rustytorch_autograd/src/cycle_detection.rs

use crate::{Node, Variable};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Type d'erreur pour les opérations d'autograd
#[derive(Debug, Clone)]
pub enum AutogradError {
    /// Un cycle a été détecté dans le graphe de calcul
    CycleDetected(String),
    /// Erreur générique pour les opérations d'autograd
    OperationFailed(String),
}

impl std::fmt::Display for AutogradError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AutogradError::CycleDetected(message) => {
                write!(f, "Cycle detected in computation graph: {}", message)
            }
            AutogradError::OperationFailed(message) => {
                write!(f, "Autograd operation failed: {}", message)
            }
        }
    }
}

impl std::error::Error for AutogradError {}

/// Structure pour détecter les cycles dans le graphe de calcul
pub struct CycleDetector {
    /// Map des nœuds visités pendant le parcours actuel
    visiting: HashSet<usize>,
    /// Map des nœuds déjà visités et confirmés sans cycle
    visited: HashSet<usize>,
}

impl CycleDetector {
    /// Crée un nouveau détecteur de cycles
    pub fn new() -> Self {
        Self {
            visiting: HashSet::new(),
            visited: HashSet::new(),
        }
    }

    /// Vérifie si un graphe de calcul contient des cycles
    pub fn check_cycles(&mut self, var: &Variable) -> Result<(), AutogradError> {
        // Réinitialiser l'état pour une nouvelle vérification
        self.visiting.clear();
        self.visited.clear();

        // Commencer le parcours DFS à partir de la variable
        self.dfs_check(var)
    }

    /// Parcours en profondeur pour détecter les cycles
    fn dfs_check(&mut self, var: &Variable) -> Result<(), AutogradError> {
        if !var.requires_grad {
            // Si la variable ne requiert pas de gradient, pas besoin de vérifier
            return Ok(());
        }

        // Utiliser l'adresse du tenseur comme identifiant unique
        let tensor_id = &var.tensor as *const _ as usize;

        // Si ce nœud est déjà visité et confirmé sans cycle, retourner immédiatement
        if self.visited.contains(&tensor_id) {
            return Ok(());
        }

        // Si nous revisitions un nœud en cours de visite, c'est un cycle
        if self.visiting.contains(&tensor_id) {
            return Err(AutogradError::CycleDetected(format!(
                "Cycle detected involving tensor at address {:p}",
                &var.tensor
            )));
        }

        // Marquer ce nœud comme en cours de visite
        self.visiting.insert(tensor_id);

        // Vérifier récursivement tous les nœuds d'entrée
        if let Some(ref grad_fn) = var.grad_fn {
            for input_var in &grad_fn.inputs {
                self.dfs_check(input_var)?;
            }
        }

        // Marquer ce nœud comme visité et le retirer des nœuds en cours de visite
        self.visiting.remove(&tensor_id);
        self.visited.insert(tensor_id);

        Ok(())
    }
}

/// Extension du module Variable pour intégrer la détection de cycles
impl Variable {
    /// Vérifie si le graphe de calcul contient des cycles avant la rétropropagation
    pub fn check_cycles(&self) -> Result<(), AutogradError> {
        let mut detector = CycleDetector::new();
        detector.check_cycles(self)
    }

    /// Version améliorée de backward avec détection de cycles
    pub fn backward_safe(&mut self) -> Result<(), AutogradError> {
        // Vérifier d'abord s'il y a des cycles
        self.check_cycles()?;

        // Si pas de cycle, procéder à la rétropropagation normalement
        self.backward();
        Ok(())
    }
}

// Tests pour la détection de cycles
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Operation;
    use rustytorch_tensor::Tensor;

    #[test]
    fn test_no_cycle() {
        // Créer un graphe simple sans cycle
        let tensor_a = Tensor::from_data(&[2.0], vec![1], None);
        let tensor_b = Tensor::from_data(&[3.0], vec![1], None);

        let var_a = Variable::from_tensor(tensor_a, true);
        let var_b = Variable::from_tensor(tensor_b, true);

        // c = a * b
        let var_c = var_a.mul(&var_b);

        // Vérifier qu'il n'y a pas de cycle
        let result = var_c.check_cycles();
        assert!(result.is_ok());
    }

    #[test]
    #[ignore]
    fn test_artificial_cycle() {
        // Créer artificiellement un graphe avec un cycle
        let tensor_a = Tensor::from_data(&[2.0], vec![1], None);
        let tensor_b = Tensor::from_data(&[3.0], vec![1], None);

        let var_a = Variable::from_tensor(tensor_a.clone(), true);
        let var_b = Variable::from_tensor(tensor_b, true);

        // c = a * b
        let var_c = var_a.mul(&var_b);

        // Créer artificiellement un cycle: a -> c -> a
        let node = Node {
            operation: Operation::Mul,
            inputs: vec![var_c.clone()],
            grad_fn: None,
        };

        let mut var_a_cyclic = var_a.clone();
        var_a_cyclic.grad_fn = Some(Arc::new(node));

        // Modifier var_c pour inclure var_a_cyclic dans ses entrées
        let cyclic_node = Node {
            operation: Operation::Mul,
            inputs: vec![var_a_cyclic, var_b],
            grad_fn: None,
        };

        let mut var_c_cyclic = var_c;
        var_c_cyclic.grad_fn = Some(Arc::new(cyclic_node));

        // Vérifier que le cycle est détecté
        let result = var_c_cyclic.check_cycles();
        println!("Cycle detection result: {:?}", result);
        assert!(result.is_err());

        if let Err(AutogradError::CycleDetected(_)) = result {
            // Correct, un cycle a été détecté
        } else {
            panic!("Expected CycleDetected error, got: {:?}", result);
        }
    }
}
