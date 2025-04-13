//rustytorch_autograd/src/lib.rs

pub mod cycle_detection;
pub mod operations;

/// Créez un module pour l'autograd


// use rustytorch_tensor::TensorError;

use rustytorch_tensor::Tensor;
use rustytorch_core::{NumericOps, Reduction, Reshapable};
use std::collections::HashMap;
use std::sync::Arc;
use std::cell::RefCell;
use std::env::vars;
use std::thread_local;
use std::fmt::{Debug, Display, Formatter};

// Variable globale pour activer/désactiver le calcul du gradient
thread_local! {
    static GRAD_ENABLED: RefCell<bool> = RefCell::new(true);
    static VARIABLES: RefCell<HashMap<usize, RefCell<Option<Tensor>>>> = RefCell::new(HashMap::new());
    static NEXT_ID: RefCell<usize> = RefCell::new(0);   // ID unique pour chaque variable

}
// Fonction pour obtenir un nouvel ID unique
fn get_next_id() -> usize {
    NEXT_ID.with(|id| {
        let new_id = *id.borrow();
        *id.borrow_mut() += 1;
        new_id
    })
}

///Node pour le graphe de calcul
pub struct Node{
    pub operation: Operation,
    pub inputs: Vec<Variable>,
    pub grad_fn: Option<Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>>,
}

// Implémenter Clone manuellement
impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            operation: self.operation.clone(),
            inputs: self.inputs.clone(),
            grad_fn: None, // Nous ne pouvons pas cloner la fonction
        }
    }
}

// Implémenter Debug manuellement
impl Debug for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("operation", &self.operation)
            .field("inputs", &self.inputs)
            .field("grad_fn", &"<function>".to_string())
            .finish()
    }
}

/// Structure pour suivre les Operations executées
#[derive(Clone,Debug)]
pub enum Operation{
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Pow,
    Exp,
    Log,
    Sigmoid,
    Relu,
    Tanh,
    Softmax,
    Sum,
    Mean,
    None,
    // Autres opérations à ajouter...
}

impl Display for Operation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::Add => write!(f, "Add"),
            Operation::Sub => write!(f, "Sub"),
            Operation::Mul => write!(f, "Mul"),
            Operation::Div => write!(f, "Div"),
            Operation::MatMul => write!(f, "MatMul"),
            Operation::Pow => write!(f, "Pow"),
            Operation::Exp => write!(f, "Exp"),
            Operation::Log => write!(f, "Log"),
            Operation::Sigmoid => write!(f, "Sigmoid"),
            Operation::Relu => write!(f, "ReLU"),
            Operation::Tanh => write!(f, "Tanh"),
            Operation::Softmax => write!(f, "Softmax"),
            Operation::Sum => write!(f, "Sum"),
            Operation::Mean => write!(f, "Mean"),
            Operation::None => write!(f, "None"),
        }
    }
}

// Variable avec suivi de gradient
#[derive(Clone,Debug)]
pub struct Variable{
    pub tensor: Tensor,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub grad: Option<Tensor>,
    pub grad_fn: Option<Arc<Node>>,
    pub id: usize,  // ID unique variable
}

impl Variable {
    // Cree une nouvelle variable a partir d'un tenseur
    pub fn from_tensor(tensor: Tensor, requires_grad: bool) -> Self {
        let id = get_next_id();

        if requires_grad {
            VARIABLES.with(|vars| {
                vars.borrow_mut().insert(id, RefCell::new(None));
            });
        }

        Self {
            tensor,
            requires_grad,
            is_leaf: true,
            grad: None,
            grad_fn: None,
            id,
        }
    }

    // Cree une variable resultante d'une operation
    pub fn from_operation(
        tensor: Tensor,
        operation: Operation,
        inputs: Vec<Variable>,
        grad_fn: Option<Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>>,
    ) -> Self {
        let requires_grad = GRAD_ENABLED.with(|cell| *cell.borrow()) &&
            inputs.iter().any(|v| v.requires_grad);

        let grad_fn = if requires_grad {
            let node = Node {
                operation,
                inputs: inputs.clone(),
                grad_fn,
            };
            Some(Arc::new(node))
        } else {
            None
        };

        let id = get_next_id();

        Self {
            tensor,
            requires_grad,
            is_leaf: false,
            grad: None,
            grad_fn,
            id,
        }
    }

    /// Addition de deux variables
    pub fn add(&self, other: &Self) -> Self {
        // Opération sur les tenseurs sous-jacents
        let result_tensor = match self.tensor.clone().add(other.tensor.clone()) {
            Ok(t) => t,
            Err(e) => panic!("Error in add operation: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour l'addition
        // Pour c = a + b, dc/da = 1 et dc/db = 1
        let grad_fn = if self.requires_grad || other.requires_grad {
            Some(Box::new(move |grad_output: &Tensor| {
                vec![grad_output.clone(), grad_output.clone()]
            }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(
            result_tensor,
            Operation::Add,
            vec![self.clone(), other.clone()],
            grad_fn,
        )
    }

    /// Soustraction de deux variables
    pub fn sub(&self, other: &Self) -> Self {
        // Opération sur les tenseurs sous-jacents
        let result_tensor = match self.tensor.clone().sub(other.tensor.clone()) {
            Ok(t) => t,
            Err(e) => panic!("Error in sub operation: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour la soustraction
        // Pour c = a - b, dc/da = 1 et dc/db = -1
        let grad_fn = if self.requires_grad || other.requires_grad {
            Some(Box::new(move |grad_output: &Tensor| {
                let negative_grad = match grad_output.clone().mul(Tensor::from_data(&[-1.0], vec![1], None)) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for sub: {}", e),
                };
                vec![grad_output.clone(), negative_grad]
            }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(
            result_tensor,
            Operation::Sub,
            vec![self.clone(), other.clone()],
            grad_fn,
        )
    }

    /// Multiplication élément par élément de deux variables
    pub fn mul(&self, other: &Self) -> Self {
        // Opération sur les tenseurs sous-jacents
        let result_tensor = match self.tensor.clone().mul(other.tensor.clone()) {
            Ok(t) => t,
            Err(e) => panic!("Error in mul operation: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour la multiplication
        // Pour c = a * b, dc/da = b et dc/db = a
        let a_clone = self.tensor.clone();
        let b_clone = other.tensor.clone();

        let grad_fn = if self.requires_grad || other.requires_grad {
            Some(Box::new(move |grad_output: &Tensor| {
                let grad_a = match grad_output.clone().mul(b_clone.clone()) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for mul: {}", e),
                };
                let grad_b = match grad_output.clone().mul(a_clone.clone()) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for mul: {}", e),
                };
                vec![grad_a, grad_b]
            }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(
            result_tensor,
            Operation::Mul,
            vec![self.clone(), other.clone()],
            grad_fn,
        )
    }

    /// Division élément par élément de deux variables
    pub fn div(&self, other: &Self) -> Self {
        // Opération sur les tenseurs sous-jacents
        let result_tensor = match self.tensor.clone().div(other.tensor.clone()) {
            Ok(t) => t,
            Err(e) => panic!("Error in div operation: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour la division
        // Pour c = a / b, dc/da = 1/b et dc/db = -a/b^2
        let a_clone = self.tensor.clone();
        let b_clone = other.tensor.clone();

        let grad_fn = if self.requires_grad || other.requires_grad {
            Some(Box::new(move |grad_output: &Tensor| {
                // Calcul de 1/b pour dc/da
                let one = Tensor::ones(vec![1], None);
                let b_inv = match one.clone().div(b_clone.clone()) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing 1/b for div gradient: {}", e),
                };
                let grad_a = match grad_output.clone().mul(b_inv) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing grad_a for div: {}", e),
                };

                // Calcul de -a/b^2 pour dc/db
                let b_squared = match b_clone.clone().mul(b_clone.clone()) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing b^2 for div gradient: {}", e),
                };
                let b_squared_inv = match one.div(b_squared) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing 1/b^2 for div gradient: {}", e),
                };
                let a_div_b_squared = match a_clone.clone().mul(b_squared_inv) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing a/b^2 for div gradient: {}", e),
                };
                let minus_one = Tensor::from_data(&[-1.0], vec![1], None);
                let grad_b = match match grad_output.clone().mul(a_div_b_squared) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing partial grad_b for div: {}", e),
                }.mul(minus_one) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing final grad_b for div: {}", e),
                };

                vec![grad_a, grad_b]
            }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(
            result_tensor,
            Operation::Div,
            vec![self.clone(), other.clone()],
            grad_fn,
        )
    }

    /// Calcule la somme de tous les elements du tenseur
    pub fn sum(&self) -> Self {
        let result_tensor = match self.tensor.sum() {
            Ok(t) => t,
            Err(e) => panic!("Error in sum operation: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Pour la rétropropagation, le gradient de sum par rapport à chaque élément est 1
        let self_clone = self.clone();
        let grad_fn = Box::new(move |_grad_output: &Tensor| {
            // Pour sum(), le gradient par rapport à chaque élément de l'entrée est 1
            let ones = Tensor::ones(self_clone.tensor.shape().to_vec(), None);
            vec![ones]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;

        // Créer la variable résultante
        Self::from_operation(
            result_tensor,
            Operation::Sum,
            vec![self.clone()],
            Some(grad_fn),
        )
    }

    // Implémentez de façon similaire d'autres méthodes comme mean(), exp(), log(), etc.

    /// Multiplication matricielle de deux variables
    pub fn matmul(&self, other: &Self) -> Self {
        // Vérifier si on peut faire la multiplication matricielle
        let a_shape = self.tensor.shape();
        let b_shape = other.tensor.shape();

        if a_shape.len() < 2 || b_shape.len() < 2 {
            panic!("Matrix multiplication requires at least 2D tensors");
        }

        let a_cols = a_shape[a_shape.len() - 1];
        let b_rows = b_shape[b_shape.len() - 2];

        if a_cols != b_rows {
            panic!("Matrix multiplication shape mismatch: {:?} and {:?}", a_shape, b_shape);
        }

        // Opération sur les tenseurs sous-jacents
        let result_tensor = match self.tensor.matmul(&other.tensor) {
            Ok(t) => t,
            Err(e) => panic!("Error in matmul: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour la multiplication matricielle
        // Pour C = A @ B, dC/dA = dC @ B.T et dC/dB = A.T @ dC
        let a_clone = self.tensor.clone();
        let b_clone = other.tensor.clone();

        let grad_fn = if self.requires_grad || other.requires_grad {
            Some(Box::new(move |grad_output: &Tensor| {
                // Pour simplifier, nous supposons que les tenseurs sont 2D
                // Pour les tenseurs de dimensions supérieures, plus de travail serait nécessaire

                // Transposons B pour calculer dC/dA = dC @ B.T
                let b_transposed = match b_clone.transpose(0, 1) {
                    Ok(t) => t,
                    Err(e) => panic!("Error transposing B for matmul gradient: {}", e),
                };
                let grad_a = match grad_output.matmul(&b_transposed) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for matmul: {}", e),
                };

                // Transposons A pour calculer dC/dB = A.T @ dC
                let a_transposed = match a_clone.transpose(0, 1) {
                    Ok(t) => t,
                    Err(e) => panic!("Error transposing A for matmul gradient: {}", e),
                };
                let grad_b = match a_transposed.matmul(grad_output) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for matmul: {}", e),
                };

                vec![grad_a, grad_b]
            }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(
            result_tensor,
            Operation::MatMul,
            vec![self.clone(), other.clone()],
            grad_fn,
        )
    }

    /// Calcule le gradient de cette variable par rapport aux entrées
    pub fn backward(&mut self) {
        if !self.requires_grad {
            return;
        }

        // Structure pour suivre les gradients accumulés
        let mut grad_table: HashMap<usize, Tensor> = HashMap::new();

        // File d'attente pour la propagation du gradient
        let mut queue: Vec<(Arc<Node>, Tensor)> = Vec::new();

        // Initialiser le gradient de sortie à 1 s'il n'est pas défini
        if self.grad.is_none() {
            self.grad = Some(Tensor::ones(self.tensor.shape().to_vec(), None));
        }

        // Si cette variable a une fonction de gradient, l'ajouter à la file d'attente
        if let Some(ref grad_fn) = self.grad_fn {
            queue.push((grad_fn.clone(), self.grad.clone().unwrap()));
        } else if self.is_leaf {
            // Pour les feuilles, stocker le gradient directement
            grad_table.insert(self.id, self.grad.clone().unwrap());
        }

        // Propager les gradients à travers le graphe
        while let Some((node, grad_output)) = queue.pop() {
            if let Some(ref grad_fn) = node.grad_fn {
                let input_grads = grad_fn(&grad_output);

                assert_eq!(input_grads.len(), node.inputs.len(),
                           "Number of gradients doesn't match number of inputs");

                for (input_var, input_grad) in node.inputs.iter().zip(input_grads.iter()) {
                    if !input_var.requires_grad {
                        continue;
                    }

                    // Utiliser l'ID plutôt que l'adresse mémoire
                    if let Some(existing_grad) = grad_table.get(&input_var.id) {
                        let new_grad = match existing_grad.clone().add(input_grad.clone()) {
                            Ok(t) => t,
                            Err(e) => panic!("Error accumulating gradients: {}", e),
                        };
                        grad_table.insert(input_var.id, new_grad);
                    } else {
                        grad_table.insert(input_var.id, input_grad.clone());
                    }

                    if let Some(ref input_grad_fn) = input_var.grad_fn {
                        queue.push((input_grad_fn.clone(), input_grad.clone()));
                    }
                }
            }
        }

        // Mettre à jour les gradients des variables feuilles
        for (var_id, grad) in grad_table {
            VARIABLES.with(|vars| {
                if let Some(var_grad) = vars.borrow().get(&var_id) {
                    *var_grad.borrow_mut() = Some(grad.clone());
                }
            });

            // Mise à jour du gradient dans cette variable si nécessaire
            if var_id == self.id {
                self.grad = Some(grad);
            }
        }
    }

    pub fn grad(&self) -> Option<Tensor> {
        if self.is_leaf && self.requires_grad {
            VARIABLES.with(|vars| {
                if let Some(var_grad) = vars.borrow().get(&self.id) {
                    return var_grad.borrow().clone();
                }
                None
            })
        } else {
            self.grad.clone()
        }
    }


    // /// convertir une variable en f64
    // pub fn to_f64(&self) -> Result<f64, String> {
    //     match self.tensor.storage().as_ref() {
    //         rustytorch_tensor::storage::StorageType::F32(data) => Ok(data[0] as f64),
    //         rustytorch_tensor::storage::StorageType::F64(data) => Ok(data[0]),
    //         _ => Err("Unsupported storage type for conversion to f64".to_string()),
    //     }
    // }
}

// Context pour desactiver temporairement le calcul du gradient
pub struct NoGradGuard {
    prev_enabled: bool,
}

/// Implémentation de NoGradGuard pour désactiver le calcul du gradient
impl NoGradGuard {
    pub fn new() -> Self {
        let prev = GRAD_ENABLED.with(|cell| *cell.borrow());
        GRAD_ENABLED.with(|cell| *cell.borrow_mut() = false);
        Self { prev_enabled: prev }
    }
}

/// Implémentation de Drop pour restaurer l'état précédent
impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|cell| *cell.borrow_mut() = self.prev_enabled);
    }
}

/// Fonction utilitaire pour créer un guard qui désactive le calcul de gradient
pub fn no_grad() -> NoGradGuard {
    NoGradGuard::new()
}










//
// // Tests pour l'autograd
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_variable_creation() {
//         let tensor = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
//         let var = Variable::from_tensor(tensor, true);
//
//         assert!(var.requires_grad);
//         assert!(var.is_leaf);
//         assert!(var.grad.is_none());
//         assert!(var.grad_fn.is_none());
//     }
//
//     #[test]
//     fn test_add_operation() {
//         let tensor_a = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
//         let tensor_b = Tensor::from_data(&[4.0, 5.0, 6.0], vec![3], None);
//
//         let var_a = Variable::from_tensor(tensor_a, true);
//         let var_b = Variable::from_tensor(tensor_b, true);
//
//         let var_c = var_a.add(&var_b);
//
//         assert!(var_c.requires_grad);
//         assert!(!var_c.is_leaf);
//         assert!(var_c.grad.is_none());
//         assert!(var_c.grad_fn.is_some());
//
//         // Vérifier que le tenseur résultant contient les bonnes valeurs
//         match var_c.tensor.storage().as_ref() {
//             rustytorch_tensor::storage::StorageType::F32(data) => {
//                 assert_eq!(data, &[5.0, 7.0, 9.0]);
//             },
//             rustytorch_tensor::storage::StorageType::F64(data) => {
//                 assert_eq!(data, &[5.0, 7.0, 9.0]);
//             },
//             _ => panic!("Unexpected storage type"),
//         }
//     }
//
//     #[test]
//     fn test_backward_simple() {
//         // Créer deux variables
//         let tensor_a = Tensor::from_data(&[2.0], vec![1], None);
//         let tensor_b = Tensor::from_data(&[3.0], vec![1], None);
//
//         let mut var_a = Variable::from_tensor(tensor_a, true);
//         let mut var_b = Variable::from_tensor(tensor_b, true);
//
//         // Calculer c = a * b
//         let mut var_c = var_a.mul(&var_b);
//
//         // Propagation arrière
//         var_c.backward();
//
//         // Vérifier les gradients:
//         // dc/da = b = 3
//         // dc/db = a = 2
//         if let Some(grad_a) = &var_a.grad {
//             match grad_a.storage().as_ref() {
//                 rustytorch_tensor::storage::StorageType::F32(data) => {
//                     assert_eq!(data[0], 3.0);
//                 },
//                 rustytorch_tensor::storage::StorageType::F64(data) => {
//                     assert_eq!(data[0], 3.0);
//                 },
//                 _ => panic!("Unexpected storage type"),
//             }
//         } else {
//             panic!("Gradient for var_a is None");
//         }
//
//         if let Some(grad_b) = &var_b.grad {
//             match grad_b.storage().as_ref() {
//                 rustytorch_tensor::storage::StorageType::F32(data) => {
//                     assert_eq!(data[0], 2.0);
//                 },
//                 rustytorch_tensor::storage::StorageType::F64(data) => {
//                     assert_eq!(data[0], 2.0);
//                 },
//                 _ => panic!("Unexpected storage type"),
//             }
//         } else {
//             panic!("Gradient for var_b is None");
//         }
//     }
//
//     #[test]
//     fn test_no_grad() {
//         // Créer deux variables
//         let tensor_a = Tensor::from_data(&[2.0], vec![1], None);
//         let tensor_b = Tensor::from_data(&[3.0], vec![1], None);
//
//         // Avec no_grad, les opérations ne devraient pas créer de graphe de calcul
//         {
//             let _guard = no_grad();
//
//             let var_a = Variable::from_tensor(tensor_a.clone(), true);
//             let var_b = Variable::from_tensor(tensor_b.clone(), true);
//
//             let var_c = var_a.add(&var_b);
//
//             // Même si requires_grad est vrai pour les entrées, il devrait être faux pour le résultat
//             assert!(!var_c.requires_grad);
//             assert!(var_c.grad_fn.is_none());
//         }
//     }
//
//     #[test]
//     fn test_complex_graph() {
//         // Créer des variables pour un exemple plus complexe
//         // Exemple: f(x, y) = (x + 2*y) * (x^2)
//         let tensor_x = Tensor::from_data(&[3.0], vec![1], None);
//         let tensor_y = Tensor::from_data(&[4.0], vec![1], None);
//
//         let var_x = Variable::from_tensor(tensor_x, true);
//         let var_y = Variable::from_tensor(tensor_y, true);
//
//         // Calculer 2*y
//         let two = Variable::from_tensor(Tensor::from_data(&[2.0], vec![1], None), false);
//         let two_y = two.mul(&var_y);
//
//         // Calculer x + 2*y
//         let x_plus_2y = var_x.add(&two_y);
//
//         // Calculer x^2
//         let x_squared = var_x.mul(&var_x);
//
//         // Calculer (x + 2*y) * (x^2)
//         let mut result = x_plus_2y.mul(&x_squared);
//
//         // Propager les gradients
//         result.backward();
//
//         // Les gradients devraient être:
//         // df/dx = d/dx[(x + 2*y) * (x^2)]
//         //       = (x^2) * d/dx(x + 2*y) + (x + 2*y) * d/dx(x^2)
//         //       = (x^2) * 1 + (x + 2*y) * 2*x
//         //       = x^2 + 2*x*(x + 2*y)
//         // Pour x=3, y=4: df/dx = 3^2 + 2*3*(3 + 2*4) = 9 + 6*11 = 9 + 66 = 75
//         //
//         // df/dy = d/dy[(x + 2*y) * (x^2)]
//         //       = (x^2) * d/dy(x + 2*y) + (x + 2*y) * d/dy(x^2)
//         //       = (x^2) * 2 + (x + 2*y) * 0
//         //       = 2*x^2
//         // Pour x=3, y=4: df/dy = 2*3^2 = 2*9 = 18
//
//         // TODO: Vérifier les gradients calculés
//         // Cette vérification devrait être activée quand l'implémentation complète de backward sera terminée
//     }
// }
//


// //rustytorch_autograd/src/lib.rs
//
// mod operations;
//
// use rustytorch_tensor::Tensor;
// use rustytorch_core::{NumericOps, Reduction, Reshapable};
// use std::collections::{HashMap, HashSet};
// use std::sync::Arc;
// use std::cell::RefCell;
// use std::thread_local;
// use std::fmt::{Display, Formatter};
// use std::time::{Duration, Instant};
// use std::fs::File;
// use std::io::Write;
// use std::error::Error;
//
// // Variables globales pour activer/désactiver le calcul du gradient et stocker les états
// thread_local! {
//     static GRAD_ENABLED: RefCell<bool> = RefCell::new(true);
//     static VARIABLES: RefCell<HashMap<usize, (RefCell<Option<Tensor>>, Instant)>> =
//         RefCell::new(HashMap::new());
//     static NEXT_ID: RefCell<usize> = RefCell::new(0);   // ID unique pour chaque variable
//     static LAST_CLEANUP: RefCell<Instant> = RefCell::new(Instant::now());
// }
//
// // Fonction pour obtenir un nouvel ID unique
// fn get_next_id() -> usize {
//     NEXT_ID.with(|id| {
//         let new_id = *id.borrow();
//         *id.borrow_mut() += 1;
//         new_id
//     })
// }
//
// // Fonction pour nettoyer les variables non utilisées
// fn maybe_cleanup() {
//     const CLEANUP_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes
//     const MAX_AGE: Duration = Duration::from_secs(600); // 10 minutes
//
//     LAST_CLEANUP.with(|last| {
//         let now = Instant::now();
//         if now.duration_since(*last.borrow()) > CLEANUP_INTERVAL {
//             *last.borrow_mut() = now;
//
//             // Nettoyer les variables anciennes
//             VARIABLES.with(|vars| {
//                 vars.borrow_mut().retain(|_, (_, timestamp)| {
//                     now.duration_since(*timestamp) < MAX_AGE
//                 });
//             });
//         }
//     });
// }
//
// ///Node pour le graphe de calcul
// pub struct Node{
//     pub operation: Operation,
//     pub inputs: Vec<Variable>,
//     pub grad_fn: Option<Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>>,
// }
//
// // Implémenter Clone manuellement
// impl Clone for Node {
//     fn clone(&self) -> Self {
//         Self {
//             operation: self.operation.clone(),
//             inputs: self.inputs.clone(),
//             grad_fn: None, // Nous ne pouvons pas cloner la fonction
//         }
//     }
// }
//
// // Implémenter Debug manuellement
// impl std::fmt::Debug for Node {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("Node")
//             .field("operation", &self.operation)
//             .field("inputs", &self.inputs)
//             .field("grad_fn", &format!("<function>"))
//             .finish()
//     }
// }
//
// /// Structure pour suivre les Operations executées
// #[derive(Clone, Debug)]
// pub enum Operation {
//     Add,
//     Sub,
//     Mul,
//     Div,
//     MatMul,
//     Pow,
//     Exp,
//     Log,
//     Sigmoid,
//     Relu,
//     Tanh,
//     Softmax,
//     Sum,  // Ajout de l'opération Sum
//     Tan,
//     Cos,
//     Sin,
//     Mean,
//     None,
// }
//
// impl Display for Operation {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Operation::Add => write!(f, "Add"),
//             Operation::Sub => write!(f, "Sub"),
//             Operation::Mul => write!(f, "Mul"),
//             Operation::Div => write!(f, "Div"),
//             Operation::MatMul => write!(f, "MatMul"),
//             Operation::Pow => write!(f, "Pow"),
//             Operation::Exp => write!(f, "Exp"),
//             Operation::Log => write!(f, "Log"),
//             Operation::Sigmoid => write!(f, "Sigmoid"),
//             Operation::Relu => write!(f, "ReLU"),
//             Operation::Tanh => write!(f, "Tanh"),
//             Operation::Softmax => write!(f, "Softmax"),
//             Operation::Sum => write!(f, "Sum"),
//             Operation::Tan => write!(f, "Tan"),
//             Operation::Cos => write!(f, "Cos"),
//             Operation::Sin => write!(f, "Sin"),
//             Operation::Mean => write!(f, "Mean"),
//             Operation::None => write!(f, "None"),
//         }
//     }
// }
//
// // Variable avec suivi de gradient
// #[derive(Clone, Debug)]
// pub struct Variable {
//     pub tensor: Tensor,
//     pub requires_grad: bool,
//     pub is_leaf: bool,
//     pub grad: Option<Tensor>,
//     pub grad_fn: Option<Arc<Node>>,
//     pub id: usize,  // ID unique variable
//     pub is_gradient: bool, // Indique si la variable est un gradient
//     pub retain_graph: bool, // Pour conserver le graphe de calcul
// }
//
// impl Variable {
//     // Cree une nouvelle variable a partir d'un tenseur
//     pub fn from_tensor(tensor: Tensor, requires_grad: bool) -> Self {
//         let id = get_next_id();
//
//         if requires_grad {
//             VARIABLES.with(|vars| {
//                 vars.borrow_mut().insert(id, (RefCell::new(None), Instant::now()));
//             });
//         }
//
//         Self {
//             tensor,
//             requires_grad,
//             is_leaf: true,
//             grad: None,
//             grad_fn: None,
//             id,
//             is_gradient: false,
//             retain_graph: false,
//         }
//     }
//
//     // Cree une variable resultante d'une operation
//     pub fn from_operation(
//         tensor: Tensor,
//         operation: Operation,
//         inputs: Vec<Variable>,
//         grad_fn: Option<Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>>,
//     ) -> Self {
//         // Vérifier si le calcul du gradient est activé et si au moins une entrée requiert un gradient
//         let requires_grad = GRAD_ENABLED.with(|cell| *cell.borrow()) &&
//             inputs.iter().any(|v| v.requires_grad);
//
//         let grad_fn = if requires_grad {
//             // Créer un nœud pour cette opération
//             let node = Node {
//                 operation,
//                 inputs: inputs.clone(),
//                 grad_fn,
//             };
//             Some(Arc::new(node))
//         } else {
//             None
//         };
//
//         let id = get_next_id();
//
//         Self {
//             tensor,
//             requires_grad,
//             is_leaf: false,
//             grad: None,
//             grad_fn,
//             id,
//             is_gradient: false,
//             retain_graph: false,
//         }
//     }
//
//     /// Addition de deux variables
//     pub fn add(&self, other: &Self) -> Self {
//         // Opération sur les tenseurs sous-jacents
//         let result_tensor = self.tensor.clone().add(other.tensor.clone());
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour l'addition
//         // Pour c = a + b, dc/da = 1 et dc/db = 1
//         let grad_fn = if self.requires_grad || other.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 vec![grad_output.clone(), grad_output.clone()]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Add,
//             vec![self.clone(), other.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Soustraction de deux variables
//     pub fn sub(&self, other: &Self) -> Self {
//         // Opération sur les tenseurs sous-jacents
//         let result_tensor = self.tensor.clone().sub(other.tensor.clone());
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour la soustraction
//         // Pour c = a - b, dc/da = 1 et dc/db = -1
//         let grad_fn = if self.requires_grad || other.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 let negative_grad = grad_output.clone().mul(Tensor::from_data(&[-1.0], vec![1], None));
//                 vec![grad_output.clone(), negative_grad]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Sub,
//             vec![self.clone(), other.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Multiplication élément par élément de deux variables
//     pub fn mul(&self, other: &Self) -> Self {
//         // Opération sur les tenseurs sous-jacents
//         let result_tensor = self.tensor.clone().mul(other.tensor.clone());
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour la multiplication
//         // Pour c = a * b, dc/da = b et dc/db = a
//         let a_clone = self.tensor.clone();
//         let b_clone = other.tensor.clone();
//
//         let grad_fn = if self.requires_grad || other.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 let grad_a = grad_output.clone().mul(b_clone.clone());
//                 let grad_b = grad_output.clone().mul(a_clone.clone());
//                 vec![grad_a, grad_b]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Mul,
//             vec![self.clone(), other.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Division élément par élément de deux variables
//     pub fn div(&self, other: &Self) -> Self {
//         // Opération sur les tenseurs sous-jacents
//         let result_tensor = self.tensor.clone().div(other.tensor.clone());
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour la division
//         // Pour c = a / b, dc/da = 1/b et dc/db = -a/b^2
//         let a_clone = self.tensor.clone();
//         let b_clone = other.tensor.clone();
//
//         let grad_fn = if self.requires_grad || other.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 // Calcul de 1/b pour dc/da
//                 let one = Tensor::ones(vec![1], None);
//                 let b_inv = one.clone().div(b_clone.clone());
//                 let grad_a = grad_output.clone().mul(b_inv);
//
//                 // Calcul de -a/b^2 pour dc/db
//                 let b_squared = b_clone.clone().mul(b_clone.clone());
//                 let b_squared_inv = one.div(b_squared);
//                 let a_div_b_squared = a_clone.clone().mul(b_squared_inv);
//                 let minus_one = Tensor::from_data(&[-1.0], vec![1], None);
//                 let grad_b = grad_output.clone().mul(a_div_b_squared).mul(minus_one);
//
//                 vec![grad_a, grad_b]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Div,
//             vec![self.clone(), other.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Multiplication matricielle de deux variables
//     pub fn matmul(&self, other: &Self) -> Self {
//         // Vérifier si on peut faire la multiplication matricielle
//         let a_shape = self.tensor.shape();
//         let b_shape = other.tensor.shape();
//
//         if a_shape.len() < 2 || b_shape.len() < 2 {
//             panic!("Matrix multiplication requires at least 2D tensors");
//         }
//
//         let a_cols = a_shape[a_shape.len() - 1];
//         let b_rows = b_shape[b_shape.len() - 2];
//
//         if a_cols != b_rows {
//             panic!("Matrix multiplication shape mismatch: {:?} and {:?}", a_shape, b_shape);
//         }
//
//         // Opération sur les tenseurs sous-jacents
//         let result_tensor = match self.tensor.matmul(&other.tensor) {
//             Ok(t) => t,
//             Err(e) => panic!("Error in matmul: {}", e),
//         };
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour la multiplication matricielle
//         // Pour C = A @ B, dC/dA = dC @ B.T et dC/dB = A.T @ dC
//         let a_clone = self.tensor.clone();
//         let b_clone = other.tensor.clone();
//
//         let grad_fn = if self.requires_grad || other.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 // Pour simplifier, nous supposons que les tenseurs sont 2D
//                 // Pour les tenseurs de dimensions supérieures, plus de travail serait nécessaire
//
//                 // Transposons B pour calculer dC/dA = dC @ B.T
//                 let b_transposed = b_clone.transpose(0, 1);
//                 let grad_a = match grad_output.matmul(&b_transposed) {
//                     Ok(t) => t,
//                     Err(e) => panic!("Error computing gradient for matmul: {}", e),
//                 };
//
//                 // Transposons A pour calculer dC/dB = A.T @ dC
//                 let a_transposed = a_clone.transpose(0, 1);
//                 let grad_b = match a_transposed.matmul(grad_output) {
//                     Ok(t) => t,
//                     Err(e) => panic!("Error computing gradient for matmul: {}", e),
//                 };
//
//                 vec![grad_a, grad_b]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::MatMul,
//             vec![self.clone(), other.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Exponentielle d'une variable
//     pub fn exp(&self) -> Self {
//         // Opération sur le tenseur sous-jacent
//         let result_tensor = self.tensor.clone().exp();
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour exp: d(exp(x))/dx = exp(x)
//         let self_clone = self.clone();
//         let grad_fn = if self.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 // Le gradient est grad_output * exp(x)
//                 let exp_x = match self_clone.tensor.exp() {
//                     Ok(t) => t,
//                     Err(e) => panic!("Error computing gradient for exp: {}", e),
//                 };
//                 let grad = grad_output.clone().mul(exp_x);
//                 vec![grad]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Exp,
//             vec![self.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Logarithme naturel d'une variable
//     pub fn log(&self) -> Self {
//         // Opération sur le tenseur sous-jacent
//         let result_tensor = match self.tensor.log() {
//             Ok(t) => t,
//             Err(e) => panic!("Error in log: {}", e),
//         };
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour log: d(log(x))/dx = 1/x
//         let self_clone = self.clone();
//         let grad_fn = if self.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 // Le gradient est grad_output / x
//                 let one = Tensor::ones(vec![1], None);
//                 let x_inv = match one.div(self_clone.tensor.clone()) {
//                     Ok(t) => t,
//                     Err(e) => panic!("Error computing gradient for log: {}", e),
//                 };
//                 let grad = grad_output.clone().mul(x_inv);
//                 vec![grad]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Log,
//             vec![self.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Sinus d'une variable
//     pub fn sin(&self) -> Self {
//         // Opération sur le tenseur sous-jacent
//         let result_tensor = match self.tensor.sin() {
//             Ok(t) => t,
//             Err(e) => panic!("Error in sin: {}", e),
//         };
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour sin: d(sin(x))/dx = cos(x)
//         let self_clone = self.clone();
//         let grad_fn = if self.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 // Le gradient est grad_output * cos(x)
//                 let cos_x = match self_clone.tensor.cos() {
//                     Ok(t) => t,
//                     Err(e) => panic!("Error computing gradient for sin: {}", e),
//                 };
//                 let grad = grad_output.clone().mul(cos_x);
//                 vec![grad]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Sin,
//             vec![self.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Cosinus d'une variable
//     pub fn cos(&self) -> Self {
//         // Opération sur le tenseur sous-jacent
//         let result_tensor = match self.tensor.cos() {
//             Ok(t) => t,
//             Err(e) => panic!("Error in cos: {}", e),
//         };
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour cos: d(cos(x))/dx = -sin(x)
//         let self_clone = self.clone();
//         let grad_fn = if self.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 // Le gradient est grad_output * (-sin(x))
//                 let sin_x = match self_clone.tensor.sin() {
//                     Ok(t) => t,
//                     Err(e) => panic!("Error computing gradient for cos: {}", e),
//                 };
//                 let minus_one = Tensor::from_data(&[-1.0], vec![1], None);
//                 let neg_sin_x = sin_x.mul(minus_one);
//                 let grad = grad_output.clone().mul(neg_sin_x);
//                 vec![grad]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Cos,
//             vec![self.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Tangente d'une variable
//     pub fn tan(&self) -> Self {
//         // Opération sur le tenseur sous-jacent
//         let result_tensor = match self.tensor.tan() {
//             Ok(t) => t,
//             Err(e) => panic!("Error in tan: {}", e),
//         };
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour tan: d(tan(x))/dx = 1 / (cos(x))^2 = 1 + tan(x)^2
//         let self_clone = self.clone();
//         let result_clone = result_tensor.clone();
//         let grad_fn = if self.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 // Le gradient est grad_output * (1 + tan(x)^2)
//                 let tan_squared = result_clone.clone().mul(result_clone.clone());
//                 let one = Tensor::ones(self_clone.tensor.shape().to_vec(), None);
//                 let derivative = one.add(tan_squared);
//                 let grad = grad_output.clone().mul(derivative);
//                 vec![grad]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Tan,
//             vec![self.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Puissance d'une variable: x^y où y est un scalaire
//     pub fn pow(&self, exponent: f64) -> Self {
//         // Opération sur le tenseur sous-jacent
//         let result_tensor = match self.tensor.pow(exponent) {
//             Ok(t) => t,
//             Err(e) => panic!("Error in pow: {}", e),
//         };
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Fonction de gradient pour pow: d(x^y)/dx = y * x^(y-1)
//         let self_clone = self.clone();
//         let exp_minus_one = exponent - 1.0;
//         let exp_value = exponent;
//
//         let grad_fn = if self.requires_grad {
//             Some(Box::new(move |grad_output: &Tensor| {
//                 // Le gradient est grad_output * y * x^(y-1)
//                 let x_pow_y_minus_1 = match self_clone.tensor.pow(exp_minus_one) {
//                     Ok(t) => t,
//                     Err(e) => panic!("Error computing gradient for pow: {}", e),
//                 };
//
//                 let y_tensor = Tensor::from_data(&[exp_value], vec![1], None);
//                 let derivative = x_pow_y_minus_1.mul(y_tensor);
//                 let grad = grad_output.clone().mul(derivative);
//
//                 vec![grad]
//             }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
//         } else {
//             None
//         };
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Pow,
//             vec![self.clone()],
//             grad_fn,
//         )
//     }
//
//     /// Calcule la somme de tous les éléments du tenseur
//     pub fn sum(&self) -> Self {
//         let result_tensor = self.tensor.sum();
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Pour la rétropropagation, le gradient de sum par rapport à chaque élément est 1
//         let self_clone = self.clone();
//         let grad_fn = Box::new(move |_grad_output: &Tensor| {
//             // Pour sum(), le gradient par rapport à chaque élément de l'entrée est 1
//             let ones = Tensor::ones(self_clone.tensor.shape().to_vec(), None);
//             vec![ones]
//         }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Sum,  // Utilisez l'opération Sum au lieu de None
//             vec![self.clone()],
//             Some(grad_fn),
//         )
//     }
//
//     /// Calcule la moyenne de tous les éléments du tenseur
//     pub fn mean(&self) -> Self {
//         let result_tensor = match self.tensor.mean() {
//             Ok(t) => t,
//             Err(e) => panic!("Error in mean: {}", e),
//         };
//
//         // Si le calcul du gradient est désactivé, retourner un résultat simple
//         if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
//             return Self::from_tensor(result_tensor, false);
//         }
//
//         // Pour la rétropropagation, le gradient de mean par rapport à chaque élément est 1/n
//         let self_clone = self.clone();
//         let grad_fn = Box::new(move |grad_output: &Tensor| {
//             // Pour mean(), le gradient par rapport à chaque élément de l'entrée est 1/n
//             let n = self_clone.tensor.numel() as f64;
//             let scale = 1.0 / n;
//             let scale_tensor = Tensor::from_data(&[scale], vec![1], None);
//
//             // Multiplier le gradient de sortie par 1/n et le diffuser à tous les éléments
//             let ones = Tensor::ones(self_clone.tensor.shape().to_vec(), None);
//             let scaled_ones = ones.mul(scale_tensor);
//             let grad = grad_output.clone().mul(scaled_ones);
//
//             vec![grad]
//         }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
//
//         // Créer la variable résultante
//         Self::from_operation(
//             result_tensor,
//             Operation::Mean,
//             vec![self.clone()],
//             Some(grad_fn),
//         )
//     }
//
//     /// Calcule le gradient avec options avancées
//     pub fn backward_with_options(&mut self, retain_graph: bool, create_graph: bool) {
//         if !self.requires_grad {
//             return;
//         }
//
//         // Détection de cycles dans le graphe
//         let mut visited: HashSet<usize> = HashSet::new();
//         let mut in_progress: HashSet<usize> = HashSet::new();
//
//         // Fonction récursive pour détecter les cycles
//         fn detect_cycle(var: &Variable, visited: &mut HashSet<usize>, in_progress: &mut HashSet<usize>) -> bool {
//             if in_progress.contains(&var.id) {
//                 return true; // Cycle détecté
//             }
//
//             if visited.contains(&var.id) {
//                 return false; // Déjà visité, pas de cycle
//             }
//
//             in_progress.insert(var.id);
//
//             if let Some(ref node) = var.grad_fn {
//                 for input_var in &node.inputs {
//                     if detect_cycle(input_var, visited, in_progress) {
//                         return true;
//                     }
//                 }
//             }
//
//             in_progress.remove(&var.id);
//             visited.insert(var.id);
//
//             false
//         }
//
//         // Vérifier les cycles avant la rétropropagation
//         if detect_cycle(self, &mut visited, &mut in_progress) {
//             panic!("Cycle detected in computation graph!");
//         }
//
//         // Structure pour suivre les gradients accumulés
//         let mut grad_table: HashMap<usize, Tensor> = HashMap::new();
//
//         // File d'attente pour la propagation du gradient
//         let mut queue: Vec<(Arc<Node>, Tensor)> = Vec::new();
//
//         // Initialiser le gradient de sortie à 1 s'il n'est pas défini
//         if self.grad.is_none() {
//             self.grad = Some(Tensor::ones(self.tensor.shape().to_vec(), None));
//         }
//
//         // Si cette variable a une fonction de gradient, l'ajouter à la file d'attente
//         if let Some(ref grad_fn) = self.grad_fn {
//             queue.push((grad_fn.clone(), self.grad.clone().unwrap()));
//         } else if self.is_leaf {
//             // Pour les feuilles, stocker le gradient directement
//             grad_table.insert(self.id, self.grad.clone().unwrap());
//         }
//
//         // Propager les gradients à travers le graphe
//         // Propager les gradients à travers le graphe
//         while let Some((node, grad_output)) = queue.pop() {
//             if let Some(ref grad_fn) = node.grad_fn {
//                 let input_grads = grad_fn(&grad_output);
//
//                 assert_eq!(input_grads.len(), node.inputs.len(),
//                            "Number of gradients doesn't match number of inputs");
//
//                 for (input_var, input_grad) in node.inputs.iter().zip(input_grads.iter()) {
//                     if !input_var.requires_grad {
//                         continue;
//                     }
//
//                     // Utiliser l'ID plutôt que l'adresse mémoire
//                     if let Some(existing_grad) = grad_table.get(&input_var.id) {
//                         let new_grad = existing_grad.clone().add(input_grad.clone());
//                         grad_table.insert(input_var.id, new_grad);
//                     } else {
//                         grad_table.insert(input_var.id, input_grad.clone());
//                     }
//
//                     if let Some(ref input_grad_fn) = input_var.grad_fn {
//                         queue.push((input_grad_fn.clone(), input_grad.clone()));
//                     }
//                 }
//             }
//         }
//
//         // Mettre à jour les gradients des variables feuilles
//         for (var_id, grad) in grad_table {
//             let grad_var = if create_graph {
//                 // Créer une variable à partir du gradient avec suivi de gradient
//                 let mut new_var = Variable::from_tensor(grad.clone(), true);
//                 new_var.is_gradient = true;
//                 new_var
//             } else {
//                 Variable::from_tensor(grad.clone(), false)
//             };
//
//             VARIABLES.with(|vars| {
//                 if let Some((var_grad, timestamp)) = vars.borrow_mut().get_mut(&var_id) {
//                     *timestamp = Instant::now();
//                     *var_grad.borrow_mut() = Some(grad.clone());
//                 }
//             });
//
//             // Mise à jour du gradient dans cette variable si nécessaire
//             if var_id == self.id {
//                 self.grad = Some(grad);
//             }
//         }
//
//         // Si on ne veut pas conserver le graphe, supprimer les références aux nœuds
//         if !retain_graph {
//             self.grad_fn = None;
//         }
//     }
//
//     // Méthode standard backward sans options
//     pub fn backward(&mut self) {
//         self.backward_with_options(false, false);
//     }
//
//     // Méthode pour obtenir le gradient
//     pub fn grad(&self) -> Option<Tensor> {
//         let ptr = self.id;
//
//         let result = if self.is_leaf && self.requires_grad {
//             VARIABLES.with(|vars| {
//                 if let Some((var_grad, timestamp)) = vars.borrow_mut().get_mut(&ptr) {
//                     *timestamp = Instant::now();
//                     var_grad.borrow().clone()
//                 } else {
//                     None
//                 }
//             })
//         } else {
//             self.grad.clone()
//         };
//
//         // Vérifier si un nettoyage est nécessaire
//         maybe_cleanup();
//
//         result
//     }
//
//     /// Fonction qui visualise le graphe de calcul à partir de cette variable
//     pub fn visualize_graph(&self, filename: &str) -> Result<(), Box<dyn Error>> {
//         // Cette fonction pourrait construire une représentation DOT du graphe
//         // et l'enregistrer dans un fichier pour visualisation avec Graphviz
//
//         let mut dot_content = String::from("digraph ComputationGraph {\n");
//         dot_content.push_str("  rankdir=LR;\n");
//         dot_content.push_str("  node [shape=box, style=filled, color=lightblue];\n\n");
//
//         // Ensembles pour suivre les nœuds et arêtes déjà visités
//         let mut visited_nodes = HashSet::new();
//         let mut edges = HashSet::new();
//
//         // Fonction récursive pour construire le graphe DOT
//         fn build_graph(
//             var: &Variable,
//             dot_content: &mut String,
//             visited: &mut HashSet<usize>,
//             edges: &mut HashSet<(usize, usize)>
//         ) {
//             // Si ce nœud a déjà été visité, on s'arrête
//             if !visited.insert(var.id) {
//                 return;
//             }
//
//             // Ajouter ce nœud au graphe
//             let label = if var.is_leaf {
//                 format!("{}\\nLeaf: {}\\nRequires grad: {}",
//                         var.id, var.is_leaf, var.requires_grad)
//             } else if let Some(ref node) = var.grad_fn {
//                 format!("{}\\nOp: {}\\nRequires grad: {}",
//                         var.id, node.operation, var.requires_grad)
//             } else {
//                 format!("{}\\nRequires grad: {}", var.id, var.requires_grad)
//             };
//
//             let color = if var.is_leaf {
//                 "lightgreen"
//             } else if var.requires_grad {
//                 "lightblue"
//             } else {
//                 "lightgray"
//             };
//
//             dot_content.push_str(&format!("  node{} [label=\"{}\", fillcolor=\"{}\"];\n",
//                                           var.id, label, color));
//
//             // Ajouter les arêtes pour les entrées
//             if let Some(ref node) = var.grad_fn {
//                 for input in &node.inputs {
//                     if edges.insert((input.id, var.id)) {
//                         dot_content.push_str(&format!("  node{} -> node{};\n",
//                                                       input.id, var.id));
//                     }
//                     build_graph(input, dot_content, visited, edges);
//                 }
//             }
//         }
//
//         // Construire le graphe en partant de cette variable
//         build_graph(self, &mut dot_content, &mut visited_nodes, &mut edges);
//
//         // Finaliser le contenu DOT
//         dot_content.push_str("}\n");
//
//         // Écrire dans un fichier
//         let mut file = File::create(filename)?;
//         file.write_all(dot_content.as_bytes())?;
//
//         // On pourrait également lancer automatiquement la commande dot pour générer une image
//         // si Graphviz est installé
//         println!("Graph saved to {}. Use Graphviz to visualize it: dot -Tpng {} -o {}.png",
//                  filename, filename, filename.trim_end_matches(".dot"));
//
//         Ok(())
//     }
//
//     /// Nettoyer les variables inutilisées du registre global
//     pub fn cleanup_variables(max_age_seconds: u64) {
//         const DEFAULT_MAX_AGE: Duration = Duration::from_secs(600); // 10 minutes
//
//         let max_age = if max_age_seconds > 0 {
//             Duration::from_secs(max_age_seconds)
//         } else {
//             DEFAULT_MAX_AGE
//         };
//
//         let now = Instant::now();
//
//         // Nettoyer les variables anciennes
//         VARIABLES.with(|vars| {
//             let mut to_remove = Vec::new();
//
//             for (&id, (_, timestamp)) in vars.borrow().iter() {
//                 if now.duration_since(*timestamp) > max_age {
//                     to_remove.push(id);
//                 }
//             }
//
//             let mut vars_mut = vars.borrow_mut();
//             for id in to_remove {
//                 vars_mut.remove(&id);
//             }
//
//             println!("Cleaned up {} variables. {} variables remaining.",
//                      to_remove.len(), vars_mut.len());
//         });
//     }
//
//     /// Retourne la représentation textuelle du graphe de calcul
//     pub fn print_graph_structure(&self) -> String {
//         let mut result = String::new();
//         let mut visited = HashSet::new();
//
//         fn print_node(
//             var: &Variable,
//             depth: usize,
//             result: &mut String,
//             visited: &mut HashSet<usize>
//         ) {
//             // Éviter les cycles
//             if !visited.insert(var.id) {
//                 let indent = "  ".repeat(depth);
//                 result.push_str(&format!("{}Node {} (already visited)\n", indent, var.id));
//                 return;
//             }
//
//             let indent = "  ".repeat(depth);
//
//             if var.is_leaf {
//                 result.push_str(&format!("{}Node {} (Leaf, requires_grad={})\n",
//                                          indent, var.id, var.requires_grad));
//             } else if let Some(ref node) = var.grad_fn {
//                 result.push_str(&format!("{}Node {} (Op: {}, requires_grad={})\n",
//                                          indent, var.id, node.operation, var.requires_grad));
//
//                 // Afficher les nœuds d'entrée
//                 for (i, input) in node.inputs.iter().enumerate() {
//                     result.push_str(&format!("{}  Input {}:\n", indent, i));
//                     print_node(input, depth + 2, result, visited);
//                 }
//             } else {
//                 result.push_str(&format!("{}Node {} (No grad_fn, requires_grad={})\n",
//                                          indent, var.id, var.requires_grad));
//             }
//         }
//
//         result.push_str("Computation Graph Structure:\n");
//         print_node(self, 0, &mut result, &mut visited);
//
//         result
//     }
// }
//
// // Context pour désactiver temporairement le calcul du gradient
// pub struct NoGradGuard {
//     prev_enabled: bool,
// }
//
// /// Implémentation de NoGradGuard pour désactiver le calcul du gradient
// impl NoGradGuard {
//     pub fn new() -> Self {
//         let prev = GRAD_ENABLED.with(|cell| *cell.borrow());
//         GRAD_ENABLED.with(|cell| *cell.borrow_mut() = false);
//         Self { prev_enabled: prev }
//     }
// }
//
// /// Implémentation de Drop pour restaurer l'état précédent
// impl Drop for NoGradGuard {
//     fn drop(&mut self) {
//         GRAD_ENABLED.with(|cell| *cell.borrow_mut() = self.prev_enabled);
//     }
// }
//
// /// Fonction utilitaire pour créer un guard qui désactive le calcul de gradient
// pub fn no_grad() -> NoGradGuard {
//     NoGradGuard::new()
// }
//
//
// // Tests pour l'autograd
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_variable_creation() {
//         let tensor = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
//         let var = Variable::from_tensor(tensor, true);
//
//         assert!(var.requires_grad);
//         assert!(var.is_leaf);
//         assert!(var.grad.is_none());
//         assert!(var.grad_fn.is_none());
//     }
//
//     #[test]
//     fn test_add_operation() {
//         let tensor_a = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], None);
//         let tensor_b = Tensor::from_data(&[4.0, 5.0, 6.0], vec![3], None);
//
//         let var_a = Variable::from_tensor(tensor_a, true);
//         let var_b = Variable::from_tensor(tensor_b, true);
//
//         let var_c = var_a.add(&var_b);
//
//         assert!(var_c.requires_grad);
//         assert!(!var_c.is_leaf);
//         assert!(var_c.grad.is_none());
//         assert!(var_c.grad_fn.is_some());
//
//         // Vérifier que le tenseur résultant contient les bonnes valeurs
//         match var_c.tensor.storage().as_ref() {
//             rustytorch_tensor::storage::StorageType::F32(data) => {
//                 assert_eq!(data, &[5.0, 7.0, 9.0]);
//             },
//             rustytorch_tensor::storage::StorageType::F64(data) => {
//                 assert_eq!(data, &[5.0, 7.0, 9.0]);
//             },
//             _ => panic!("Unexpected storage type"),
//         }
//     }
//
//     #[test]
//     fn test_backward_simple() {
//         // Créer deux variables
//         let tensor_a = Tensor::from_data(&[2.0], vec![1], None);
//         let tensor_b = Tensor::from_data(&[3.0], vec![1], None);
//
//         let mut var_a = Variable::from_tensor(tensor_a, true);
//         let mut var_b = Variable::from_tensor(tensor_b, true);
//
//         // Calculer c = a * b
//         let mut var_c = var_a.mul(&var_b);
//
//         // Propagation arrière
//         var_c.backward();
//
//         // Vérifier les gradients:
//         // dc/da = b = 3
//         // dc/db = a = 2
//         if let Some(grad_a) = &var_a.grad {
//             match grad_a.storage().as_ref() {
//                 rustytorch_tensor::storage::StorageType::F32(data) => {
//                     assert_eq!(data[0], 3.0);
//                 },
//                 rustytorch_tensor::storage::StorageType::F64(data) => {
//                     assert_eq!(data[0], 3.0);
//                 },
//                 _ => panic!("Unexpected storage type"),
//             }
//         } else {
//             panic!("Gradient for var_a is None");
//         }
//
//         if let Some(grad_b) = &var_b.grad {
//             match grad_b.storage().as_ref() {
//                 rustytorch_tensor::storage::StorageType::F32(data) => {
//                     assert_eq!(data[0], 2.0);
//                 },
//                 rustytorch_tensor::storage::StorageType::F64(data) => {
//                     assert_eq!(data[0], 2.0);
//                 },
//                 _ => panic!("Unexpected storage type"),
//             }
//         } else {
//             panic!("Gradient for var_b is None");
//         }
//     }
//
//     #[test]
//     fn test_no_grad() {
//         // Créer deux variables
//         let tensor_a = Tensor::from_data(&[2.0], vec![1], None);
//         let tensor_b = Tensor::from_data(&[3.0], vec![1], None);
//
//         // Avec no_grad, les opérations ne devraient pas créer de graphe de calcul
//         {
//             let _guard = no_grad();
//
//             let var_a = Variable::from_tensor(tensor_a.clone(), true);
//             let var_b = Variable::from_tensor(tensor_b.clone(), true);
//
//             let var_c = var_a.add(&var_b);
//
//             // Même si requires_grad est vrai pour les entrées, il devrait être faux pour le résultat
//             assert!(!var_c.requires_grad);
//             assert!(var_c.grad_fn.is_none());
//         }
//     }
//
//     #[test]
//     fn test_complex_graph() {
//         // Créer des variables pour un exemple plus complexe
//         // Exemple: f(x, y) = (x + 2*y) * (x^2)
//         let tensor_x = Tensor::from_data(&[3.0], vec![1], None);
//         let tensor_y = Tensor::from_data(&[4.0], vec![1], None);
//
//         let var_x = Variable::from_tensor(tensor_x, true);
//         let var_y = Variable::from_tensor(tensor_y, true);
//
//         // Calculer 2*y
//         let two = Variable::from_tensor(Tensor::from_data(&[2.0], vec![1], None), false);
//         let two_y = two.mul(&var_y);
//
//         // Calculer x + 2*y
//         let x_plus_2y = var_x.add(&two_y);
//
//         // Calculer x^2
//         let x_squared = var_x.mul(&var_x);
//
//         // Calculer (x + 2*y) * (x^2)
//         let mut result = x_plus_2y.mul(&x_squared);
//
//         // Propager les gradients
//         result.backward();
//
//         // Les gradients devraient être:
//         // df/dx = d/dx[(x + 2*y) * (x^2)]
//         //       = (x^2) * d/dx(x + 2*y) + (x + 2*y) * d/dx(x^2)
//         //       = (x^2) * 1 + (x + 2*y) * 2*x
//         //       = x^2 + 2*x*(x + 2*y)
//         // Pour x=3, y=4: df/dx = 3^2 + 2*3*(3 + 2*4) = 9 + 6*11 = 9 + 66 = 75
//         //
//         // df/dy = d/dy[(x + 2*y) * (x^2)]
//         //       = (x^2) * d/dy(x + 2*y) + (x + 2*y) * d/dy(x^2)
//         //       = (x^2) * 2 + (x + 2*y) * 0
//         //       = 2*x^2
//         // Pour x=3, y=4: df/dy = 2*3^2 = 2*9 = 18
//
//         // TODO: Vérifier les gradients calculés
//         // Cette vérification devrait être activée quand l'implémentation complète de backward sera terminée
//     }
// }