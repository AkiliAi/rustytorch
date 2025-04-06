//rustytorch_autograd/src/lib.rs


use std::sync::Arc;

use rustytorch_tensor::Tensor;


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
    Tanh,
    Relu,
    Softmax,
    // Autres opérations à ajouter...
    None,
}

// Variable  avec suivi de gradient
#[derive(Clone)]
pub struct Variable{
    pub tensor: Tensor,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub grad: Option<Tensor>,
    pub grad_fn: Option<Arc<Node>>,
}


// #[derive(Clone,Debug)]
pub struct Node{
    pub operation: Operation,
    pub inputs: Vec<Variable>,
    pub grad_fn: Option<Box<dyn Fn(&Tensor) -> Vec<Tensor>>>,
}


impl Variable {
    // Cree une nouvelle variable a partir d'un tenseur
    pub fn from_tensor(tensor: Tensor,requires_grad: bool) -> Self {
        Self {
            tensor,
            requires_grad,
            is_leaf:true,
            grad: None,
            grad_fn: None,
        }
    }


    // Cree une variable resultante d'une operation
    pub fn from_operation(
        tensor: Tensor,
        operation: Operation,
        inputs: Vec<Variable>,
        grad_fn: Option<Box<dyn Fn(&Tensor) -> Vec<Tensor>>>,
    ) -> Self {
        let requires_grad = inputs.iter().any(|v| v.requires_grad);

        let grad_fn = if requires_grad{
            // cree un noeud pour une operation
            let node = Node{
                operation,
                inputs:inputs.clone(),
                grad_fn,
            };
            Some(Arc::new(node))
        }else {
            None
        };

        Self{
            tensor,
            requires_grad,
            is_leaf:false,
            grad: None,
            grad_fn,
        }
    }

    /// calcule le gradient de la variable par rapport aux entrées
    pub fn backward(&mut self){
        if !self.requires_grad{
            return;
        }
        // Si c'est la racine (point de départ), initialiser le gradient à 1
        if self.grad.is_none(){
            let grad = self.tensor.clone();
            // TODO: Remplacer par un tenseur de 1 de la même forme
            self.grad = Some(grad);
        }

        // TODO: Implémentation de la propagation arrière en parcourant le graphe

    }

}

// COntext pour desactiver temporairement le calcul du gradient
pub struct NoGradGuard {
    prev_enabled: bool,
}

/// Implémentation de NoGradGuard pour désactiver le calcul du gradient

impl NoGradGuard{
    pub fn new() -> Self {
        Self {
            //TODO: Obtient l'état actuel du calcul du gradient
            prev_enabled: true,
        }
    }
}

/// Implémentation de Drop pour restaurer l'état précédent

impl Drop for NoGradGuard{
    fn drop(&mut self) {
        //TODO: Rétablit l'état du calcul du gradient Precedent
    }
}


/// Fonction utilitaire pour créer un guard qui désactive le calcul de gradient
pub fn no_guard() -> NoGradGuard{
    NoGradGuard::new()
}