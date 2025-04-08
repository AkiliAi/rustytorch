//rustytorch_autograd/src/operations.rs

use std::collections::HashSet;
use std::time::{Duration, Instant};
use rustytorch_core::{NumericOps, Reduction};
use rustytorch_tensor::Tensor;
use crate::{GRAD_ENABLED, Operation, Variable, VARIABLES};
impl Variable {


    // /// Exponentielle d'une variable
    // pub fn exp(&self) -> Self {
    //     // Opération sur le tenseur sous-jacent
    //     let result_tensor = match self.tensor.exp() {
    //         Ok(t) => t,
    //         Err(e) => panic!("Error in exp: {}", e),
    //     };
    //
    //     // Si le calcul du gradient est désactivé, retourner un résultat simple
    //     if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
    //         return Self::from_tensor(result_tensor, false);
    //     }
    //
    //     // Fonction de gradient pour exp: d(exp(x))/dx = exp(x)
    //     let self_clone = self.clone();
    //     let grad_fn = if self.requires_grad {
    //         Some(Box::new(move |grad_output: &Tensor| {
    //             // Le gradient est grad_output * exp(x)
    //             let exp_x = match self_clone.tensor.exp() {
    //                 Ok(t) => t,
    //                 Err(e) => panic!("Error computing gradient for exp: {}", e),
    //             };
    //             let grad = grad_output.clone().mul(exp_x);
    //             vec![grad]
    //         }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
    //     } else {
    //         None
    //     };
    //
    //     // Créer la variable résultante
    //     Self::from_operation(
    //         result_tensor,
    //         Operation::Exp,
    //         vec![self.clone()],
    //         grad_fn,
    //     )
    // }
    //
    // /// Logarithme naturel d'une variable
    // pub fn log(&self) -> Self {
    //     // Opération sur le tenseur sous-jacent
    //     let result_tensor = match self.tensor.log() {
    //         Ok(t) => t,
    //         Err(e) => panic!("Error in log: {}", e),
    //     };
    //
    //     // Si le calcul du gradient est désactivé, retourner un résultat simple
    //     if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
    //         return Self::from_tensor(result_tensor, false);
    //     }
    //
    //     // Fonction de gradient pour log: d(log(x))/dx = 1/x
    //     let self_clone = self.clone();
    //     let grad_fn = if self.requires_grad {
    //         Some(Box::new(move |grad_output: &Tensor| {
    //             // Le gradient est grad_output / x
    //             let one = Tensor::ones(vec![1], None);
    //             let x_inv = match one.div(self_clone.tensor.clone()) {
    //                 Ok(t) => t,
    //                 Err(e) => panic!("Error computing gradient for log: {}", e),
    //             };
    //             let grad = grad_output.clone().mul(x_inv);
    //             vec![grad]
    //         }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
    //     } else {
    //         None
    //     };
    //
    //     // Créer la variable résultante
    //     Self::from_operation(
    //         result_tensor,
    //         Operation::Log,
    //         vec![self.clone()],
    //         grad_fn,
    //     )
    // }
    //
    // /// Sinus d'une variable
    // pub fn sin(&self) -> Self {
    //     // Opération sur le tenseur sous-jacent
    //     let result_tensor = match self.tensor.sin() {
    //         Ok(t) => t,
    //         Err(e) => panic!("Error in sin: {}", e),
    //     };
    //
    //     // Si le calcul du gradient est désactivé, retourner un résultat simple
    //     if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
    //         return Self::from_tensor(result_tensor, false);
    //     }
    //
    //     // Fonction de gradient pour sin: d(sin(x))/dx = cos(x)
    //     let self_clone = self.clone();
    //     let grad_fn = if self.requires_grad {
    //         Some(Box::new(move |grad_output: &Tensor| {
    //             // Le gradient est grad_output * cos(x)
    //             let cos_x = match self_clone.tensor.cos() {
    //                 Ok(t) => t,
    //                 Err(e) => panic!("Error computing gradient for sin: {}", e),
    //             };
    //             let grad = grad_output.clone().mul(cos_x);
    //             vec![grad]
    //         }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
    //     } else {
    //         None
    //     };
    //
    //     // Créer la variable résultante
    //     Self::from_operation(
    //         result_tensor,
    //         Operation::Sin,
    //         vec![self.clone()],
    //         grad_fn,
    //     )
    // }
    //
    // /// Cosinus d'une variable
    // pub fn cos(&self) -> Self {
    //     // Opération sur le tenseur sous-jacent
    //     let result_tensor = match self.tensor.cos() {
    //         Ok(t) => t,
    //         Err(e) => panic!("Error in cos: {}", e),
    //     };
    //
    //     // Si le calcul du gradient est désactivé, retourner un résultat simple
    //     if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
    //         return Self::from_tensor(result_tensor, false);
    //     }
    //
    //     // Fonction de gradient pour cos: d(cos(x))/dx = -sin(x)
    //     let self_clone = self.clone();
    //     let grad_fn = if self.requires_grad {
    //         Some(Box::new(move |grad_output: &Tensor| {
    //             // Le gradient est grad_output * (-sin(x))
    //             let sin_x = match self_clone.tensor.sin() {
    //                 Ok(t) => t,
    //                 Err(e) => panic!("Error computing gradient for cos: {}", e),
    //             };
    //             let minus_one = Tensor::from_data(&[-1.0], vec![1], None);
    //             let neg_sin_x = sin_x.mul(minus_one);
    //             let grad = grad_output.clone().mul(neg_sin_x);
    //             vec![grad]
    //         }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
    //     } else {
    //         None
    //     };
    //
    //     // Créer la variable résultante
    //     Self::from_operation(
    //         result_tensor,
    //         Operation::Cos,
    //         vec![self.clone()],
    //         grad_fn,
    //     )
    // }
    //
    // /// Tangente d'une variable
    // pub fn tan(&self) -> Self {
    //     // Opération sur le tenseur sous-jacent
    //     let result_tensor = match self.tensor.tan() {
    //         Ok(t) => t,
    //         Err(e) => panic!("Error in tan: {}", e),
    //     };
    //
    //     // Si le calcul du gradient est désactivé, retourner un résultat simple
    //     if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
    //         return Self::from_tensor(result_tensor, false);
    //     }
    //
    //     // Fonction de gradient pour tan: d(tan(x))/dx = 1 / (cos(x))^2 = 1 + tan(x)^2
    //     let self_clone = self.clone();
    //     let result_clone = result_tensor.clone();
    //     let grad_fn = if self.requires_grad {
    //         Some(Box::new(move |grad_output: &Tensor| {
    //             // Le gradient est grad_output * (1 + tan(x)^2)
    //             let tan_squared = result_clone.clone().mul(result_clone.clone());
    //             let one = Tensor::ones(self_clone.tensor.shape().to_vec(), None);
    //             let derivative = one.add(tan_squared);
    //             let grad = grad_output.clone().mul(derivative);
    //             vec![grad]
    //         }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
    //     } else {
    //         None
    //     };
    //
    //     // Créer la variable résultante
    //     Self::from_operation(
    //         result_tensor,
    //         Operation::Tan,
    //         vec![self.clone()],
    //         grad_fn,
    //     )
    // }
    //
    // /// Puissance d'une variable: x^y où y est un scalaire
    // pub fn pow(&self, exponent: f64) -> Self {
    //     // Opération sur le tenseur sous-jacent
    //     let result_tensor = match self.tensor.pow(exponent) {
    //         Ok(t) => t,
    //         Err(e) => panic!("Error in pow: {}", e),
    //     };
    //
    //     // Si le calcul du gradient est désactivé, retourner un résultat simple
    //     if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
    //         return Self::from_tensor(result_tensor, false);
    //     }
    //
    //     // Fonction de gradient pour pow: d(x^y)/dx = y * x^(y-1)
    //     let self_clone = self.clone();
    //     let exp_minus_one = exponent - 1.0;
    //     let exp_value = exponent;
    //
    //     let grad_fn = if self.requires_grad {
    //         Some(Box::new(move |grad_output: &Tensor| {
    //             // Le gradient est grad_output * y * x^(y-1)
    //             let x_pow_y_minus_1 = match self_clone.tensor.pow(exp_minus_one) {
    //                 Ok(t) => t,
    //                 Err(e) => panic!("Error computing gradient for pow: {}", e),
    //             };
    //
    //             let y_tensor = Tensor::from_data(&[exp_value], vec![1], None);
    //             let derivative = x_pow_y_minus_1.mul(y_tensor);
    //             let grad = grad_output.clone().mul(derivative);
    //
    //             vec![grad]
    //         }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
    //     } else {
    //         None
    //     };
    //
    //     // Créer la variable résultante
    //     Self::from_operation(
    //         result_tensor,
    //         Operation::Pow,
    //         vec![self.clone()],
    //         grad_fn,
    //     )
    // }
    //
    // /// Calcule la somme de tous les éléments du tenseur
    // pub fn sum(&self) -> Self {
    //     let result_tensor = self.tensor.sum();
    //
    //     // Si le calcul du gradient est désactivé, retourner un résultat simple
    //     if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
    //         return Self::from_tensor(result_tensor, false);
    //     }
    //
    //     // Pour la rétropropagation, le gradient de sum par rapport à chaque élément est 1
    //     let self_clone = self.clone();
    //     let grad_fn = Box::new(move |_grad_output: &Tensor| {
    //         // Pour sum(), le gradient par rapport à chaque élément de l'entrée est 1
    //         let ones = Tensor::ones(self_clone.tensor.shape().to_vec(), None);
    //         vec![ones]
    //     }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
    //
    //     // Créer la variable résultante
    //     Self::from_operation(
    //         result_tensor,
    //         Operation::Sum,  // Utilisez l'opération Sum au lieu de None
    //         vec![self.clone()],
    //         Some(grad_fn),
    //     )
    // }
    //
    // /// Calcule la moyenne de tous les éléments du tenseur
    // pub fn mean(&self) -> Self {
    //     let result_tensor = match self.tensor.mean() {
    //         Ok(t) => t,
    //         Err(e) => panic!("Error in mean: {}", e),
    //     };
    //
    //     // Si le calcul du gradient est désactivé, retourner un résultat simple
    //     if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
    //         return Self::from_tensor(result_tensor, false);
    //     }
    //
    //     // Pour la rétropropagation, le gradient de mean par rapport à chaque élément est 1/n
    //     let self_clone = self.clone();
    //     let grad_fn = Box::new(move |grad_output: &Tensor| {
    //         // Pour mean(), le gradient par rapport à chaque élément de l'entrée est 1/n
    //         let n = self_clone.tensor.numel() as f64;
    //         let scale = 1.0 / n;
    //         let scale_tensor = Tensor::from_data(&[scale], vec![1], None);
    //
    //         // Multiplier le gradient de sortie par 1/n et le diffuser à tous les éléments
    //         let ones = Tensor::ones(self_clone.tensor.shape().to_vec(), None);
    //         let scaled_ones = ones.mul(scale_tensor);
    //         let grad = grad_output.clone().mul(scaled_ones);
    //
    //         vec![grad]
    //     }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
    //
    //     // Créer la variable résultante
    //     Self::from_operation(
    //         result_tensor,
    //         Operation::Mean,
    //         vec![self.clone()],
    //         Some(grad_fn),
    //     )
    // }
    //
    /// Fonction qui visualise le graphe de calcul à partir de cette variable
    pub fn visualize_graph(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Cette fonction pourrait construire une représentation DOT du graphe
        // et l'enregistrer dans un fichier pour visualisation avec Graphviz

        use std::fs::File;
        use std::io::Write;

        let mut dot_content = String::from("digraph ComputationGraph {\n");
        dot_content.push_str("  rankdir=LR;\n");
        dot_content.push_str("  node [shape=box, style=filled, color=lightblue];\n\n");

        // Ensembles pour suivre les nœuds et arêtes déjà visités
        let mut visited_nodes = HashSet::new();
        let mut edges = HashSet::new();

        // Fonction récursive pour construire le graphe DOT
        fn build_graph(
            var: &Variable,
            dot_content: &mut String,
            visited: &mut HashSet<usize>,
            edges: &mut HashSet<(usize, usize)>
        ) {
            // Si ce nœud a déjà été visité, on s'arrête
            if !visited.insert(var.id) {
                return;
            }

            // Ajouter ce nœud au graphe
            let label = if var.is_leaf {
                format!("{}\\nLeaf: {}\\nRequires grad: {}",
                        var.id, var.is_leaf, var.requires_grad)
            } else if let Some(ref node) = var.grad_fn {
                format!("{}\\nOp: {}\\nRequires grad: {}",
                        var.id, node.operation, var.requires_grad)
            } else {
                format!("{}\\nRequires grad: {}", var.id, var.requires_grad)
            };

            let color = if var.is_leaf {
                "lightgreen"
            } else if var.requires_grad {
                "lightblue"
            } else {
                "lightgray"
            };

            dot_content.push_str(&format!("  node{} [label=\"{}\", fillcolor=\"{}\"];\n",
                                          var.id, label, color));

            // Ajouter les arêtes pour les entrées
            if let Some(ref node) = var.grad_fn {
                for input in &node.inputs {
                    if edges.insert((input.id, var.id)) {
                        dot_content.push_str(&format!("  node{} -> node{};\n",
                                                      input.id, var.id));
                    }
                    build_graph(input, dot_content, visited, edges);
                }
            }
        }

        // Construire le graphe en partant de cette variable
        build_graph(self, &mut dot_content, &mut visited_nodes, &mut edges);

        // Finaliser le contenu DOT
        dot_content.push_str("}\n");

        // Écrire dans un fichier
        let mut file = File::create(filename)?;
        file.write_all(dot_content.as_bytes())?;

        // On pourrait également lancer automatiquement la commande dot pour générer une image
        // si Graphviz est installé
        println!("Graph saved to {}. Use Graphviz to visualize it: dot -Tpng {} -o {}.png",
                 filename, filename, filename.trim_end_matches(".dot"));

        Ok(())
    }
    //
    // /// Nettoyer les variables inutilisées du registre global
    // pub fn cleanup_variables(max_age_seconds: u64) {
    //     const DEFAULT_MAX_AGE: Duration = Duration::from_secs(600); // 10 minutes
    //
    //     let max_age = if max_age_seconds > 0 {
    //         Duration::from_secs(max_age_seconds)
    //     } else {
    //         DEFAULT_MAX_AGE
    //     };
    //
    //     let now = Instant::now();
    //
    //     // Nettoyer les variables anciennes
    //     VARIABLES.with(|vars| {
    //         let mut to_remove = Vec::new();
    //
    //         for (&id, (_, timestamp)) in vars.borrow().iter() {
    //             if now.duration_since(*timestamp) > max_age {
    //                 to_remove.push(id);
    //             }
    //         }
    //
    //         let mut vars_mut = vars.borrow_mut();
    //         for id in to_remove {
    //             vars_mut.remove(&id);
    //         }
    //
    //         println!("Cleaned up {} variables. {} variables remaining.",
    //                  to_remove.len(), vars_mut.len());
    //     });
    // }

    /// Retourne la représentation textuelle du graphe de calcul
    pub fn print_graph_structure(&self) -> String {
        let mut result = String::new();
        let mut visited = HashSet::new();

        fn print_node(
            var: &Variable,
            depth: usize,
            result: &mut String,
            visited: &mut HashSet<usize>
        ) {
            // Éviter les cycles
            if !visited.insert(var.id) {
                let indent = "  ".repeat(depth);
                result.push_str(&format!("{}Node {} (already visited)\n", indent, var.id));
                return;
            }

            let indent = "  ".repeat(depth);

            if var.is_leaf {
                result.push_str(&format!("{}Node {} (Leaf, requires_grad={})\n",
                                         indent, var.id, var.requires_grad));
            } else if let Some(ref node) = var.grad_fn {
                result.push_str(&format!("{}Node {} (Op: {}, requires_grad={})\n",
                                         indent, var.id, node.operation, var.requires_grad));

                // Afficher les nœuds d'entrée
                for (i, input) in node.inputs.iter().enumerate() {
                    result.push_str(&format!("{}  Input {}:\n", indent, i));
                    print_node(input, depth + 2, result, visited);
                }
            } else {
                result.push_str(&format!("{}Node {} (No grad_fn, requires_grad={})\n",
                                         indent, var.id, var.requires_grad));
            }
        }

        result.push_str("Computation Graph Structure:\n");
        print_node(self, 0, &mut result, &mut visited);

        result
    }
}