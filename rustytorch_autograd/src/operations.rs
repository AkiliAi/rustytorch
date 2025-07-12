//rustytorch_autograd/src/operations.rs

use crate::{Operation, Variable, GRAD_ENABLED};
use rustytorch_core::{NumericOps, Reduction, Reshapable};
use rustytorch_tensor::Tensor;
use std::collections::HashSet;
impl Variable {
    pub fn relu(&self) -> Self {
        let result_tensor = self.tensor().relu().expect("Failed to apply ReLU");

        if !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour ReLU
        let self_clone = self.clone();
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            let grad_input = self_clone
                .tensor()
                .relu_backward(grad_output)
                .expect("Failed to compute ReLU gradient");
            vec![grad_input]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;

        Self::from_operation(
            result_tensor,
            Operation::Relu,
            vec![self.clone()],
            Some(grad_fn),
        )
    }

    /// Applique la fonction d'activation Sigmoid
    pub fn sigmoid(&self) -> Self {
        let result_tensor = self.tensor().sigmoid().expect("Failed to apply Sigmoid");

        if !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour Sigmoid
        let self_clone = self.clone();
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            let grad_input = self_clone
                .tensor()
                .sigmoid_backward(grad_output)
                .expect("Failed to compute Sigmoid gradient");
            vec![grad_input]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;

        Self::from_operation(
            result_tensor,
            Operation::Sigmoid,
            vec![self.clone()],
            Some(grad_fn),
        )
    }

    /// Exponentielle d'une variable
    pub fn exp(&self) -> Self {
        // Opération sur le tenseur sous-jacent
        let result_tensor = match self.tensor().exp() {
            Ok(t) => t,
            Err(e) => panic!("Error in exp: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour exp: d(exp(x))/dx = exp(x)
        let self_clone = self.clone();
        let grad_fn = if self.requires_grad() {
            Some(Box::new(move |grad_output: &Tensor| {
                // Le gradient est grad_output * exp(x)
                let exp_x = match self_clone.tensor().exp() {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for exp: {}", e),
                };
                let grad = match grad_output.clone().mul(exp_x) {
                    Ok(t) => t,
                    Err(e) => panic!("Error in exp gradient: {}", e),
                };
                vec![grad]
            })
                as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(result_tensor, Operation::Exp, vec![self.clone()], grad_fn)
    }

    /// Logarithme naturel d'une variable
    pub fn log(&self) -> Self {
        // Opération sur le tenseur sous-jacent
        let result_tensor = match self.tensor().log() {
            Ok(t) => t,
            Err(e) => panic!("Error in log: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour log: d(log(x))/dx = 1/x
        let self_clone = self.clone();
        let grad_fn = if self.requires_grad() {
            Some(Box::new(move |grad_output: &Tensor| {
                // Le gradient est grad_output / x
                let one = Tensor::ones(vec![1], None);
                let x_inv = match one.div(self_clone.tensor()) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for log: {}", e),
                };
                let grad = match grad_output.clone().mul(x_inv) {
                    Ok(t) => t,
                    Err(e) => panic!("Error in log gradient: {}", e),
                };
                vec![grad]
            })
                as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(result_tensor, Operation::Log, vec![self.clone()], grad_fn)
    }

    /// Sinus d'une variable
    pub fn sin(&self) -> Self {
        // Opération sur le tenseur sous-jacent
        let result_tensor = match self.tensor().sin() {
            Ok(t) => t,
            Err(e) => panic!("Error in sin: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour sin: d(sin(x))/dx = cos(x)
        let self_clone = self.clone();
        let grad_fn = if self.requires_grad() {
            Some(Box::new(move |grad_output: &Tensor| {
                // Le gradient est grad_output * cos(x)
                let cos_x = match self_clone.tensor().cos() {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for sin: {}", e),
                };
                let grad = match grad_output.clone().mul(cos_x) {
                    Ok(t) => t,
                    Err(e) => panic!("Error in sin gradient: {}", e),
                };
                vec![grad]
            })
                as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(result_tensor, Operation::Sin, vec![self.clone()], grad_fn)
    }

    /// Cosinus d'une variable
    pub fn cos(&self) -> Self {
        // Opération sur le tenseur sous-jacent
        let result_tensor = match self.tensor().cos() {
            Ok(t) => t,
            Err(e) => panic!("Error in cos: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour cos: d(cos(x))/dx = -sin(x)
        let self_clone = self.clone();
        let grad_fn = if self.requires_grad() {
            Some(Box::new(move |grad_output: &Tensor| {
                // Le gradient est grad_output * (-sin(x))
                let sin_x = match self_clone.tensor().sin() {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for cos: {}", e),
                };
                let minus_one = Tensor::from_data(&[-1.0], vec![1], None);
                let neg_sin_x = match sin_x.mul(minus_one) {
                    Ok(t) => t,
                    Err(e) => panic!("Error in cos gradient: {}", e),
                };
                let grad = match grad_output.clone().mul(neg_sin_x) {
                    Ok(t) => t,
                    Err(e) => panic!("Error in cos gradient: {}", e),
                };
                vec![grad]
            })
                as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(result_tensor, Operation::Cos, vec![self.clone()], grad_fn)
    }

    /// Tangente d'une variable
    pub fn tan(&self) -> Self {
        // Opération sur le tenseur sous-jacent
        let result_tensor = match self.tensor().tan() {
            Ok(t) => t,
            Err(e) => panic!("Error in tan: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour tan: d(tan(x))/dx = 1 / (cos(x))^2 = 1 + tan(x)^2
        let self_clone = self.clone();
        let result_clone = result_tensor.clone();
        let grad_fn = if self.requires_grad() {
            Some(Box::new(move |grad_output: &Tensor| {
                // Le gradient est grad_output * (1 + tan(x)^2)
                let tan_squared = result_clone
                    .clone()
                    .mul(result_clone.clone())
                    .expect("Failed to square tan in gradient");
                let one = Tensor::ones(self_clone.tensor().shape().to_vec(), None);
                let derivative = one
                    .add(tan_squared)
                    .expect("Failed to add one to tan squared in gradient");
                let grad = grad_output
                    .clone()
                    .mul(derivative)
                    .expect("Failed to apply gradient in tan");
                vec![grad]
            })
                as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(result_tensor, Operation::Tan, vec![self.clone()], grad_fn)
    }

    /// Puissance d'une variable: x^y où y est un scalaire
    pub fn pow(&self, exponent: f64) -> Self {
        // Créer un tenseur scalaire pour l'exposant
        let exp_tensor = Tensor::from_data(&[exponent], vec![1], None);

        // Opération sur le tenseur sous-jacent
        let result_tensor = match self.tensor().pow(exp_tensor) {
            Ok(t) => t,
            Err(e) => panic!("Error in pow: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Fonction de gradient pour pow: d(x^y)/dx = y * x^(y-1)
        let self_clone = self.clone();
        let exp_minus_one = exponent - 1.0;
        let exp_value = exponent;

        let grad_fn = if self.requires_grad() {
            Some(Box::new(move |grad_output: &Tensor| {
                // Le gradient est grad_output * y * x^(y-1)
                let exp_minus_one_tensor = Tensor::from_data(&[exp_minus_one], vec![1], None);
                let x_pow_y_minus_1 = match self_clone.tensor().pow(exp_minus_one_tensor) {
                    Ok(t) => t,
                    Err(e) => panic!("Error computing gradient for pow: {}", e),
                };

                let y_tensor = Tensor::from_data(&[exp_value], vec![1], None);
                let derivative = match x_pow_y_minus_1.mul(y_tensor) {
                    Ok(t) => t,
                    Err(e) => panic!("Error in pow gradient: {}", e),
                };
                let grad = match grad_output.clone().mul(derivative) {
                    Ok(t) => t,
                    Err(e) => panic!("Error in pow gradient: {}", e),
                };

                vec![grad]
            })
                as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        // Créer la variable résultante
        Self::from_operation(result_tensor, Operation::Pow, vec![self.clone()], grad_fn)
    }

    // sum() function moved to lib.rs with better error handling

    /// Calcule la moyenne de tous les éléments du tenseur
    pub fn mean(&self) -> Self {
        let result_tensor = match self.tensor().mean() {
            Ok(t) => t,
            Err(e) => panic!("Error in mean: {}", e),
        };

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        // Pour la rétropropagation, le gradient de mean par rapport à chaque élément est 1/n
        let self_clone = self.clone();
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            // Pour mean(), le gradient par rapport à chaque élément de l'entrée est 1/n
            let n = self_clone.tensor().numel() as f64;
            let scale = 1.0 / n;
            let scale_tensor = Tensor::from_data(&[scale], vec![1], None);

            // Multiplier le gradient de sortie par 1/n et le diffuser à tous les éléments
            let ones = Tensor::ones(self_clone.tensor().shape().to_vec(), None);
            let scaled_ones = ones
                .mul(scale_tensor)
                .expect("Failed to scale ones in mean gradient");
            let grad = grad_output
                .clone()
                .mul(scaled_ones)
                .expect("Failed to apply gradient in mean");

            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;

        // Créer la variable résultante
        Self::from_operation(
            result_tensor,
            Operation::Mean,
            vec![self.clone()],
            Some(grad_fn),
        )
    }




    /// Applique la fonction d'activation Tanh avec support autograd
    ///
    /// Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    /// Gradient: d(Tanh(x))/dx = 1 - Tanh(x)²

    pub fn tanh(&self) -> Self {
        let result_tensor = match self.tensor().tanh() {
            Ok(t) => t,
            Err(e) => panic!("Error in tanh: {}", e),
        };

        if !GRAD_ENABLED.with(|cell| *cell.borrow()) {
            return Self::from_tensor(result_tensor, false);
        }

        let self_clone = self.clone();
        let grad_fn = if self.requires_grad() {
            Some(Box::new(move |grad_output: &Tensor| {
                match self_clone.tensor().tanh_backward(grad_output) {
                    Ok(t) => vec![t],
                    Err(e) => panic!("Error computing gradient for tanh: {}", e),
                }
            }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>)
        } else {
            None
        };

        Self::from_operation(
            result_tensor,
            Operation::Tanh,
            vec![self.clone()],
            grad_fn,
        )
    }

    // NOTE: swish, gelu, mish, leaky_relu seront implémentés plus tard
    // une fois que les méthodes correspondantes seront ajoutées au struct Tensor



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
            edges: &mut HashSet<(usize, usize)>,
        ) {
            // Si ce nœud a déjà été visité, on s'arrête
            if !visited.insert(var.id()) {
                return;
            }

            // Ajouter ce nœud au graphe
            let data = var.data.read().unwrap();
            let label = if var.is_leaf() {
                format!(
                    "{}\\nLeaf: {}\\nRequires grad: {}",
                    var.id(), var.is_leaf(), var.requires_grad()
                )
            } else if let Some(ref node) = data.grad_fn {
                format!(
                    "{}\\nOp: {}\\nRequires grad: {}",
                    var.id(), node.operation, var.requires_grad()
                )
            } else {
                format!("{}\\nRequires grad: {}", var.id(), var.requires_grad())
            };

            let color = if var.is_leaf() {
                "lightgreen"
            } else if var.requires_grad() {
                "lightblue"
            } else {
                "lightgray"
            };

            dot_content.push_str(&format!(
                "  node{} [label=\"{}\", fillcolor=\"{}\"];\n",
                var.id(), label, color
            ));

            // Ajouter les arêtes pour les entrées avec weak references
            if let Some(ref node) = data.grad_fn {
                for weak_input in &node.inputs {
                    if let Some(input_data) = weak_input.upgrade() {
                        let input_var_data = input_data.read().unwrap();
                        let input_id = input_var_data.id;
                        if edges.insert((input_id, var.id())) {
                            dot_content.push_str(&format!("  node{} -> node{};\n", input_id, var.id()));
                        }
                        // Note: On ne peut plus facilement traverser récursivement avec les weak refs
                        // Il faudrait maintenir une map des variables pour la traversée complète
                    }
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
        println!(
            "Graph saved to {}. Use Graphviz to visualize it: dot -Tpng {} -o {}.png",
            filename,
            filename,
            filename.trim_end_matches(".dot")
        );

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
            visited: &mut HashSet<usize>,
        ) {
            // Éviter les cycles
            if !visited.insert(var.id()) {
                let indent = "  ".repeat(depth);
                result.push_str(&format!("{}Node {} (already visited)\n", indent, var.id()));
                return;
            }

            let indent = "  ".repeat(depth);

            if var.is_leaf() {
                result.push_str(&format!(
                    "{}Node {} (Leaf, requires_grad={})\n",
                    indent, var.id(), var.requires_grad()
                ));
            } else {
                let data = var.data.read().unwrap();
                if let Some(ref node) = data.grad_fn {
                    result.push_str(&format!(
                        "{}Node {} (Op: {}, requires_grad={})\n",
                        indent, var.id(), node.operation, var.requires_grad()
                    ));

                    // Afficher les nœuds d'entrée - Note: avec weak refs, on ne peut plus traverser facilement
                    result.push_str(&format!("{}  [Inputs via weak references]\n", indent));
                } else {
                    result.push_str(&format!(
                        "{}Node {} (No grad_fn, requires_grad={})\n",
                        indent, var.id(), var.requires_grad()
                    ));
                }
            }
        }

        result.push_str("Computation Graph Structure:\n");
        print_node(self, 0, &mut result, &mut visited);

        result
    }
    
    /// Calcule la valeur absolue
    pub fn abs(&self) -> Self {
        let result_tensor = self.tensor().abs().expect("Failed to compute abs");
        
        if !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }
        
        let self_clone = self.clone();
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            // d/dx |x| = sign(x)
            let sign = self_clone.tensor().sign().unwrap();
            let grad = grad_output.clone().mul(sign).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
        
        Self::from_operation(
            result_tensor,
            Operation::None,
            vec![self.clone()],
            Some(grad_fn),
        )
    }
    
    /// Calcule le négatif (opposé)
    pub fn neg(&self) -> Self {
        let result_tensor = self.tensor().neg().expect("Failed to compute neg");
        
        if !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }
        
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            // d/dx (-x) = -1
            let grad = grad_output.neg().unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
        
        Self::from_operation(
            result_tensor,
            Operation::None,
            vec![self.clone()],
            Some(grad_fn),
        )
    }
    
    /// Calcule la racine carrée
    pub fn sqrt(&self) -> Self {
        let result_tensor = self.tensor().sqrt().expect("Failed to compute sqrt");
        
        if !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }
        
        let result_clone = result_tensor.clone();
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            // d/dx sqrt(x) = 1 / (2 * sqrt(x))
            let two = Tensor::full(result_clone.shape().to_vec(), 2.0, result_clone.dtype()).unwrap();
            let denominator = two.mul(result_clone.clone()).unwrap();
            let grad = grad_output.clone().div(denominator).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
        
        Self::from_operation(
            result_tensor,
            Operation::None,
            vec![self.clone()],
            Some(grad_fn),
        )
    }
    
    /// Redimensionne le tenseur
    pub fn reshape(&self, shape: &[usize]) -> Self {
        let result_tensor = self.tensor().reshape(shape).expect("Failed to reshape");
        
        if !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }
        
        let original_shape = self.shape();
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            // Le gradient doit être remodelé vers la forme originale
            let grad = grad_output.reshape(&original_shape).unwrap();
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
        
        Self::from_operation(
            result_tensor,
            Operation::None,
            vec![self.clone()],
            Some(grad_fn),
        )
    }
    
    /// Calcule la moyenne le long d'une dimension spécifique
    pub fn mean_dim(&self, dim: usize, keep_dim: bool) -> Self {
        let result_tensor = self.tensor().mean_dim(Some(dim)).expect("Failed to compute mean");
        
        if !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }
        
        let input_shape = self.shape();
        let dim_size = input_shape[dim] as f64;
        
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            // Le gradient est divisé par la taille de la dimension et broadcast
            let scale = 1.0 / dim_size;
            let scaled_grad = grad_output.mul_scalar(scale).unwrap();
            
            // Si keep_dim est false, on doit unsqueeze avant de broadcaster
            let grad = if keep_dim {
                scaled_grad.broadcast_to(&input_shape).unwrap()
            } else {
                let mut unsqueezed_shape = grad_output.shape().to_vec();
                unsqueezed_shape.insert(dim, 1);
                scaled_grad.reshape(&unsqueezed_shape).unwrap()
                    .broadcast_to(&input_shape).unwrap()
            };
            
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
        
        Self::from_operation(
            result_tensor,
            Operation::Mean,
            vec![self.clone()],
            Some(grad_fn),
        )
    }
    
    /// Calcule la somme le long d'une dimension spécifique
    pub fn sum_dim(&self, dim: usize, keep_dim: bool) -> Self {
        let result_tensor = self.tensor().sum_dim(Some(dim)).expect("Failed to compute sum");
        
        if !self.requires_grad() {
            return Self::from_tensor(result_tensor, false);
        }
        
        let input_shape = self.shape();
        
        let grad_fn = Box::new(move |grad_output: &Tensor| {
            // Le gradient est simplement broadcast à la forme d'entrée
            let grad = if keep_dim {
                grad_output.broadcast_to(&input_shape).unwrap()
            } else {
                let mut unsqueezed_shape = grad_output.shape().to_vec();
                unsqueezed_shape.insert(dim, 1);
                grad_output.reshape(&unsqueezed_shape).unwrap()
                    .broadcast_to(&input_shape).unwrap()
            };
            
            vec![grad]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;
        
        Self::from_operation(
            result_tensor,
            Operation::Sum,
            vec![self.clone()],
            Some(grad_fn),
        )
    }
    
    /// Multiplies by a scalar
    pub fn mul_scalar(&self, scalar: f64) -> Self {
        let result_tensor = self.tensor().mul_scalar(scalar).expect("Failed to multiply by scalar");
        Self::from_tensor(result_tensor, self.requires_grad())
    }
    
    /// Adds a scalar
    pub fn add_scalar(&self, scalar: f64) -> Self {
        let result_tensor = self.tensor().add_scalar(scalar).expect("Failed to add scalar");
        Self::from_tensor(result_tensor, self.requires_grad())
    }
}
