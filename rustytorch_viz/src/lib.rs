// Dans rustytorch_viz/src/lib.rs
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use rustytorch_autograd::{Variable};

pub struct GraphViz {
    nodes: HashMap<usize, String>,
    edges: Vec<(usize, usize)>,
}

impl GraphViz {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_variable(&mut self, var: &Variable) {
        // Ajouter ce nœud s'il n'existe pas déjà
        if !self.nodes.contains_key(&var.id) {
            let label = if var.is_leaf {
                format!("Leaf({:?})", var.tensor.shape())
            } else {
                format!("{:?}({:?})", var.grad_fn.as_ref().unwrap().operation, var.tensor.shape())
            };
            self.nodes.insert(var.id, label);

            // Parcourir le graphe et ajouter les nœuds/arêtes
            if let Some(ref node) = var.grad_fn {
                for input in &node.inputs {
                    self.add_variable(input);
                    self.edges.push((input.id, var.id));
                }
            }
        }
    }

    pub fn to_dot(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Écrire l'en-tête DOT
        writeln!(file, "digraph ComputationGraph {{")?;

        // Écrire les nœuds
        for (id, label) in &self.nodes {
            writeln!(file, "    node{} [label=\"{}\"];", id, label)?;
        }

        // Écrire les arêtes
        for (from, to) in &self.edges {
            writeln!(file, "    node{} -> node{};", from, to)?;
        }

        // Fermer le graphe
        writeln!(file, "}}")?;

        Ok(())
    }


    // /// Fonction qui visualise le graphe de calcul à partir de cette variable
    // pub fn visualize_graph(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    //     // Cette fonction pourrait construire une représentation DOT du graphe
    //     // et l'enregistrer dans un fichier pour visualisation avec Graphviz
    //
    //     use std::fs::File;
    //     use std::io::Write;
    //
    //     let mut dot_content = String::from("digraph ComputationGraph {\n");
    //     dot_content.push_str("  rankdir=LR;\n");
    //     dot_content.push_str("  node [shape=box, style=filled, color=lightblue];\n\n");
    //
    //     // Ensembles pour suivre les nœuds et arêtes déjà visités
    //     let mut visited_nodes = HashSet::new();
    //     let mut edges = HashSet::new();
    //
    //     // Fonction récursive pour construire le graphe DOT
    //     fn build_graph(
    //         var: &Variable,
    //         dot_content: &mut String,
    //         visited: &mut HashSet<usize>,
    //         edges: &mut HashSet<(usize, usize)>
    //     ) {
    //         // Si ce nœud a déjà été visité, on s'arrête
    //         if !visited.insert(var.id) {
    //             return;
    //         }
    //
    //         // Ajouter ce nœud au graphe
    //         let label = if var.is_leaf {
    //             format!("{}\\nLeaf: {}\\nRequires grad: {}",
    //                     var.id, var.is_leaf, var.requires_grad)
    //         } else if let Some(ref node) = var.grad_fn {
    //             format!("{}\\nOp: {}\\nRequires grad: {}",
    //                     var.id, node.operation, var.requires_grad)
    //         } else {
    //             format!("{}\\nRequires grad: {}", var.id, var.requires_grad)
    //         };
    //
    //         let color = if var.is_leaf {
    //             "lightgreen"
    //         } else if var.requires_grad {
    //             "lightblue"
    //         } else {
    //             "lightgray"
    //         };
    //
    //         dot_content.push_str(&format!("  node{} [label=\"{}\", fillcolor=\"{}\"];\n",
    //                                       var.id, label, color));
    //
    //         // Ajouter les arêtes pour les entrées
    //         if let Some(ref node) = var.grad_fn {
    //             for input in &node.inputs {
    //                 if edges.insert((input.id, var.id)) {
    //                     dot_content.push_str(&format!("  node{} -> node{};\n",
    //                                                   input.id, var.id));
    //                 }
    //                 build_graph(input, dot_content, visited, edges);
    //             }
    //         }
    //     }
    //
    //     // Construire le graphe en partant de cette variable
    //     build_graph(self, &mut dot_content, &mut visited_nodes, &mut edges);
    //
    //     // Finaliser le contenu DOT
    //     dot_content.push_str("}\n");
    //
    //     // Écrire dans un fichier
    //     let mut file = File::create(filename)?;
    //     file.write_all(dot_content.as_bytes())?;
    //
    //     // On pourrait également lancer automatiquement la commande dot pour générer une image
    //     // si Graphviz est installé
    //     println!("Graph saved to {}. Use Graphviz to visualize it: dot -Tpng {} -o {}.png",
    //              filename, filename, filename.trim_end_matches(".dot"));
    //
    //     Ok(())
    // }
    //
    //
    //
    // /// Retourne la représentation textuelle du graphe de calcul
    // pub fn print_graph_structure(&self) -> String {
    //     let mut result = String::new();
    //     let mut visited = HashSet::new();
    //
    //     fn print_node(
    //         var: &Variable,
    //         depth: usize,
    //         result: &mut String,
    //         visited: &mut HashSet<usize>
    //     ) {
    //         // Éviter les cycles
    //         if !visited.insert(var.id) {
    //             let indent = "  ".repeat(depth);
    //             result.push_str(&format!("{}Node {} (already visited)\n", indent, var.id));
    //             return;
    //         }
    //
    //         let indent = "  ".repeat(depth);
    //
    //         if var.is_leaf {
    //             result.push_str(&format!("{}Node {} (Leaf, requires_grad={})\n",
    //                                      indent, var.id, var.requires_grad));
    //         } else if let Some(ref node) = var.grad_fn {
    //             result.push_str(&format!("{}Node {} (Op: {}, requires_grad={})\n",
    //                                      indent, var.id, node.operation, var.requires_grad));
    //
    //             // Afficher les nœuds d'entrée
    //             for (i, input) in node.inputs.iter().enumerate() {
    //                 result.push_str(&format!("{}  Input {}:\n", indent, i));
    //                 print_node(input, depth + 2, result, visited);
    //             }
    //         } else {
    //             result.push_str(&format!("{}Node {} (No grad_fn, requires_grad={})\n",
    //                                      indent, var.id, var.requires_grad));
    //         }
    //     }
    //
    //     result.push_str("Computation Graph Structure:\n");
    //     print_node(self, 0, &mut result, &mut visited);
    //
    //     result
    // }
    //




}