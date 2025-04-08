// Dans rustytorch_viz/src/lib.rs
use std::collections::{HashMap};
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

}