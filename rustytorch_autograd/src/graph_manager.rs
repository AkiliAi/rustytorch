// rustytorch_autograd/src/graph_manager.rs

use crate::{Operation, Variable};
use rustytorch_tensor::Tensor;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Weak, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Node optimisé avec weak references pour éviter les cycles de références
pub struct OptimizedNode {
    pub operation: Operation,
    /// Weak references vers les variables d'entrée pour éviter les cycles
    pub inputs: Vec<Weak<RwLock<VariableData>>>,
    /// Fonction de gradient boxée
    pub grad_fn: Option<Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>>,
    /// Timestamp de création pour le garbage collection
    pub created_at: Instant,
}

/// Données internes d'une variable séparées pour permettre les weak references
pub struct VariableData {
    pub tensor: Tensor,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub grad: Option<Tensor>,
    pub grad_fn: Option<Arc<OptimizedNode>>,
    pub id: usize,
    pub version: u64, // Version pour invalider les caches
    pub hooks: Vec<Box<dyn Fn(&Tensor) -> Tensor + Send + Sync>>,
}

/// Handle pour un hook enregistré
pub struct HookHandle {
    pub variable_id: usize,
    pub hook_id: usize,
}

/// Gestionnaire global du graphe de calcul avec memory management optimisé
pub struct GraphManager {
    /// Map des nœuds actifs avec weak references
    nodes: Arc<RwLock<HashMap<usize, Weak<OptimizedNode>>>>,
    /// Map des variables actives
    variables: Arc<RwLock<HashMap<usize, Arc<RwLock<VariableData>>>>>,
    /// File d'attente pour le garbage collection
    gc_queue: Arc<Mutex<VecDeque<usize>>>,
    /// Configuration du garbage collector
    gc_config: GCConfig,
    /// Statistiques du graphe
    stats: Arc<RwLock<GraphStats>>,
}

/// Configuration du garbage collector
pub struct GCConfig {
    /// Intervalle entre les collections
    pub cleanup_interval: Duration,
    /// Age maximum des nœuds avant collection
    pub max_node_age: Duration,
    /// Taille maximale du graphe avant collection forcée
    pub max_graph_size: usize,
    /// Activer le GC automatique
    pub auto_gc_enabled: bool,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            cleanup_interval: Duration::from_secs(60), // 1 minute
            max_node_age: Duration::from_secs(300),    // 5 minutes
            max_graph_size: 10_000,                    // 10k nodes max
            auto_gc_enabled: true,
        }
    }
}

/// Statistiques du graphe pour monitoring
#[derive(Default, Debug, Clone)]
pub struct GraphStats {
    pub total_nodes_created: u64,
    pub active_nodes: usize,
    pub total_variables_created: u64,
    pub active_variables: usize,
    pub gc_runs: u64,
    pub nodes_collected: u64,
    pub last_gc_time: Option<Instant>,
}

impl GraphManager {
    /// Crée un nouveau gestionnaire de graphe
    pub fn new() -> Self {
        Self::with_config(GCConfig::default())
    }

    /// Crée un gestionnaire avec une configuration personnalisée
    pub fn with_config(config: GCConfig) -> Self {
        let manager = Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            variables: Arc::new(RwLock::new(HashMap::new())),
            gc_queue: Arc::new(Mutex::new(VecDeque::new())),
            gc_config: config,
            stats: Arc::new(RwLock::new(GraphStats::default())),
        };

        // Lancer le thread de GC si activé
        if manager.gc_config.auto_gc_enabled {
            manager.start_gc_thread();
        }

        manager
    }

    /// Lance un thread de garbage collection en arrière-plan
    fn start_gc_thread(&self) {
        let nodes = Arc::clone(&self.nodes);
        let variables = Arc::clone(&self.variables);
        let gc_queue = Arc::clone(&self.gc_queue);
        let stats = Arc::clone(&self.stats);
        let interval = self.gc_config.cleanup_interval;
        let max_age = self.gc_config.max_node_age;

        std::thread::spawn(move || {
            loop {
                std::thread::sleep(interval);
                Self::run_gc_cycle(&nodes, &variables, &gc_queue, &stats, max_age);
            }
        });
    }

    /// Execute un cycle de garbage collection
    fn run_gc_cycle(
        nodes: &Arc<RwLock<HashMap<usize, Weak<OptimizedNode>>>>,
        variables: &Arc<RwLock<HashMap<usize, Arc<RwLock<VariableData>>>>>,
        gc_queue: &Arc<Mutex<VecDeque<usize>>>,
        stats: &Arc<RwLock<GraphStats>>,
        max_age: Duration,
    ) {
        let now = Instant::now();
        let mut nodes_to_remove = Vec::new();
        let mut vars_to_remove = Vec::new();

        // Phase 1: Identifier les nœuds morts
        {
            let nodes_guard = nodes.read().unwrap();
            for (&id, weak_node) in nodes_guard.iter() {
                if weak_node.strong_count() == 0 {
                    nodes_to_remove.push(id);
                } else if let Some(node) = weak_node.upgrade() {
                    // Vérifier l'âge du nœud
                    if now.duration_since(node.created_at) > max_age {
                        // Vérifier si le nœud est encore référencé
                        let has_valid_refs = node.inputs.iter()
                            .any(|weak_var| weak_var.strong_count() > 0);
                        
                        if !has_valid_refs {
                            nodes_to_remove.push(id);
                        }
                    }
                }
            }
        }

        // Phase 2: Nettoyer les nœuds morts
        if !nodes_to_remove.is_empty() {
            let mut nodes_guard = nodes.write().unwrap();
            for id in &nodes_to_remove {
                nodes_guard.remove(id);
            }
        }

        // Phase 3: Identifier les variables non référencées
        {
            let vars_guard = variables.read().unwrap();
            for (&id, var_arc) in vars_guard.iter() {
                // Garder les variables leaf avec gradient requis
                if Arc::strong_count(var_arc) <= 1 {
                    let var_data = var_arc.read().unwrap();
                    if !var_data.is_leaf || !var_data.requires_grad {
                        vars_to_remove.push(id);
                    }
                }
            }
        }

        // Phase 4: Nettoyer les variables
        if !vars_to_remove.is_empty() {
            let mut vars_guard = variables.write().unwrap();
            for id in &vars_to_remove {
                vars_guard.remove(id);
            }
        }

        // Mettre à jour les statistiques
        {
            let mut stats_guard = stats.write().unwrap();
            stats_guard.gc_runs += 1;
            stats_guard.nodes_collected += nodes_to_remove.len() as u64;
            stats_guard.active_nodes = nodes.read().unwrap().len();
            stats_guard.active_variables = variables.read().unwrap().len();
            stats_guard.last_gc_time = Some(now);
        }
    }

    /// Force un cycle de garbage collection
    pub fn force_gc(&self) {
        Self::run_gc_cycle(
            &self.nodes,
            &self.variables,
            &self.gc_queue,
            &self.stats,
            Duration::from_secs(0), // Collecter tous les nœuds
        );
    }

    /// Enregistre une nouvelle variable dans le graphe
    pub fn register_variable(&self, var_data: VariableData) -> Arc<RwLock<VariableData>> {
        let id = var_data.id;
        let var_arc = Arc::new(RwLock::new(var_data));
        
        {
            let mut vars_guard = self.variables.write().unwrap();
            vars_guard.insert(id, Arc::clone(&var_arc));
        }

        // Mettre à jour les stats
        {
            let mut stats_guard = self.stats.write().unwrap();
            stats_guard.total_variables_created += 1;
            stats_guard.active_variables = self.variables.read().unwrap().len();
        }

        // Vérifier si on doit déclencher un GC
        if self.should_trigger_gc() {
            self.gc_queue.lock().unwrap().push_back(id);
        }

        var_arc
    }

    /// Enregistre un nouveau nœud dans le graphe
    pub fn register_node(&self, node: OptimizedNode) -> Arc<OptimizedNode> {
        let node_arc = Arc::new(node);
        let node_id = Arc::as_ptr(&node_arc) as usize;
        
        {
            let mut nodes_guard = self.nodes.write().unwrap();
            nodes_guard.insert(node_id, Arc::downgrade(&node_arc));
        }

        // Mettre à jour les stats
        {
            let mut stats_guard = self.stats.write().unwrap();
            stats_guard.total_nodes_created += 1;
            stats_guard.active_nodes = self.nodes.read().unwrap().len();
        }

        node_arc
    }

    /// Vérifie si un GC doit être déclenché
    fn should_trigger_gc(&self) -> bool {
        let stats = self.stats.read().unwrap();
        stats.active_nodes > self.gc_config.max_graph_size ||
        stats.active_variables > self.gc_config.max_graph_size
    }

    /// Obtient les statistiques actuelles du graphe
    pub fn get_stats(&self) -> GraphStats {
        self.stats.read().unwrap().clone()
    }

    /// Configure le garbage collector
    pub fn set_gc_config(&mut self, config: GCConfig) {
        self.gc_config = config;
    }

    /// Nettoie complètement le graphe
    pub fn clear(&self) {
        self.nodes.write().unwrap().clear();
        self.variables.write().unwrap().clear();
        self.gc_queue.lock().unwrap().clear();
        
        let mut stats = self.stats.write().unwrap();
        stats.active_nodes = 0;
        stats.active_variables = 0;
    }

    /// Retourne le nombre de nœuds actifs
    pub fn active_nodes_count(&self) -> usize {
        self.nodes.read().unwrap().len()
    }

    /// Retourne le nombre de variables actives
    pub fn active_variables_count(&self) -> usize {
        self.variables.read().unwrap().len()
    }
}

/// Singleton global pour le gestionnaire de graphe
lazy_static::lazy_static! {
    pub static ref GRAPH_MANAGER: GraphManager = GraphManager::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_manager_creation() {
        let manager = GraphManager::new();
        assert_eq!(manager.active_nodes_count(), 0);
        assert_eq!(manager.active_variables_count(), 0);
    }

    #[test]
    fn test_gc_config() {
        let config = GCConfig {
            cleanup_interval: Duration::from_secs(30),
            max_node_age: Duration::from_secs(120),
            max_graph_size: 5000,
            auto_gc_enabled: false,
        };
        
        let manager = GraphManager::with_config(config);
        assert_eq!(manager.gc_config.max_graph_size, 5000);
    }

    #[test]
    fn test_variable_registration() {
        let manager = GraphManager::new();
        
        let var_data = VariableData {
            tensor: Tensor::zeros(vec![2, 2], None),
            requires_grad: true,
            is_leaf: true,
            grad: None,
            grad_fn: None,
            id: 1,
            version: 0,
            hooks: Vec::new(),
        };
        
        let _var_ref = manager.register_variable(var_data);
        assert_eq!(manager.active_variables_count(), 1);
        
        let stats = manager.get_stats();
        assert_eq!(stats.total_variables_created, 1);
        assert_eq!(stats.active_variables, 1);
    }
}