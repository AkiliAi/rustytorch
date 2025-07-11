//! Optimisations de performance pour le système autograd
//! 
//! Ce module contient des améliorations de performance critiques pour le système
//! de différentiation automatique, incluant:
//! - Optimisations mémoire
//! - Fusion d'opérations
//! - Gestion de cache optimisée
//! - Gradient accumulation optimisée

use crate::{Variable, Operation, VariableData, OptimizedNode};
use rustytorch_tensor::Tensor;
use rustytorch_core::{Result as CoreResult, NumericOps, Reduction};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Weak};

/// Configuration pour les optimisations de performance
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Taille initiale de la file pour le backward pass
    pub initial_queue_capacity: usize,
    /// Taille initiale du HashMap pour l'accumulation de gradients
    pub initial_accumulator_capacity: usize,
    /// Activer la fusion d'opérations
    pub enable_operation_fusion: bool,
    /// Activer le cache de gradients
    pub enable_gradient_cache: bool,
    /// Seuil pour le checkpointing automatique
    pub checkpointing_threshold: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            initial_queue_capacity: 64,
            initial_accumulator_capacity: 32,
            enable_operation_fusion: true,
            enable_gradient_cache: true,
            checkpointing_threshold: 1000,
        }
    }
}

/// Cache pour les gradients calculés
pub struct GradientCache {
    cache: HashMap<(usize, String), Tensor>,
    max_size: usize,
    hits: usize,
    misses: usize,
}

impl GradientCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, key: &(usize, String)) -> Option<&Tensor> {
        if let Some(tensor) = self.cache.get(key) {
            self.hits += 1;
            Some(tensor)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: (usize, String), value: Tensor) {
        if self.cache.len() >= self.max_size {
            // Simple LRU: remove first entry
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, value);
    }

    pub fn stats(&self) -> (usize, usize, f64) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 { self.hits as f64 / total as f64 } else { 0.0 };
        (self.hits, self.misses, hit_rate)
    }
}

/// Pool de buffers réutilisables pour éviter les allocations
pub struct BufferPool {
    tensor_buffers: Vec<Vec<Tensor>>,
    queue_buffers: Vec<Vec<(Arc<RwLock<VariableData>>, Tensor)>>,
    id_buffers: Vec<Vec<usize>>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            tensor_buffers: Vec::new(),
            queue_buffers: Vec::new(),
            id_buffers: Vec::new(),
        }
    }

    pub fn get_tensor_buffer(&mut self, min_capacity: usize) -> Vec<Tensor> {
        if let Some(mut buffer) = self.tensor_buffers.pop() {
            buffer.clear();
            if buffer.capacity() < min_capacity {
                buffer.reserve(min_capacity - buffer.capacity());
            }
            buffer
        } else {
            Vec::with_capacity(min_capacity)
        }
    }

    pub fn return_tensor_buffer(&mut self, buffer: Vec<Tensor>) {
        if buffer.capacity() > 0 && self.tensor_buffers.len() < 10 {
            self.tensor_buffers.push(buffer);
        }
    }

    pub fn get_queue_buffer(&mut self, min_capacity: usize) -> Vec<(Arc<RwLock<VariableData>>, Tensor)> {
        if let Some(mut buffer) = self.queue_buffers.pop() {
            buffer.clear();
            if buffer.capacity() < min_capacity {
                buffer.reserve(min_capacity - buffer.capacity());
            }
            buffer
        } else {
            Vec::with_capacity(min_capacity)
        }
    }

    pub fn return_queue_buffer(&mut self, buffer: Vec<(Arc<RwLock<VariableData>>, Tensor)>) {
        if buffer.capacity() > 0 && self.queue_buffers.len() < 10 {
            self.queue_buffers.push(buffer);
        }
    }
}

/// Optimiseur de gradient avec accumulation efficace
pub struct OptimizedGradientAccumulator {
    gradients: HashMap<usize, Tensor>,
    pending_updates: Vec<(usize, Tensor)>,
    batch_size: usize,
}

impl OptimizedGradientAccumulator {
    pub fn new(initial_capacity: usize, batch_size: usize) -> Self {
        Self {
            gradients: HashMap::with_capacity(initial_capacity),
            pending_updates: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    /// Ajoute un gradient à accumuler
    pub fn add_gradient(&mut self, var_id: usize, grad: Tensor) -> CoreResult<()> {
        self.pending_updates.push((var_id, grad));
        
        // Flush en batch quand on atteint la taille limite
        if self.pending_updates.len() >= self.batch_size {
            self.flush_pending()?;
        }
        
        Ok(())
    }

    /// Applique tous les gradients en attente
    pub fn flush_pending(&mut self) -> CoreResult<()> {
        for (var_id, grad) in self.pending_updates.drain(..) {
            if let Some(existing_grad) = self.gradients.get_mut(&var_id) {
                *existing_grad = existing_grad.clone().add(grad)?;
            } else {
                self.gradients.insert(var_id, grad);
            }
        }
        Ok(())
    }

    /// Récupère le gradient accumulé pour une variable
    pub fn get_gradient(&self, var_id: usize) -> Option<&Tensor> {
        self.gradients.get(&var_id)
    }

    /// Finalise l'accumulation et retourne tous les gradients
    pub fn finalize(mut self) -> CoreResult<HashMap<usize, Tensor>> {
        self.flush_pending()?;
        Ok(self.gradients)
    }
}

/// Détecteur de patterns pour la fusion d'opérations
#[derive(Debug, Clone, PartialEq)]
pub enum FusablePattern {
    /// Addition suivie de multiplication (axpy: a*x + y)
    AddMul,
    /// Exp suivi de Sum (pour softmax)
    ExpSum,
    /// Activation + multiplication (pour scaling)
    ActivationScale,
    /// Chaîne de activations (ReLU + Sigmoid, etc.)
    ActivationChain,
}

pub struct OperationFuser {
    enabled: bool,
    fusion_buffer: Vec<Operation>,
    max_fusion_length: usize,
}

impl OperationFuser {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            fusion_buffer: Vec::with_capacity(8),
            max_fusion_length: 4,
        }
    }

    /// Analyse une séquence d'opérations pour détecter des patterns fusables
    pub fn analyze_sequence(&mut self, ops: &[Operation]) -> Vec<FusablePattern> {
        if !self.enabled || ops.len() < 2 {
            return Vec::new();
        }

        let mut patterns = Vec::new();
        
        for window in ops.windows(2) {
            match (&window[0], &window[1]) {
                (Operation::Add, Operation::Mul) => patterns.push(FusablePattern::AddMul),
                (Operation::Exp, Operation::Sum) => patterns.push(FusablePattern::ExpSum),
                (Operation::Relu, Operation::Mul) => patterns.push(FusablePattern::ActivationScale),
                (Operation::Sigmoid, Operation::Mul) => patterns.push(FusablePattern::ActivationScale),
                (Operation::Relu, Operation::Sigmoid) => patterns.push(FusablePattern::ActivationChain),
                _ => {}
            }
        }
        
        patterns
    }

    /// Fusionne les opérations détectées pour optimiser le calcul
    pub fn apply_fusion(&self, pattern: &FusablePattern, operands: &[Tensor]) -> CoreResult<Tensor> {
        match pattern {
            FusablePattern::AddMul => {
                // Optimisation: a*x + y en une seule opération
                if operands.len() >= 3 {
                    let scaled = operands[0].clone().mul(operands[1].clone())
                        .map_err(|e| rustytorch_core::CoreError::InvalidOperation { 
                            operation: "AddMul fusion".to_string(), 
                            reason: format!("Tensor multiplication failed: {}", e) 
                        })?;
                    scaled.add(operands[2].clone())
                        .map_err(|e| rustytorch_core::CoreError::InvalidOperation { 
                            operation: "AddMul fusion".to_string(), 
                            reason: format!("Tensor addition failed: {}", e) 
                        })
                } else {
                    Err(rustytorch_core::CoreError::InvalidOperation {
                        operation: "AddMul fusion".to_string(),
                        reason: "Not enough operands".to_string()
                    })
                }
            },
            FusablePattern::ExpSum => {
                // Optimisation pour softmax: exp(x) puis sum
                if !operands.is_empty() {
                    let exp_result = operands[0].exp()
                        .map_err(|e| rustytorch_core::CoreError::InvalidOperation { 
                            operation: "ExpSum fusion".to_string(), 
                            reason: format!("Tensor exp failed: {}", e) 
                        })?;
                    exp_result.sum()
                        .map_err(|e| rustytorch_core::CoreError::InvalidOperation { 
                            operation: "ExpSum fusion".to_string(), 
                            reason: format!("Tensor sum failed: {}", e) 
                        })
                } else {
                    Err(rustytorch_core::CoreError::InvalidOperation {
                        operation: "ExpSum fusion".to_string(),
                        reason: "No operands".to_string()
                    })
                }
            },
            _ => {
                // Autres patterns non implémentés pour l'instant
                Err(rustytorch_core::CoreError::InvalidOperation {
                    operation: "Operation fusion".to_string(),
                    reason: "Pattern not implemented".to_string()
                })
            }
        }
    }
}

/// Gestionnaire de checkpointing pour économiser la mémoire
pub struct CheckpointManager {
    checkpoints: HashMap<usize, Tensor>,
    checkpoint_threshold: usize,
    current_memory_usage: usize,
}

impl CheckpointManager {
    pub fn new(threshold: usize) -> Self {
        Self {
            checkpoints: HashMap::new(),
            checkpoint_threshold: threshold,
            current_memory_usage: 0,
        }
    }

    /// Vérifie si une variable doit être checkpointée
    pub fn should_checkpoint(&self, node_count: usize) -> bool {
        node_count > self.checkpoint_threshold
    }

    /// Sauvegarde un tensor en checkpoint
    pub fn save_checkpoint(&mut self, var_id: usize, tensor: Tensor) {
        let tensor_size = tensor.numel();
        self.current_memory_usage += tensor_size;
        self.checkpoints.insert(var_id, tensor);
    }

    /// Restaure un tensor depuis un checkpoint
    pub fn restore_checkpoint(&mut self, var_id: usize) -> Option<Tensor> {
        if let Some(tensor) = self.checkpoints.remove(&var_id) {
            self.current_memory_usage = self.current_memory_usage.saturating_sub(tensor.numel());
            Some(tensor)
        } else {
            None
        }
    }

    /// Retourne l'utilisation mémoire actuelle des checkpoints
    pub fn memory_usage(&self) -> usize {
        self.current_memory_usage
    }

    /// Nettoie tous les checkpoints
    pub fn clear(&mut self) {
        self.checkpoints.clear();
        self.current_memory_usage = 0;
    }
}

/// Configuration globale des optimisations
thread_local! {
    static PERFORMANCE_CONFIG: std::cell::RefCell<PerformanceConfig> = std::cell::RefCell::new(PerformanceConfig::default());
    static GRADIENT_CACHE: std::cell::RefCell<GradientCache> = std::cell::RefCell::new(GradientCache::new(1000));
    static BUFFER_POOL: std::cell::RefCell<BufferPool> = std::cell::RefCell::new(BufferPool::new());
}

/// Interface publique pour configurer les optimisations
pub fn set_performance_config(config: PerformanceConfig) {
    PERFORMANCE_CONFIG.with(|c| *c.borrow_mut() = config);
}

pub fn get_performance_config() -> PerformanceConfig {
    PERFORMANCE_CONFIG.with(|c| c.borrow().clone())
}

/// Interface pour accéder au cache de gradients
pub fn with_gradient_cache<F, R>(f: F) -> R 
where 
    F: FnOnce(&mut GradientCache) -> R,
{
    GRADIENT_CACHE.with(|cache| f(&mut cache.borrow_mut()))
}

/// Interface pour accéder au pool de buffers
pub fn with_buffer_pool<F, R>(f: F) -> R 
where 
    F: FnOnce(&mut BufferPool) -> R,
{
    BUFFER_POOL.with(|pool| f(&mut pool.borrow_mut()))
}

/// Statistiques de performance
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_hit_rate: f64,
    pub operations_fused: usize,
    pub checkpoints_created: usize,
    pub memory_saved: usize,
}

/// Collecte les statistiques de performance actuelles
pub fn get_performance_stats() -> PerformanceStats {
    with_gradient_cache(|cache| {
        let (hits, misses, hit_rate) = cache.stats();
        PerformanceStats {
            cache_hits: hits,
            cache_misses: misses,
            cache_hit_rate: hit_rate,
            operations_fused: 0, // TODO: implémenter le tracking
            checkpoints_created: 0, // TODO: implémenter le tracking
            memory_saved: 0, // TODO: implémenter le tracking
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_cache() {
        let mut cache = GradientCache::new(3);
        let tensor = Tensor::ones(vec![2, 2], None);
        
        // Test cache miss
        assert!(cache.get(&(1, "test".to_string())).is_none());
        
        // Test cache insert and hit
        cache.insert((1, "test".to_string()), tensor.clone());
        assert!(cache.get(&(1, "test".to_string())).is_some());
        
        // Test cache eviction
        let tensor2 = Tensor::ones(vec![3, 3], None);
        let tensor3 = Tensor::ones(vec![4, 4], None);
        let tensor4 = Tensor::ones(vec![5, 5], None);
        
        cache.insert((2, "test2".to_string()), tensor2);
        cache.insert((3, "test3".to_string()), tensor3);
        cache.insert((4, "test4".to_string()), tensor4);
        
        // First entry should be evicted
        assert!(cache.get(&(1, "test".to_string())).is_none());
        assert!(cache.get(&(4, "test4".to_string())).is_some());
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new();
        
        // Get buffer
        let buffer = pool.get_tensor_buffer(10);
        assert!(buffer.capacity() >= 10);
        
        // Return buffer
        pool.return_tensor_buffer(buffer);
        
        // Get buffer again - should reuse
        let buffer2 = pool.get_tensor_buffer(5);
        assert!(buffer2.capacity() >= 10); // Should have previous capacity
    }

    #[test]
    fn test_optimized_gradient_accumulator() {
        let mut accumulator = OptimizedGradientAccumulator::new(10, 2);
        
        let grad1 = Tensor::ones(vec![2, 2], None);
        let grad2 = Tensor::ones(vec![2, 2], None);
        
        accumulator.add_gradient(1, grad1).unwrap();
        accumulator.add_gradient(1, grad2).unwrap(); // Should trigger flush
        
        let gradients = accumulator.finalize().unwrap();
        assert!(gradients.contains_key(&1));
    }

    #[test]
    fn test_operation_fuser() {
        let mut fuser = OperationFuser::new(true);
        let ops = vec![Operation::Add, Operation::Mul, Operation::Exp, Operation::Sum];
        
        let patterns = fuser.analyze_sequence(&ops);
        assert!(!patterns.is_empty());
        assert!(patterns.contains(&FusablePattern::AddMul));
        assert!(patterns.contains(&FusablePattern::ExpSum));
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager = CheckpointManager::new(100);
        
        assert!(!manager.should_checkpoint(50));
        assert!(manager.should_checkpoint(150));
        
        let tensor = Tensor::ones(vec![10, 10], None);
        manager.save_checkpoint(1, tensor);
        
        let restored = manager.restore_checkpoint(1);
        assert!(restored.is_some());
        assert_eq!(manager.memory_usage(), 0);
    }
}