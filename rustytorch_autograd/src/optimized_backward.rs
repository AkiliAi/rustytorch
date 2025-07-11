//! Backward pass optimisé pour les performances
//! 
//! Ce module contient une implémentation optimisée du backward pass qui:
//! - Utilise des allocations mémoire plus efficaces
//! - Implémente l'accumulation de gradient en batch
//! - Utilise le cache et le pooling de buffers
//! - Optimise les parcours de graphe

use crate::{Variable, VariableData, OptimizedNode};
use crate::performance_optimizations::{
    OptimizedGradientAccumulator, CheckpointManager, OperationFuser,
    get_performance_config, with_buffer_pool, with_gradient_cache
};
use rustytorch_tensor::Tensor;
use rustytorch_core::{Result as CoreResult, NumericOps};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// Version optimisée du backward pass
impl Variable {
    /// Backward pass optimisé avec gestion mémoire améliorée
    pub fn backward_optimized(
        &mut self,
        grad_output: Option<Tensor>,
        retain_graph: bool,
        create_graph: bool,
    ) -> CoreResult<()> {
        let config = get_performance_config();
        
        // Accumulateur de gradients optimisé
        let mut accumulator = OptimizedGradientAccumulator::new(
            config.initial_accumulator_capacity,
            8, // batch size
        );
        
        // Manager de checkpointing si nécessaire
        let mut checkpoint_manager = if config.checkpointing_threshold > 0 {
            Some(CheckpointManager::new(config.checkpointing_threshold))
        } else {
            None
        };
        
        // Fusionneur d'opérations
        let operation_fuser = OperationFuser::new(config.enable_operation_fusion);
        
        // File optimisée avec buffer pool
        let mut queue = with_buffer_pool(|pool| {
            pool.get_queue_buffer(config.initial_queue_capacity)
        });
        
        // Gradient initial
        let initial_grad = grad_output.unwrap_or_else(|| {
            Tensor::ones(self.shape(), None)
        });
        
        // Initialiser la file
        queue.push((Arc::clone(&self.data), initial_grad));
        
        // Statistiques pour optimisations futures
        let mut nodes_processed = 0;
        let mut cache_lookups = 0;
        
        // Parcours optimisé du graphe
        while let Some((var_data_ref, grad_output)) = queue.pop() {
            nodes_processed += 1;
            
            // Lire les données de la variable
            let var_data = var_data_ref.read().unwrap();
            let var_id = var_data.id;
            
            // Vérifier le cache de gradients si activé
            if config.enable_gradient_cache {
                let cache_key = (var_id, format!("{:?}", grad_output.shape()));
                
                let cached_result = with_gradient_cache(|cache| {
                    cache_lookups += 1;
                    cache.get(&cache_key).cloned()
                });
                
                if let Some(cached_grad) = cached_result {
                    // Utiliser le gradient mis en cache
                    accumulator.add_gradient(var_id, cached_grad)?;
                    continue;
                }
            }
            
            // Accumuler le gradient de façon optimisée
            accumulator.add_gradient(var_id, grad_output.clone())?;
            
            // Gérer les checkpoints si nécessaire
            if let Some(ref mut manager) = checkpoint_manager {
                if manager.should_checkpoint(nodes_processed) {
                    manager.save_checkpoint(var_id, grad_output.clone());
                }
            }
            
            // Si c'est une feuille ou pas de grad_fn, continuer
            if var_data.is_leaf || var_data.grad_fn.is_none() {
                drop(var_data); // Release lock early
                continue;
            }
            
            // Propager à travers le nœud
            if let Some(ref node) = var_data.grad_fn {
                if let Some(ref grad_fn) = node.grad_fn {
                    // Calculer les gradients pour les inputs
                    let input_grads = self.compute_gradients_optimized(
                        grad_fn,
                        &grad_output,
                        create_graph,
                        &operation_fuser,
                    )?;
                    
                    // Mettre en cache le résultat si activé
                    if config.enable_gradient_cache && !input_grads.is_empty() {
                        let cache_key = (var_id, format!("{:?}", grad_output.shape()));
                        with_gradient_cache(|cache| {
                            cache.insert(cache_key, input_grads[0].clone());
                        });
                    }
                    
                    // Ajouter les gradients d'entrée à la file
                    for (weak_input, input_grad) in node.inputs.iter().zip(input_grads.iter()) {
                        if let Some(input_data) = weak_input.upgrade() {
                            queue.push((input_data, input_grad.clone()));
                        }
                    }
                }
            }
            
            drop(var_data); // Release lock as soon as possible
        }
        
        // Finaliser l'accumulation
        let final_gradients = accumulator.finalize()?;
        
        // Appliquer les gradients finaux aux variables
        self.apply_gradients_optimized(final_gradients, retain_graph)?;
        
        // Nettoyer les ressources
        if let Some(mut manager) = checkpoint_manager {
            manager.clear();
        }
        
        // Retourner le buffer à la pool
        with_buffer_pool(|pool| {
            pool.return_queue_buffer(queue);
        });
        
        Ok(())
    }
    
    /// Calcul optimisé des gradients avec fusion d'opérations
    fn compute_gradients_optimized(
        &self,
        grad_fn: &dyn Fn(&Tensor) -> Vec<Tensor>,
        grad_output: &Tensor,
        create_graph: bool,
        _operation_fuser: &OperationFuser,
    ) -> CoreResult<Vec<Tensor>> {
        // Pour l'instant, utiliser la fonction de gradient existante
        // TODO: Implémenter la fusion d'opérations ici
        Ok(grad_fn(grad_output))
    }
    
    /// Application optimisée des gradients calculés
    fn apply_gradients_optimized(
        &mut self,
        gradients: HashMap<usize, Tensor>,
        retain_graph: bool,
    ) -> CoreResult<()> {
        // Batching des mises à jour de gradients
        let mut updates: Vec<(Arc<RwLock<VariableData>>, Tensor)> = Vec::with_capacity(gradients.len());
        
        // Collecter toutes les mises à jour
        for (var_id, gradient) in gradients {
            if var_id == self.id() {
                // Mise à jour de cette variable
                updates.push((Arc::clone(&self.data), gradient));
            }
            // TODO: Gérer les autres variables du graphe
        }
        
        // Appliquer les mises à jour en batch
        for (var_data_ref, new_grad) in updates {
            let mut var_data = var_data_ref.write().unwrap();
            
            // Accumulation optimisée du gradient
            if let Some(ref mut existing_grad) = var_data.grad {
                *existing_grad = existing_grad.clone().add(new_grad)?;
            } else {
                var_data.grad = Some(new_grad);
            }
            
            // Appliquer les hooks si présents
            self.apply_gradient_hooks_optimized(&mut var_data)?;
        }
        
        // Nettoyer le graphe si retain_graph est false
        if !retain_graph {
            self.clear_graph_optimized()?;
        }
        
        Ok(())
    }
    
    /// Application optimisée des hooks de gradient
    fn apply_gradient_hooks_optimized(
        &self,
        var_data: &mut std::sync::RwLockWriteGuard<VariableData>
    ) -> CoreResult<()> {
        if let Some(ref grad) = var_data.grad {
            let mut current_grad = grad.clone();
            
            // Appliquer tous les hooks en une seule passe
            for hook_fn in &var_data.hooks {
                current_grad = hook_fn(&current_grad);
            }
            
            // Mettre à jour le gradient final
            var_data.grad = Some(current_grad);
        }
        Ok(())
    }
    
    /// Nettoyage optimisé du graphe
    fn clear_graph_optimized(&mut self) -> CoreResult<()> {
        // Implementation simplifiée pour l'instant
        // TODO: Implémenter un nettoyage plus sophistiqué avec tracking des références
        
        let mut data = self.data.write().unwrap();
        
        // Nettoyer le grad_fn si c'est un nœud intermédiaire
        if !data.is_leaf {
            data.grad_fn = None;
        }
        
        Ok(())
    }
}

/// Utilitaires pour l'analyse de performance
pub struct BackwardPassProfiler {
    nodes_processed: usize,
    gradients_computed: usize,
    cache_hits: usize,
    cache_misses: usize,
    memory_allocated: usize,
    start_time: std::time::Instant,
}

impl BackwardPassProfiler {
    pub fn new() -> Self {
        Self {
            nodes_processed: 0,
            gradients_computed: 0,
            cache_hits: 0,
            cache_misses: 0,
            memory_allocated: 0,
            start_time: std::time::Instant::now(),
        }
    }
    
    pub fn record_node_processed(&mut self) {
        self.nodes_processed += 1;
    }
    
    pub fn record_gradient_computed(&mut self) {
        self.gradients_computed += 1;
    }
    
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }
    
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }
    
    pub fn record_memory_allocation(&mut self, size: usize) {
        self.memory_allocated += size;
    }
    
    pub fn get_report(&self) -> BackwardPassReport {
        let elapsed = self.start_time.elapsed();
        let cache_hit_rate = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        } else {
            0.0
        };
        
        BackwardPassReport {
            nodes_processed: self.nodes_processed,
            gradients_computed: self.gradients_computed,
            cache_hit_rate,
            memory_allocated: self.memory_allocated,
            elapsed_time: elapsed,
            nodes_per_second: self.nodes_processed as f64 / elapsed.as_secs_f64(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackwardPassReport {
    pub nodes_processed: usize,
    pub gradients_computed: usize,
    pub cache_hit_rate: f64,
    pub memory_allocated: usize,
    pub elapsed_time: std::time::Duration,
    pub nodes_per_second: f64,
}

impl std::fmt::Display for BackwardPassReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, 
            "Backward Pass Report:\n\
             - Nodes processed: {}\n\
             - Gradients computed: {}\n\
             - Cache hit rate: {:.2}%\n\
             - Memory allocated: {} bytes\n\
             - Elapsed time: {:.2}ms\n\
             - Nodes per second: {:.0}",
            self.nodes_processed,
            self.gradients_computed,
            self.cache_hit_rate * 100.0,
            self.memory_allocated,
            self.elapsed_time.as_millis(),
            self.nodes_per_second
        )
    }
}

/// Configuration pour le profiling du backward pass
thread_local! {
    static PROFILER_ENABLED: std::cell::Cell<bool> = std::cell::Cell::new(false);
    static CURRENT_PROFILER: std::cell::RefCell<Option<BackwardPassProfiler>> = std::cell::RefCell::new(None);
}

/// Active le profiling pour le prochain backward pass
pub fn enable_backward_profiling() {
    PROFILER_ENABLED.with(|enabled| enabled.set(true));
    CURRENT_PROFILER.with(|profiler| {
        *profiler.borrow_mut() = Some(BackwardPassProfiler::new());
    });
}

/// Désactive le profiling et retourne le rapport
pub fn get_backward_profile() -> Option<BackwardPassReport> {
    PROFILER_ENABLED.with(|enabled| enabled.set(false));
    CURRENT_PROFILER.with(|profiler| {
        profiler.borrow_mut().take().map(|p| p.get_report())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variable;
    use rustytorch_tensor::Tensor;

    #[test]
    fn test_optimized_backward_basic() {
        let tensor = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], None);
        let mut x = Variable::from_tensor(tensor, true);
        
        let y = x.mul(&x);
        let mut z = y.sum();
        
        // Test backward optimisé
        z.backward_optimized(None, false, false).unwrap();
        
        // Vérifier que le gradient a été calculé
        assert!(x.grad().is_some());
    }
    
    #[test]
    fn test_backward_profiling() {
        enable_backward_profiling();
        
        let tensor = Tensor::from_data(&[1.0, 2.0], vec![2], None);
        let mut x = Variable::from_tensor(tensor, true);
        
        let mut y = x.mul(&x);
        y.backward_optimized(None, false, false).unwrap();
        
        let report = get_backward_profile();
        assert!(report.is_some());
        
        let report = report.unwrap();
        assert!(report.nodes_processed > 0);
    }
    
    #[test]
    fn test_profiler_report_display() {
        let profiler = BackwardPassProfiler::new();
        let report = profiler.get_report();
        let display_str = format!("{}", report);
        
        assert!(display_str.contains("Backward Pass Report"));
        assert!(display_str.contains("Nodes processed"));
        assert!(display_str.contains("Cache hit rate"));
    }
}