//! Détection d'anomalies et debugging avancé pour l'autograd
//! 
//! Ce module fournit des outils pour:
//! - Détecter les NaN et Infinity dans les gradients
//! - Tracer le flux de gradients dans le graphe
//! - Identifier les sources d'anomalies
//! - Debugging interactif du graphe de calcul

use crate::{Variable, VariableData, OptimizedNode, Operation};
use rustytorch_tensor::Tensor;
use rustytorch_core::Result as CoreResult;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock, Weak};
use std::fmt;

/// Configuration pour la détection d'anomalies
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Activer la détection de NaN
    pub detect_nan: bool,
    /// Activer la détection d'infini
    pub detect_inf: bool,
    /// Activer le tracing des gradients
    pub enable_gradient_tracing: bool,
    /// Seuil pour détecter les gradients explosifs
    pub gradient_explosion_threshold: f64,
    /// Seuil pour détecter les gradients qui disparaissent
    pub gradient_vanishing_threshold: f64,
    /// Garder l'historique des anomalies
    pub keep_anomaly_history: bool,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            detect_nan: true,
            detect_inf: true,
            enable_gradient_tracing: false,
            gradient_explosion_threshold: 1e6,
            gradient_vanishing_threshold: 1e-7,
            keep_anomaly_history: true,
        }
    }
}

/// Types d'anomalies détectées
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    /// NaN détecté dans le gradient
    NaN,
    /// Infini positif détecté
    PositiveInfinity,
    /// Infini négatif détecté
    NegativeInfinity,
    /// Gradient explosif (trop grand)
    GradientExplosion,
    /// Gradient qui disparaît (trop petit)
    GradientVanishing,
    /// Division par zéro détectée
    DivisionByZero,
    /// Gradient non initialisé
    UninitializedGradient,
}

impl fmt::Display for AnomalyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnomalyType::NaN => write!(f, "NaN detected"),
            AnomalyType::PositiveInfinity => write!(f, "Positive infinity detected"),
            AnomalyType::NegativeInfinity => write!(f, "Negative infinity detected"),
            AnomalyType::GradientExplosion => write!(f, "Gradient explosion detected"),
            AnomalyType::GradientVanishing => write!(f, "Gradient vanishing detected"),
            AnomalyType::DivisionByZero => write!(f, "Division by zero detected"),
            AnomalyType::UninitializedGradient => write!(f, "Uninitialized gradient detected"),
        }
    }
}

/// Information sur une anomalie détectée
#[derive(Debug, Clone)]
pub struct AnomalyInfo {
    /// Type d'anomalie
    pub anomaly_type: AnomalyType,
    /// ID de la variable où l'anomalie a été détectée
    pub variable_id: usize,
    /// Nom de l'opération qui a causé l'anomalie
    pub operation: Operation,
    /// Valeur du gradient au moment de l'anomalie
    pub gradient_value: Option<f64>,
    /// Forme du tenseur
    pub tensor_shape: Vec<usize>,
    /// Timestamp de la détection
    pub timestamp: std::time::Instant,
    /// Trace de la pile (si disponible)
    pub stack_trace: String,
}

impl fmt::Display for AnomalyInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, 
            "Anomaly: {} at variable {} (op: {:?}, shape: {:?})",
            self.anomaly_type,
            self.variable_id,
            self.operation,
            self.tensor_shape
        )?;
        
        if let Some(value) = self.gradient_value {
            write!(f, ", gradient value: {}", value)?;
        }
        
        Ok(())
    }
}

/// Détecteur d'anomalies principal
pub struct AnomalyDetector {
    config: AnomalyConfig,
    anomalies: Vec<AnomalyInfo>,
    gradient_traces: HashMap<usize, Vec<GradientTrace>>,
    enabled: bool,
}

/// Trace d'un gradient à travers le graphe
#[derive(Debug, Clone)]
pub struct GradientTrace {
    /// ID de la variable
    pub variable_id: usize,
    /// Opération qui a produit ce gradient
    pub operation: Operation,
    /// Valeur du gradient (norm L2)
    pub gradient_norm: f64,
    /// Timestamp
    pub timestamp: std::time::Instant,
    /// Variables d'entrée qui ont contribué
    pub input_variables: Vec<usize>,
}

impl AnomalyDetector {
    pub fn new(config: AnomalyConfig) -> Self {
        Self {
            config,
            anomalies: Vec::new(),
            gradient_traces: HashMap::new(),
            enabled: true,
        }
    }

    /// Active ou désactive la détection
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Vérifie un tenseur pour les anomalies
    pub fn check_tensor(&mut self, 
                       tensor: &Tensor, 
                       variable_id: usize, 
                       operation: Operation) -> CoreResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let data = tensor.storage().to_vec_f64();
        
        for (i, &value) in data.iter().enumerate() {
            // Vérification NaN
            if self.config.detect_nan && value.is_nan() {
                self.record_anomaly(AnomalyInfo {
                    anomaly_type: AnomalyType::NaN,
                    variable_id,
                    operation: operation.clone(),
                    gradient_value: Some(value),
                    tensor_shape: tensor.shape().to_vec(),
                    timestamp: std::time::Instant::now(),
                    stack_trace: self.get_stack_trace(),
                });
            }

            // Vérification Infinity
            if self.config.detect_inf && value.is_infinite() {
                let anomaly_type = if value.is_sign_positive() {
                    AnomalyType::PositiveInfinity
                } else {
                    AnomalyType::NegativeInfinity
                };
                
                self.record_anomaly(AnomalyInfo {
                    anomaly_type,
                    variable_id,
                    operation: operation.clone(),
                    gradient_value: Some(value),
                    tensor_shape: tensor.shape().to_vec(),
                    timestamp: std::time::Instant::now(),
                    stack_trace: self.get_stack_trace(),
                });
            }

            // Vérification gradient explosif
            if value.abs() > self.config.gradient_explosion_threshold {
                self.record_anomaly(AnomalyInfo {
                    anomaly_type: AnomalyType::GradientExplosion,
                    variable_id,
                    operation: operation.clone(),
                    gradient_value: Some(value),
                    tensor_shape: tensor.shape().to_vec(),
                    timestamp: std::time::Instant::now(),
                    stack_trace: self.get_stack_trace(),
                });
            }

            // Vérification gradient qui disparaît
            if value.abs() > 0.0 && value.abs() < self.config.gradient_vanishing_threshold {
                self.record_anomaly(AnomalyInfo {
                    anomaly_type: AnomalyType::GradientVanishing,
                    variable_id,
                    operation: operation.clone(),
                    gradient_value: Some(value),
                    tensor_shape: tensor.shape().to_vec(),
                    timestamp: std::time::Instant::now(),
                    stack_trace: self.get_stack_trace(),
                });
                break; // Un seul warning par tenseur pour le vanishing
            }
        }

        // Traçage des gradients si activé
        if self.config.enable_gradient_tracing {
            self.trace_gradient(tensor, variable_id, operation)?;
        }

        Ok(())
    }

    /// Enregistre une anomalie
    fn record_anomaly(&mut self, anomaly: AnomalyInfo) {
        if self.config.keep_anomaly_history {
            self.anomalies.push(anomaly.clone());
        }
        
        // Affichage immédiat de l'anomalie
        eprintln!("🚨 ANOMALY DETECTED: {}", anomaly);
        
        if !anomaly.stack_trace.is_empty() {
            eprintln!("Stack trace: {}", anomaly.stack_trace);
        }
    }

    /// Trace le gradient à travers le graphe
    fn trace_gradient(&mut self, 
                     tensor: &Tensor, 
                     variable_id: usize, 
                     operation: Operation) -> CoreResult<()> {
        let data = tensor.storage().to_vec_f64();
        let gradient_norm = (data.iter().map(|x| x * x).sum::<f64>()).sqrt();
        
        let trace = GradientTrace {
            variable_id,
            operation,
            gradient_norm,
            timestamp: std::time::Instant::now(),
            input_variables: Vec::new(), // TODO: récupérer depuis le graphe
        };
        
        self.gradient_traces
            .entry(variable_id)
            .or_insert_with(Vec::new)
            .push(trace);
        
        Ok(())
    }

    /// Obtient une trace de la pile (simplifié)
    fn get_stack_trace(&self) -> String {
        // Pour une vraie implémentation, on utiliserait `backtrace` crate
        format!("at {}:{}", file!(), line!())
    }

    /// Retourne toutes les anomalies détectées
    pub fn get_anomalies(&self) -> &[AnomalyInfo] {
        &self.anomalies
    }

    /// Nettoie l'historique des anomalies
    pub fn clear_anomalies(&mut self) {
        self.anomalies.clear();
        self.gradient_traces.clear();
    }

    /// Retourne les traces de gradient pour une variable
    pub fn get_gradient_traces(&self, variable_id: usize) -> Option<&[GradientTrace]> {
        self.gradient_traces.get(&variable_id).map(|v| v.as_slice())
    }

    /// Génère un rapport de toutes les anomalies
    pub fn generate_report(&self) -> AnomalyReport {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut explosion_count = 0;
        let mut vanishing_count = 0;
        let mut other_count = 0;

        for anomaly in &self.anomalies {
            match anomaly.anomaly_type {
                AnomalyType::NaN => nan_count += 1,
                AnomalyType::PositiveInfinity | AnomalyType::NegativeInfinity => inf_count += 1,
                AnomalyType::GradientExplosion => explosion_count += 1,
                AnomalyType::GradientVanishing => vanishing_count += 1,
                _ => other_count += 1,
            }
        }

        AnomalyReport {
            total_anomalies: self.anomalies.len(),
            nan_count,
            inf_count,
            explosion_count,
            vanishing_count,
            other_count,
            variables_with_traces: self.gradient_traces.len(),
            recent_anomalies: self.anomalies.iter()
                .rev()
                .take(5)
                .cloned()
                .collect(),
        }
    }
}

/// Rapport sur les anomalies détectées
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    pub total_anomalies: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    pub explosion_count: usize,
    pub vanishing_count: usize,
    pub other_count: usize,
    pub variables_with_traces: usize,
    pub recent_anomalies: Vec<AnomalyInfo>,
}

impl fmt::Display for AnomalyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Anomaly Detection Report ===")?;
        writeln!(f, "Total anomalies detected: {}", self.total_anomalies)?;
        writeln!(f, "  - NaN: {}", self.nan_count)?;
        writeln!(f, "  - Infinity: {}", self.inf_count)?;
        writeln!(f, "  - Gradient explosion: {}", self.explosion_count)?;
        writeln!(f, "  - Gradient vanishing: {}", self.vanishing_count)?;
        writeln!(f, "  - Other: {}", self.other_count)?;
        writeln!(f, "Variables with gradient traces: {}", self.variables_with_traces)?;
        
        if !self.recent_anomalies.is_empty() {
            writeln!(f, "\nRecent anomalies:")?;
            for anomaly in &self.recent_anomalies {
                writeln!(f, "  - {}", anomaly)?;
            }
        }
        
        Ok(())
    }
}

/// Analyseur de flux de gradients
pub struct GradientFlowAnalyzer {
    flow_graph: HashMap<usize, Vec<usize>>,
    gradient_magnitudes: HashMap<usize, f64>,
}

impl GradientFlowAnalyzer {
    pub fn new() -> Self {
        Self {
            flow_graph: HashMap::new(),
            gradient_magnitudes: HashMap::new(),
        }
    }

    /// Analyse le flux de gradients dans un graphe
    pub fn analyze_flow(&mut self, root: &Variable) -> CoreResult<GradientFlowReport> {
        let mut visited = HashSet::new();
        let mut flow_paths = Vec::new();
        
        self.traverse_gradient_flow(root, &mut visited, &mut Vec::new(), &mut flow_paths)?;
        
        Ok(GradientFlowReport {
            total_variables: visited.len(),
            flow_paths,
            bottlenecks: self.identify_bottlenecks(),
            vanishing_paths: self.identify_vanishing_paths(),
        })
    }

    /// Traverse le graphe pour analyser le flux
    fn traverse_gradient_flow(
        &mut self,
        var: &Variable,
        visited: &mut HashSet<usize>,
        current_path: &mut Vec<usize>,
        flow_paths: &mut Vec<Vec<usize>>,
    ) -> CoreResult<()> {
        let var_id = var.id();
        
        if visited.contains(&var_id) {
            return Ok(());
        }
        
        visited.insert(var_id);
        current_path.push(var_id);
        
        // Enregistrer la magnitude du gradient
        if let Some(grad) = var.grad() {
            let data = grad.storage().to_vec_f64();
            let magnitude = (data.iter().map(|x| x * x).sum::<f64>()).sqrt();
            self.gradient_magnitudes.insert(var_id, magnitude);
        }
        
        // Si c'est une feuille, on a terminé ce chemin
        let data = var.data.read().unwrap();
        if data.is_leaf || data.grad_fn.is_none() {
            flow_paths.push(current_path.clone());
        } else {
            // Continuer vers les inputs
            if let Some(ref node) = data.grad_fn {
                let mut input_ids = Vec::new();
                for weak_input in &node.inputs {
                    if let Some(input_data) = weak_input.upgrade() {
                        let input_data_guard = input_data.read().unwrap();
                        input_ids.push(input_data_guard.id);
                    }
                }
                self.flow_graph.insert(var_id, input_ids);
            }
        }
        
        current_path.pop();
        Ok(())
    }

    /// Identifie les goulots d'étranglement dans le flux
    fn identify_bottlenecks(&self) -> Vec<usize> {
        let mut bottlenecks = Vec::new();
        
        for (&var_id, &magnitude) in &self.gradient_magnitudes {
            if magnitude < 1e-6 { // Seuil pour un goulot d'étranglement
                bottlenecks.push(var_id);
            }
        }
        
        bottlenecks
    }

    /// Identifie les chemins où les gradients disparaissent
    fn identify_vanishing_paths(&self) -> Vec<Vec<usize>> {
        let mut vanishing_paths = Vec::new();
        
        // TODO: Implémenter la logique pour identifier les chemins vanishing
        // Analyser les chemins où le gradient diminue drastiquement
        
        vanishing_paths
    }
}

/// Rapport d'analyse du flux de gradients
#[derive(Debug, Clone)]
pub struct GradientFlowReport {
    pub total_variables: usize,
    pub flow_paths: Vec<Vec<usize>>,
    pub bottlenecks: Vec<usize>,
    pub vanishing_paths: Vec<Vec<usize>>,
}

impl fmt::Display for GradientFlowReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Gradient Flow Analysis ===")?;
        writeln!(f, "Total variables analyzed: {}", self.total_variables)?;
        writeln!(f, "Flow paths found: {}", self.flow_paths.len())?;
        writeln!(f, "Bottlenecks detected: {}", self.bottlenecks.len())?;
        writeln!(f, "Vanishing gradient paths: {}", self.vanishing_paths.len())?;
        
        if !self.bottlenecks.is_empty() {
            writeln!(f, "\nBottleneck variables: {:?}", self.bottlenecks)?;
        }
        
        Ok(())
    }
}

/// Interface globale pour la détection d'anomalies
thread_local! {
    static GLOBAL_DETECTOR: std::cell::RefCell<Option<AnomalyDetector>> = std::cell::RefCell::new(None);
}

/// Active la détection d'anomalies globale
pub fn enable_anomaly_detection(config: Option<AnomalyConfig>) {
    let config = config.unwrap_or_default();
    GLOBAL_DETECTOR.with(|detector| {
        *detector.borrow_mut() = Some(AnomalyDetector::new(config));
    });
}

/// Désactive la détection d'anomalies globale
pub fn disable_anomaly_detection() {
    GLOBAL_DETECTOR.with(|detector| {
        *detector.borrow_mut() = None;
    });
}

/// Vérifie un tenseur avec le détecteur global
pub fn check_tensor_globally(tensor: &Tensor, variable_id: usize, operation: Operation) -> CoreResult<()> {
    GLOBAL_DETECTOR.with(|detector| {
        if let Some(ref mut det) = *detector.borrow_mut() {
            det.check_tensor(tensor, variable_id, operation)
        } else {
            Ok(())
        }
    })
}

/// Obtient le rapport global des anomalies
pub fn get_global_anomaly_report() -> Option<AnomalyReport> {
    GLOBAL_DETECTOR.with(|detector| {
        detector.borrow().as_ref().map(|det| det.generate_report())
    })
}

/// Nettoie les anomalies globales
pub fn clear_global_anomalies() {
    GLOBAL_DETECTOR.with(|detector| {
        if let Some(ref mut det) = *detector.borrow_mut() {
            det.clear_anomalies();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variable;
    use rustytorch_tensor::Tensor;

    #[test]
    fn test_anomaly_detection_nan() {
        let mut detector = AnomalyDetector::new(AnomalyConfig::default());
        
        // Créer un tenseur avec NaN
        let data = vec![1.0, f64::NAN, 3.0];
        let tensor = Tensor::from_data(&data, vec![3], None);
        
        detector.check_tensor(&tensor, 1, Operation::Add).unwrap();
        
        let anomalies = detector.get_anomalies();
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::NaN);
    }

    #[test]
    fn test_anomaly_detection_infinity() {
        let mut detector = AnomalyDetector::new(AnomalyConfig::default());
        
        // Créer un tenseur avec Infinity
        let data = vec![1.0, f64::INFINITY, 3.0];
        let tensor = Tensor::from_data(&data, vec![3], None);
        
        detector.check_tensor(&tensor, 1, Operation::Div).unwrap();
        
        let anomalies = detector.get_anomalies();
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::PositiveInfinity);
    }

    #[test]
    fn test_gradient_explosion_detection() {
        let mut config = AnomalyConfig::default();
        config.gradient_explosion_threshold = 10.0;
        let mut detector = AnomalyDetector::new(config);
        
        // Créer un tenseur avec valeur explosive
        let data = vec![1.0, 100.0, 3.0];
        let tensor = Tensor::from_data(&data, vec![3], None);
        
        detector.check_tensor(&tensor, 1, Operation::Mul).unwrap();
        
        let anomalies = detector.get_anomalies();
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::GradientExplosion);
    }

    #[test]
    fn test_global_anomaly_detection() {
        enable_anomaly_detection(None);
        
        let data = vec![1.0, f64::NAN, 3.0];
        let tensor = Tensor::from_data(&data, vec![3], None);
        
        check_tensor_globally(&tensor, 1, Operation::Add).unwrap();
        
        let report = get_global_anomaly_report().unwrap();
        assert_eq!(report.total_anomalies, 1);
        assert_eq!(report.nan_count, 1);
        
        clear_global_anomalies();
        disable_anomaly_detection();
    }

    #[test]
    fn test_anomaly_report_display() {
        let mut detector = AnomalyDetector::new(AnomalyConfig::default());
        
        let data = vec![f64::NAN, f64::INFINITY];
        let tensor = Tensor::from_data(&data, vec![2], None);
        
        detector.check_tensor(&tensor, 1, Operation::Add).unwrap();
        
        let report = detector.generate_report();
        let display_str = format!("{}", report);
        
        assert!(display_str.contains("Anomaly Detection Report"));
        assert!(display_str.contains("Total anomalies"));
    }

    #[test]
    fn test_gradient_flow_analyzer() {
        let analyzer = GradientFlowAnalyzer::new();
        
        // Test basic functionality
        assert_eq!(analyzer.flow_graph.len(), 0);
        assert_eq!(analyzer.gradient_magnitudes.len(), 0);
    }
}