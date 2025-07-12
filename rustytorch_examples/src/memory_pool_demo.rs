// rustytorch_examples/src/memory_pool_demo.rs
// Démonstration du système de memory pools pour optimisation

use rustytorch_core::{DType, Device};
use rustytorch_tensor::memory_pool::{
    allocate_pooled, clear_memory_pools, memory_pool_stats, MemoryPoolManager, PoolConfig,
    PoolStatistics,
};
use std::time::Instant;

pub fn run_memory_pool_demo() {
    println!("🧠 Démonstration du Memory Pool System RustyTorch\n");

    // === Configuration du pool ===
    println!("⚙️ Configuration du memory pool:");

    let config = PoolConfig {
        max_pool_size: 256 * 1024 * 1024, // 256 MB
        max_age_seconds: 300,             // 5 minutes
        enable_defragmentation: true,
        alignment: 64, // Cache line alignment
        growth_factor: 1.5,
        bypass_small_allocs: true,        // Enable bypass for performance test
        bypass_threshold: 4096,           // 4KB threshold
    };

    println!(
        "• Taille max du pool: {} MB",
        config.max_pool_size / (1024 * 1024)
    );
    println!("• Âge max des blocs: {} secondes", config.max_age_seconds);
    println!("• Défragmentation: {}", config.enable_defragmentation);
    println!("• Alignement: {} bytes", config.alignment);
    println!("• Facteur de croissance: {}", config.growth_factor);

    // === Création du manager ===
    println!("\n📋 Création du memory pool manager:");
    let manager = MemoryPoolManager::new(config.clone());
    println!("✓ Manager créé avec configuration personnalisée");

    // === Allocation basique ===
    println!("\n🔧 Allocations basiques:");

    // Premières allocations (cache miss)
    let start = Instant::now();
    let ptr1 = manager.allocate(1024, Device::Cpu, DType::Float32).unwrap();
    let alloc1_time = start.elapsed();
    println!("• Première allocation 1KB: {:?} (cache miss)", alloc1_time);

    let ptr2 = manager.allocate(2048, Device::Cpu, DType::Float32).unwrap();
    println!("• Allocation 2KB: succès");

    let ptr3 = manager.allocate(4096, Device::Cpu, DType::Float32).unwrap();
    println!("• Allocation 4KB: succès");

    // Libération
    manager.deallocate(ptr1, 1024, Device::Cpu, DType::Float32);
    manager.deallocate(ptr2, 2048, Device::Cpu, DType::Float32);
    println!("• Libération des blocs: succès");

    // === Test de réutilisation (cache hit) ===
    println!("\n♻️ Test de réutilisation (cache hit):");

    let start = Instant::now();
    let ptr4 = manager.allocate(1024, Device::Cpu, DType::Float32).unwrap();
    let reuse_time = start.elapsed();
    println!("• Réallocation 1KB: {:?} (cache hit)", reuse_time);

    if reuse_time < alloc1_time {
        println!("✓ Réutilisation plus rapide que allocation initiale!");
    }

    // === Test avec différents types ===
    println!("\n🔀 Test avec différents devices/dtypes:");

    // CPU F32
    let cpu_f32 = manager.allocate(8192, Device::Cpu, DType::Float32).unwrap();
    println!("• CPU F32 8KB: succès");

    // CPU F64 (différent pool)
    let cpu_f64 = manager.allocate(8192, Device::Cpu, DType::Float64).unwrap();
    println!("• CPU F64 8KB: succès (pool séparé)");

    // Test CUDA (si disponible - simulé ici)
    println!("• CUDA pools: disponibles pour allocation future");

    // === Statistiques en temps réel ===
    println!("\n📊 Statistiques du pool:");
    let stats = manager.global_statistics();
    print_pool_statistics(&stats);

    // === Test de performance ===
    println!("\n🏃 Test de performance (1000 allocations):");

    // Sans pool (allocation système directe)
    let start = Instant::now();
    let mut system_ptrs = Vec::new();
    for _ in 0..1000 {
        let layout = std::alloc::Layout::from_size_align(1024, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        system_ptrs.push((ptr, layout));
    }
    let system_time = start.elapsed();

    // Nettoyage
    for (ptr, layout) in system_ptrs {
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }

    // Avec pool
    let start = Instant::now();
    let mut pool_ptrs = Vec::new();
    for _ in 0..1000 {
        let ptr = manager.allocate(1024, Device::Cpu, DType::Float32).unwrap();
        pool_ptrs.push(ptr);
    }
    let pool_time = start.elapsed();

    // Nettoyage
    for ptr in pool_ptrs {
        manager.deallocate(ptr, 1024, Device::Cpu, DType::Float32);
    }

    println!("• Allocation système: {:?}", system_time);
    println!("• Allocation pool: {:?}", pool_time);

    let speedup = system_time.as_nanos() as f64 / pool_time.as_nanos() as f64;
    println!("• Accélération: {:.2}x", speedup);

    // === Test avec API de haut niveau ===
    println!("\n🎯 Test API haut niveau (PooledMemory):");

    {
        let pooled1 = allocate_pooled(16384, Device::Cpu, DType::Float32).unwrap();
        let pooled2 = allocate_pooled(32768, Device::Cpu, DType::Float64).unwrap();

        println!(
            "• PooledMemory 16KB: ptr={:?}, size={}",
            pooled1.as_ptr(),
            pooled1.size()
        );
        println!(
            "• PooledMemory 32KB: ptr={:?}, size={}",
            pooled2.as_ptr(),
            pooled2.size()
        );

        // Les blocs seront automatiquement retournés au pool à la fin du scope
        println!("• Libération automatique à la fin du scope");
    }

    println!("✓ Blocs automatiquement retournés au pool");

    // === Test de fragmentation ===
    println!("\n🧩 Test anti-fragmentation:");

    // Allocation de tailles variées
    let mut mixed_ptrs = Vec::new();
    let sizes = vec![512, 1024, 2048, 4096, 8192];

    for &size in &sizes {
        for _ in 0..10 {
            let ptr = manager.allocate(size, Device::Cpu, DType::Float32).unwrap();
            mixed_ptrs.push((ptr, size));
        }
    }

    println!("• Alloué {} blocs de tailles variées", mixed_ptrs.len());

    // Libération partielle (créer fragmentation)
    for i in (0..mixed_ptrs.len()).step_by(2) {
        let (ptr, size) = mixed_ptrs[i];
        manager.deallocate(ptr, size, Device::Cpu, DType::Float32);
    }

    println!("• Libéré 50% des blocs (fragmentation créée)");

    // Test allocation après fragmentation
    let large_ptr = manager.allocate(65536, Device::Cpu, DType::Float32);
    match large_ptr {
        Ok(_) => println!("✓ Allocation large réussie malgré fragmentation"),
        Err(_) => println!("⚠ Allocation large échouée (fragmentation)"),
    }

    // === Statistiques finales ===
    println!("\n📈 Statistiques finales:");
    let final_stats = memory_pool_stats();
    print_pool_statistics(&final_stats);

    // === Patterns d'utilisation recommandés ===
    println!("\n💡 Patterns d'utilisation recommandés:");
    println!("• Deep Learning:");
    println!("  - Pré-allouer pools pour activations/gradients");
    println!("  - Réutiliser mémoire entre batches");
    println!("  - Séparer pools par device (CPU/GPU)");

    println!("\n• Tenseurs temporaires:");
    println!("  - Utiliser PooledMemory pour RAII");
    println!("  - Tailles power-of-2 pour meilleure réutilisation");
    println!("  - Aligner sur cache lines (64 bytes)");

    println!("\n• Optimisation multi-thread:");
    println!("  - Un pool par thread pour éviter contention");
    println!("  - Lock-free allocation paths");
    println!("  - Batch deallocation");

    // === Configuration adaptée par cas d'usage ===
    println!("\n⚡ Configurations optimales par cas d'usage:");

    // Configuration pour training
    println!("• Deep Learning Training:");
    let training_config = PoolConfig {
        max_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
        max_age_seconds: 600,                  // 10 minutes
        enable_defragmentation: true,
        alignment: 256,     // GPU alignment
        growth_factor: 2.0, // Croissance agressive
        bypass_small_allocs: true,             // Enable bypass for performance
        bypass_threshold: 1024,                // 1KB threshold
    };
    println!(
        "  - Pool size: {}GB",
        training_config.max_pool_size / (1024 * 1024 * 1024)
    );
    println!("  - Max age: {}min", training_config.max_age_seconds / 60);
    println!(
        "  - Alignment: {} bytes (GPU optimal)",
        training_config.alignment
    );

    // Configuration pour inference
    println!("\n• Inference/Production:");
    let inference_config = PoolConfig {
        max_pool_size: 512 * 1024 * 1024, // 512MB
        max_age_seconds: 60,              // 1 minute
        enable_defragmentation: false,    // Latence prévisible
        alignment: 64,                    // CPU optimal
        growth_factor: 1.25,              // Croissance conservative
        bypass_small_allocs: true,        // Maximum performance
        bypass_threshold: 2048,           // 2KB threshold
    };
    println!(
        "  - Pool size: {}MB",
        inference_config.max_pool_size / (1024 * 1024)
    );
    println!("  - Max age: {}s", inference_config.max_age_seconds);
    println!(
        "  - Defrag: {} (latence prévisible)",
        inference_config.enable_defragmentation
    );

    // === Métriques avancées ===
    println!("\n📐 Métriques avancées disponibles:");
    println!("• Cache hit ratio: mesure efficacité réutilisation");
    println!("• Peak memory usage: dimensionnement optimal");
    println!("• Allocation count: détection fuites mémoire");
    println!("• Defragmentation frequency: optimisation layout");

    // === Intégration avec tenseurs ===
    println!("\n🔗 Intégration future avec Tensor:");
    println!("• TensorOptions::use_memory_pool: bool");
    println!("• Tensor::with_pool(pool_id): Self");
    println!("• Automatic pool selection par device");
    println!("• Pool-aware tensor operations");

    // === Nettoyage final ===
    println!("\n🧹 Nettoyage des pools:");
    clear_memory_pools();
    println!("✓ Tous les pools ont été vidés");

    let empty_stats = memory_pool_stats();
    println!("• Allocations restantes: {}", empty_stats.total_allocations);

    println!("\n✅ Démonstration Memory Pool terminée !");
    println!("🏗️ Système implémenté:");
    println!("   • DeviceMemoryPool - pool par device/dtype");
    println!("   • MemoryPoolManager - gestion globale");
    println!("   • PooledMemory - RAII smart pointer");
    println!("   • Bucket allocation - power-of-2 sizes");
    println!("   • Anti-fragmentation - compaction/cleanup");
    println!("   • Statistics tracking - performance monitoring");
    println!("   • Configuration flexible - adaptable par use case");
    println!("   • Thread-safe - parallel access");
}

/// Helper pour afficher les statistiques de pool
fn print_pool_statistics(stats: &PoolStatistics) {
    println!("  Total allocations: {}", stats.total_allocations);
    println!("  Cache hits: {}", stats.cache_hits);
    println!("  Cache misses: {}", stats.cache_misses);

    if stats.total_allocations > 0 {
        let hit_ratio = (stats.cache_hits as f64 / stats.total_allocations as f64) * 100.0;
        println!("  Cache hit ratio: {:.1}%", hit_ratio);
    }

    println!("  Defragmentations: {}", stats.defragmentations);
    println!("  Peak memory usage: {} KB", stats.peak_memory_usage / 1024);
}

/// Démonstration des patterns avancés
pub fn demo_advanced_patterns() {
    println!("\n🚀 Patterns avancés de memory management:");

    // Pattern 1: Batch processing
    println!("\n1️⃣ Batch Processing Pattern:");
    let manager = MemoryPoolManager::new(PoolConfig::default());

    // Pré-allocation pour batch
    let batch_size = 32;
    let input_size = 1024 * 1024; // 1MB per sample

    let mut batch_ptrs = Vec::new();
    for i in 0..batch_size {
        let ptr = manager
            .allocate(input_size, Device::Cpu, DType::Float32)
            .unwrap();
        batch_ptrs.push(ptr);
        if i % 8 == 0 {
            print!(".");
        }
    }
    println!("\n✓ Batch de {} samples pré-alloué", batch_size);

    // Processing simulation
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Libération batch
    for ptr in batch_ptrs {
        manager.deallocate(ptr, input_size, Device::Cpu, DType::Float32);
    }
    println!("✓ Batch processing terminé");

    // Pattern 2: Gradient accumulation
    println!("\n2️⃣ Gradient Accumulation Pattern:");

    let accumulation_steps = 4;
    let grad_size = 512 * 1024; // 512KB gradients

    // Allocation persistante pour accumulation
    let accum_ptr = manager
        .allocate(grad_size, Device::Cpu, DType::Float32)
        .unwrap();
    println!("✓ Buffer d'accumulation alloué");

    // Simulation accumulation
    for step in 0..accumulation_steps {
        let temp_grad = manager
            .allocate(grad_size, Device::Cpu, DType::Float32)
            .unwrap();
        // Accumulate gradients (simulated)
        std::thread::sleep(std::time::Duration::from_millis(5));
        manager.deallocate(temp_grad, grad_size, Device::Cpu, DType::Float32);
        println!("  Step {}/{} processed", step + 1, accumulation_steps);
    }

    manager.deallocate(accum_ptr, grad_size, Device::Cpu, DType::Float32);
    println!("✓ Gradient accumulation terminé");

    // Pattern 3: Multi-device coordination
    println!("\n3️⃣ Multi-Device Pattern:");

    // Allocation sur différents devices
    let cpu_ptr = manager
        .allocate(1024 * 1024, Device::Cpu, DType::Float32)
        .unwrap();
    println!("✓ CPU allocation: 1MB");

    // Simulation transfer CPU -> GPU
    println!("→ Transfer vers GPU (simulé)");
    let gpu_ptr = manager
        .allocate(1024 * 1024, Device::Cpu, DType::Float32)
        .unwrap(); // Simulated GPU
    println!("✓ GPU allocation: 1MB");

    // Cleanup
    manager.deallocate(cpu_ptr, 1024 * 1024, Device::Cpu, DType::Float32);
    manager.deallocate(gpu_ptr, 1024 * 1024, Device::Cpu, DType::Float32);
    println!("✓ Multi-device cleanup terminé");

    println!("\n🎯 Patterns demonstrés avec succès!");
}
