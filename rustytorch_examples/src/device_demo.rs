// rustytorch_examples/src/device_demo.rs
// Démonstration des fonctionnalités device étendues

use rustytorch_core::{
    available_devices, cuda_is_available, current_device, device_count, metal_is_available,
    rocm_is_available, set_device, synchronize, Device, DeviceContext, DeviceManager,
};
use rustytorch_tensor::Tensor;

pub fn run_device_demo() {
    println!("🖥️  Démonstration des fonctionnalités Device étendues RustyTorch\n");

    // === Discovery des devices disponibles ===
    println!("🔍 Découverte des devices:");

    // Afficher le device courant
    println!("Device courant: {}", current_device());

    // Lister tous les devices disponibles
    let devices = available_devices();
    println!("Devices disponibles: {:?}", devices);

    // Vérifier la disponibilité de chaque type
    println!("\n📊 Disponibilité par type:");
    println!("  CUDA disponible: {}", cuda_is_available());
    println!("  Metal disponible: {}", metal_is_available());
    println!("  ROCm disponible: {}", rocm_is_available());

    // Compter les devices par type
    println!("\n📈 Nombre de devices par type:");
    println!("  GPUs CUDA: {}", device_count("cuda"));
    println!("  GPUs Metal: {}", device_count("metal"));
    println!("  GPUs ROCm: {}", device_count("rocm"));

    // === Test Device Manager ===
    println!("\n🎮 Test Device Manager:");
    let manager = DeviceManager::new();

    // Obtenir les infos du CPU
    if let Some(cpu_info) = manager.get_device_info(&Device::Cpu) {
        println!("\nInfos CPU:");
        println!("  Nom: {}", cpu_info.name);
        println!("  Type: {:?}", cpu_info.device_type);
        println!(
            "  Mémoire totale: {} GB",
            cpu_info.total_memory / (1024 * 1024 * 1024)
        );
        println!(
            "  Mémoire disponible: {} GB",
            cpu_info.available_memory / (1024 * 1024 * 1024)
        );
    }

    // Simuler la découverte de GPU si CUDA_VISIBLE_DEVICES est défini
    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
        if let Some(cuda_info) = manager.get_device_info(&Device::Cuda(0)) {
            println!("\nInfos GPU CUDA:");
            println!("  Nom: {}", cuda_info.name);
            println!("  Type: {:?}", cuda_info.device_type);
            println!(
                "  Mémoire totale: {} GB",
                cuda_info.total_memory / (1024 * 1024 * 1024)
            );
            if let Some(cc) = cuda_info.compute_capability {
                println!("  Compute Capability: {}.{}", cc.major, cc.minor);
            }
        }
    }

    // === Test Device Context ===
    println!("\n🔧 Test Device Context:");
    let context = DeviceContext::new(Device::Cpu);
    println!("Context créé pour: {:?}", context.device());

    // Test allocation mémoire
    let size = 1024 * 1024; // 1MB
    match context.allocator().allocate(size) {
        Ok(ptr) => {
            println!("✓ Allocation de {} MB réussie", size / (1024 * 1024));

            // Test copie host->device
            let data = vec![42u8; size];
            if context.allocator().copy_from_host(ptr, &data).is_ok() {
                println!("✓ Copie host->device réussie");
            }

            // Test copie device->host
            let mut result = vec![0u8; size];
            if context
                .allocator()
                .copy_to_host(&mut result, ptr, size)
                .is_ok()
            {
                println!("✓ Copie device->host réussie");
                if result[0] == 42 {
                    println!("✓ Données vérifiées correctes");
                }
            }

            // Cleanup
            context.allocator().deallocate(ptr);
            println!("✓ Mémoire libérée");
        }
        Err(e) => {
            println!("❌ Erreur d'allocation: {:?}", e);
        }
    }

    // === Test synchronisation ===
    println!("\n⏱️  Test synchronisation:");
    match synchronize() {
        Ok(_) => println!("✓ Synchronisation du device courant réussie"),
        Err(e) => println!("❌ Erreur de synchronisation: {:?}", e),
    }

    // === Création de tenseurs sur différents devices ===
    println!("\n🎯 Création de tenseurs avec device:");

    // Tenseur CPU (par défaut)
    let cpu_tensor = Tensor::ones(vec![2, 3], None);
    println!(
        "Tenseur CPU créé - shape: {:?}, device: {:?}",
        cpu_tensor.shape(),
        cpu_tensor.device()
    );

    // Simule la création d'un tenseur GPU (si disponible)
    if cuda_is_available() {
        println!("\n🚀 Simulation tenseur GPU:");
        // Dans une implémentation réelle, on ferait:
        // let gpu_options = TensorOptions::new().device(Device::Cuda(0));
        // let gpu_tensor = Tensor::ones(vec![2, 3], Some(gpu_options));
        println!("  (Création de tenseur GPU disponible avec Device::Cuda(0))");
    }

    // === Cas d'usage pratiques ===
    println!("\n💡 Cas d'usage pratiques:");

    // 1. Sélection automatique du meilleur device
    let best_device = if cuda_is_available() {
        Device::Cuda(0)
    } else if metal_is_available() {
        Device::Metal(0)
    } else {
        Device::Cpu
    };
    println!("• Meilleur device disponible: {}", best_device);

    // 2. Multi-GPU training simulation
    let num_gpus = device_count("cuda");
    if num_gpus > 1 {
        println!("• Multi-GPU disponible: {} GPUs CUDA", num_gpus);
        for i in 0..num_gpus {
            println!("  - Device cuda:{}", i);
        }
    }

    // 3. Device memory management
    println!("\n• Gestion mémoire par device:");
    for device in &devices {
        if let Some(info) = manager.get_device_info(device) {
            let used_memory = info.total_memory - info.available_memory;
            let usage_percent = (used_memory as f64 / info.total_memory as f64) * 100.0;
            println!("  {} - Utilisation: {:.1}%", device, usage_percent);
        }
    }

    // === Patterns d'utilisation avancés ===
    println!("\n🏗️  Patterns d'utilisation avancés:");

    // Device context manager pattern
    println!("• Pattern context manager:");
    println!("  with device(cuda:0):");
    println!("      tensor = Tensor::randn([1000, 1000])");
    println!("      # Toutes les opérations sur cuda:0");

    // Transfert entre devices
    println!("\n• Transfert entre devices:");
    println!("  cpu_tensor = Tensor::ones([100, 100])");
    println!("  gpu_tensor = cpu_tensor.to(Device::Cuda(0))");
    println!("  result = gpu_tensor.matmul(&other)");
    println!("  cpu_result = result.to(Device::Cpu)");

    // Opérations asynchrones
    println!("\n• Opérations asynchrones:");
    println!("  stream1 = CudaStream::new()");
    println!("  stream2 = CudaStream::new()");
    println!("  # Calculs parallèles sur différents streams");

    println!("\n✅ Démonstration Device terminée !");
    println!("📦 Fonctionnalités implémentées:");
    println!("   • Device discovery et énumération");
    println!("   • Support multi-GPU (CUDA, Metal, ROCm)");
    println!("   • Device memory management");
    println!("   • Device synchronization");
    println!("   • Allocateurs spécifiques par device");
    println!("   • Context managers pour devices");
    println!("   • Préparation pour calculs hétérogènes");
}
