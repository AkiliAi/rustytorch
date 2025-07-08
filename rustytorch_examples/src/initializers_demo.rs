// rustytorch_examples/src/initializers_demo.rs
// Démonstration des fonctions d'initialisation de poids

use rustytorch_core::{DType, TensorOptions};
use rustytorch_tensor::{FanMode, Nonlinearity, Tensor};

pub fn run_initializers_demo() {
    println!("🎯 Démonstration des initialisations de poids RustyTorch\n");

    // === Test Xavier/Glorot Initialization ===
    println!("📊 Test Xavier/Glorot Initialization:");

    // Xavier uniform - pour tanh/sigmoid
    let xavier_uniform = Tensor::xavier_uniform(vec![64, 128], None, None).unwrap();
    let xu_data = xavier_uniform.storage().to_vec_f64();
    let xu_mean = xu_data.iter().sum::<f64>() / xu_data.len() as f64;
    let xu_variance =
        xu_data.iter().map(|&x| (x - xu_mean).powi(2)).sum::<f64>() / xu_data.len() as f64;

    println!("Xavier Uniform (64x128):");
    println!("  Shape: {:?}", xavier_uniform.shape());
    println!("  Mean: {:.6}, Variance: {:.6}", xu_mean, xu_variance);
    println!("  Expected variance: {:.6}", 2.0 / (64.0 + 128.0));
    println!(
        "  Range: [{:.4}, {:.4}]",
        xu_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        xu_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Xavier normal
    let xavier_normal = Tensor::xavier_normal(vec![32, 64], None, None).unwrap();
    let xn_data = xavier_normal.storage().to_vec_f64();
    let xn_mean = xn_data.iter().sum::<f64>() / xn_data.len() as f64;
    let xn_variance =
        xn_data.iter().map(|&x| (x - xn_mean).powi(2)).sum::<f64>() / xn_data.len() as f64;

    println!("\nXavier Normal (32x64):");
    println!("  Mean: {:.6}, Variance: {:.6}", xn_mean, xn_variance);
    println!("  Expected variance: {:.6}", 2.0 / (32.0 + 64.0));

    // === Test Kaiming/He Initialization ===
    println!("\n⚡ Test Kaiming/He Initialization (pour ReLU):");

    // Kaiming uniform - FanIn mode
    let kaiming_uniform_fanin = Tensor::kaiming_uniform(
        vec![256, 512],
        None,
        FanMode::FanIn,
        Nonlinearity::Relu,
        None,
    )
    .unwrap();

    let ku_data = kaiming_uniform_fanin.storage().to_vec_f64();
    let ku_mean = ku_data.iter().sum::<f64>() / ku_data.len() as f64;
    let ku_variance =
        ku_data.iter().map(|&x| (x - ku_mean).powi(2)).sum::<f64>() / ku_data.len() as f64;

    println!("Kaiming Uniform FanIn (256x512):");
    println!("  Mean: {:.6}, Variance: {:.6}", ku_mean, ku_variance);
    println!("  Expected variance: {:.6}", 2.0 / 512.0);
    println!(
        "  Range: [{:.4}, {:.4}]",
        ku_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        ku_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Kaiming normal - FanOut mode
    let kaiming_normal_fanout = Tensor::kaiming_normal(
        vec![128, 256],
        None,
        FanMode::FanOut,
        Nonlinearity::Relu,
        None,
    )
    .unwrap();

    let kn_data = kaiming_normal_fanout.storage().to_vec_f64();
    let kn_mean = kn_data.iter().sum::<f64>() / kn_data.len() as f64;
    let kn_variance =
        kn_data.iter().map(|&x| (x - kn_mean).powi(2)).sum::<f64>() / kn_data.len() as f64;

    println!("\nKaiming Normal FanOut (128x256):");
    println!("  Mean: {:.6}, Variance: {:.6}", kn_mean, kn_variance);
    println!("  Expected variance: {:.6}", 2.0 / 128.0);

    // Test avec LeakyReLU
    let kaiming_leaky = Tensor::kaiming_normal(
        vec![64, 128],
        Some(0.01), // negative_slope
        FanMode::FanIn,
        Nonlinearity::LeakyRelu,
        None,
    )
    .unwrap();

    let kl_data = kaiming_leaky.storage().to_vec_f64();
    let kl_variance = kl_data.iter().map(|&x| x.powi(2)).sum::<f64>() / kl_data.len() as f64;
    let expected_gain = ((2.0 / (1.0 + 0.01_f64.powi(2))) as f64).sqrt();

    println!("\nKaiming Normal LeakyReLU (slope=0.01, 64x128):");
    println!("  Variance: {:.6}", kl_variance);
    println!(
        "  Expected variance: {:.6}",
        (expected_gain.powi(2)) / 128.0
    );

    // === Test Orthogonal Initialization ===
    println!("\n🔄 Test Orthogonal Initialization:");

    // Matrice carrée orthogonale
    let ortho_square = Tensor::orthogonal(vec![4, 4], None, None).unwrap();
    println!("Orthogonal Square (4x4):");
    print_2d_tensor(&ortho_square, "Orthogonal Matrix");

    // Vérification de l'orthogonalité (colonnes normalisées)
    let ortho_data = ortho_square.storage().to_vec_f64();
    println!("Vérification orthogonalité:");
    for col in 0..4 {
        let column: Vec<f64> = (0..4).map(|row| ortho_data[row * 4 + col]).collect();
        let norm_squared: f64 = column.iter().map(|&x| x * x).sum();
        println!("  Colonne {}: norme² = {:.6}", col, norm_squared);
    }

    // Matrice rectangulaire
    let ortho_rect = Tensor::orthogonal(vec![3, 5], Some(2.0), None).unwrap();
    println!("\nOrthogonal Rectangular (3x5) avec gain=2.0:");
    print_2d_tensor(&ortho_rect, "Rectangular Orthogonal");

    // === Test avec différents types de données ===
    println!("\n🔢 Test avec différents types de données:");

    let mut f64_options = TensorOptions::default();
    f64_options.dtype = DType::Float64;

    let xavier_f64 = Tensor::xavier_normal(vec![8, 16], None, Some(f64_options)).unwrap();
    println!("Xavier Normal F64 (8x16): dtype = {:?}", xavier_f64.dtype());

    let kaiming_f32 =
        Tensor::kaiming_uniform(vec![16, 8], None, FanMode::FanIn, Nonlinearity::Relu, None)
            .unwrap();
    println!(
        "Kaiming Uniform F32 (16x8): dtype = {:?}",
        kaiming_f32.dtype()
    );

    // === Applications pratiques ===
    println!("\n🧠 Applications pratiques en Deep Learning:");

    // 1. Couche linéaire pour classification
    println!("• Couche de classification (1000 → 10):");
    let classifier_weights = Tensor::xavier_normal(vec![10, 1000], None, None).unwrap();
    let cw_data = classifier_weights.storage().to_vec_f64();
    let cw_std = (cw_data.iter().map(|&x| x.powi(2)).sum::<f64>() / cw_data.len() as f64).sqrt();
    println!(
        "  Shape: {:?}, Std: {:.6}",
        classifier_weights.shape(),
        cw_std
    );

    // 2. Première couche CNN
    println!("• Première couche CNN (32 filtres 3x3, 3 canaux):");
    let conv_weights = Tensor::kaiming_normal(
        vec![32, 3, 3, 3],
        None,
        FanMode::FanIn,
        Nonlinearity::Relu,
        None,
    )
    .unwrap();
    let conv_data = conv_weights.storage().to_vec_f64();
    let conv_std =
        (conv_data.iter().map(|&x| x.powi(2)).sum::<f64>() / conv_data.len() as f64).sqrt();
    println!("  Shape: {:?}, Std: {:.6}", conv_weights.shape(), conv_std);
    println!(
        "  Fan_in: 3*3*3 = 27, Expected std: {:.6}",
        (2.0_f64 / 27.0).sqrt()
    );

    // 3. LSTM/RNN avec initialisation orthogonale
    println!("• Poids récurrents LSTM (hidden_size=256):");
    let lstm_recurrent = Tensor::orthogonal(vec![256, 256], Some(1.0), None).unwrap();
    println!("  Shape: {:?}", lstm_recurrent.shape());
    println!("  Initialisation orthogonale pour éviter gradient vanishing");

    // 4. Réseau résiduel avec gain personnalisé
    println!("• Dernière couche ResNet (gain=0.1 pour stabilité):");
    let resnet_final = Tensor::kaiming_normal(
        vec![512, 512],
        None,
        FanMode::FanOut,
        Nonlinearity::Relu,
        None,
    )
    .unwrap();
    // Appliquer gain=0.1 manuellement (multiplication par scalaire)
    let resnet_data = resnet_final.storage().to_vec_f64();
    let scaled_data: Vec<f64> = resnet_data.iter().map(|&x| x * 0.1).collect();
    let resnet_scaled = Tensor::from_data(&scaled_data, resnet_final.shape().to_vec(), None);
    let rs_data = resnet_scaled.storage().to_vec_f64();
    let rs_std = (rs_data.iter().map(|&x| x.powi(2)).sum::<f64>() / rs_data.len() as f64).sqrt();
    println!(
        "  Shape: {:?}, Std après scaling: {:.6}",
        resnet_scaled.shape(),
        rs_std
    );

    // === Comparaison des méthodes ===
    println!("\n📈 Comparaison des méthodes d'initialisation:");

    let shape = vec![100, 100];

    // Standard normal
    let std_normal = Tensor::randn(shape.clone(), None).unwrap();
    let sn_data = std_normal.storage().to_vec_f64();
    let sn_std = (sn_data.iter().map(|&x| x.powi(2)).sum::<f64>() / sn_data.len() as f64).sqrt();

    // Xavier
    let xavier_comp = Tensor::xavier_normal(shape.clone(), None, None).unwrap();
    let xc_data = xavier_comp.storage().to_vec_f64();
    let xc_std = (xc_data.iter().map(|&x| x.powi(2)).sum::<f64>() / xc_data.len() as f64).sqrt();

    // Kaiming
    let kaiming_comp = Tensor::kaiming_normal(
        shape.clone(),
        None,
        FanMode::FanIn,
        Nonlinearity::Relu,
        None,
    )
    .unwrap();
    let kc_data = kaiming_comp.storage().to_vec_f64();
    let kc_std = (kc_data.iter().map(|&x| x.powi(2)).sum::<f64>() / kc_data.len() as f64).sqrt();

    println!("Pour shape [100, 100]:");
    println!("  Standard Normal:  std = {:.6}", sn_std);
    println!(
        "  Xavier Normal:    std = {:.6} (expected: {:.6})",
        xc_std,
        (2.0_f64 / 200.0).sqrt()
    );
    println!(
        "  Kaiming Normal:   std = {:.6} (expected: {:.6})",
        kc_std,
        (2.0_f64 / 100.0).sqrt()
    );

    println!("\n✅ Démonstration des initialisations terminée !");
    println!("📦 Méthodes d'initialisation implémentées:");
    println!("   • xavier_uniform() - Distribution uniforme Xavier/Glorot");
    println!("   • xavier_normal() - Distribution normale Xavier/Glorot");
    println!("   • kaiming_uniform() - Distribution uniforme Kaiming/He");
    println!("   • kaiming_normal() - Distribution normale Kaiming/He");
    println!("   • orthogonal() - Matrices orthogonales");
    println!("   • Support FanIn/FanOut et différentes non-linéarités");
    println!("   • Calculs automatiques de variance optimale");
    println!("   • Applications CNN, RNN, ResNet, Classification");
}

/// Helper function to print 2D tensor in matrix format
fn print_2d_tensor(tensor: &Tensor, name: &str) {
    let shape = tensor.shape();
    if shape.len() != 2 {
        println!("Cannot print non-2D tensor: {}", name);
        return;
    }

    let data = tensor.storage().to_vec_f64();
    let rows = shape[0];
    let cols = shape[1];

    println!("{}:", name);
    for r in 0..rows.min(4) {
        // Limiter l'affichage à 4 lignes
        print!("  [");
        for c in 0..cols.min(6) {
            // Limiter l'affichage à 6 colonnes
            let val = data[r * cols + c];
            if c > 0 {
                print!(", ");
            }
            print!("{:7.4}", val);
        }
        if cols > 6 {
            print!(", ...");
        }
        println!("]");
    }
    if rows > 4 {
        println!("  ...");
    }
}
