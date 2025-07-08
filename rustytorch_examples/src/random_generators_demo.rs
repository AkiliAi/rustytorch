// rustytorch_examples/src/random_generators_demo.rs
// Démonstration des générateurs aléatoires avancés

use rustytorch_core::{DType, TensorOptions};
use rustytorch_tensor::Tensor;

pub fn run_random_generators_demo() {
    println!("🎲 Démonstration des générateurs aléatoires avancés RustyTorch\n");

    // Test randn - distribution normale standard N(0,1)
    println!("📊 Test randn (normal standard):");
    let randn_tensor = Tensor::randn(vec![5], None).unwrap();
    println!("Shape: {:?}", randn_tensor.shape());
    println!("Values: {:?}", randn_tensor.storage().to_vec_f64());
    println!("Type: {:?}", randn_tensor.dtype());

    // Générer un tenseur plus large pour vérifier la distribution
    let large_randn = Tensor::randn(vec![1000], None).unwrap();
    let data = large_randn.storage().to_vec_f64();
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    println!(
        "Large sample (n=1000) - Mean: {:.3}, Variance: {:.3}",
        mean, variance
    );
    // Attendu: Mean ≈ 0, Variance ≈ 1

    // Test normal - distribution normale N(mean, std²)
    println!("\n🎯 Test normal (mean=5.0, std=2.0):");
    let normal_tensor = Tensor::normal(5.0, 2.0, vec![5], None).unwrap();
    println!("Values: {:?}", normal_tensor.storage().to_vec_f64());

    // Test avec différents types
    let mut f64_options = TensorOptions::default();
    f64_options.dtype = DType::Float64;
    let normal_f64 = Tensor::normal(10.0, 1.5, vec![3], Some(f64_options)).unwrap();
    println!("F64 normal: {:?}", normal_f64.storage().to_vec_f64());

    // Test randint - entiers aléatoires
    println!("\n🎲 Test randint (0 à 10):");
    let mut int_options = TensorOptions::default();
    int_options.dtype = DType::Int32;
    let randint_tensor = Tensor::randint(0, 10, vec![8], Some(int_options)).unwrap();
    println!("Int32 values: {:?}", randint_tensor.storage().to_vec_f64());

    // Test avec différents types d'entiers
    let mut i64_options = TensorOptions::default();
    i64_options.dtype = DType::Int64;
    let randint_i64 = Tensor::randint(-5, 5, vec![5], Some(i64_options.clone())).unwrap();
    println!(
        "Int64 range [-5, 5): {:?}",
        randint_i64.storage().to_vec_f64()
    );

    // Test bernoulli - distribution de Bernoulli
    println!("\n🎯 Test bernoulli (p=0.3):");
    let mut bool_options = TensorOptions::default();
    bool_options.dtype = DType::Bool;
    let bernoulli_bool = Tensor::bernoulli(0.3, vec![10], Some(bool_options.clone())).unwrap();
    println!("Bool values: {:?}", bernoulli_bool.storage().to_vec_f64());

    // Bernoulli avec type float
    let bernoulli_float = Tensor::bernoulli(0.7, vec![8], None).unwrap();
    println!(
        "Float values (p=0.7): {:?}",
        bernoulli_float.storage().to_vec_f64()
    );

    // Test de la proportion
    let large_bernoulli = Tensor::bernoulli(0.4, vec![1000], Some(bool_options)).unwrap();
    let bern_data = large_bernoulli.storage().to_vec_f64();
    let true_count = bern_data.iter().filter(|&&x| x != 0.0).count();
    let proportion = true_count as f64 / bern_data.len() as f64;
    println!(
        "Large Bernoulli (n=1000, p=0.4) - Observed proportion: {:.3}",
        proportion
    );

    // Test uniform - distribution uniforme
    println!("\n📐 Test uniform [2.0, 8.0):");
    let uniform_tensor = Tensor::uniform(2.0, 8.0, vec![6], None).unwrap();
    println!("Values: {:?}", uniform_tensor.storage().to_vec_f64());

    // Test multinomial - échantillonnage multinomial
    println!("\n🎰 Test multinomial:");
    let weights = Tensor::from_data(&[1.0f64, 3.0, 2.0, 4.0], vec![4], None);
    println!("Weights: {:?}", weights.storage().to_vec_f64());

    // Avec remplacement
    let samples_with_replacement = weights.multinomial(10, true).unwrap();
    println!(
        "Samples with replacement (n=10): {:?}",
        samples_with_replacement.storage().to_vec_f64()
    );

    // Sans remplacement
    let samples_without_replacement = weights.multinomial(3, false).unwrap();
    println!(
        "Samples without replacement (n=3): {:?}",
        samples_without_replacement.storage().to_vec_f64()
    );

    // Applications pratiques
    println!("\n🧠 Applications pratiques:");

    // 1. Initialisation de poids de réseau neuronal
    println!("• Initialisation Xavier/Glorot:");
    let fan_in = 100;
    let fan_out = 50;
    let xavier_std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let xavier_weights = Tensor::normal(0.0, xavier_std, vec![fan_out, fan_in], None).unwrap();
    println!(
        "  Shape: {:?}, Std: {:.4}",
        xavier_weights.shape(),
        xavier_std
    );

    // 2. Dropout mask
    println!("• Masque de dropout (p=0.2):");
    let dropout_rate = 0.2;
    let mut mask_options = TensorOptions::default();
    mask_options.dtype = DType::Bool;
    let dropout_mask =
        Tensor::bernoulli(1.0 - dropout_rate, vec![4, 4], Some(mask_options)).unwrap();
    println!("  Dropout mask (4x4):");
    print_2d_bool_tensor(&dropout_mask);

    // 3. Échantillonnage de batch
    println!("• Échantillonnage de batch:");
    let dataset_size = 1000;
    let batch_size = 8;
    let batch_indices =
        Tensor::randint(0, dataset_size, vec![batch_size], Some(i64_options)).unwrap();
    println!(
        "  Batch indices: {:?}",
        batch_indices.storage().to_vec_f64()
    );

    // 4. Bruit gaussien pour augmentation de données
    println!("• Bruit gaussien (σ=0.1):");
    let noise = Tensor::normal(0.0, 0.1, vec![3, 3], None).unwrap();
    println!("  Noise matrix (3x3):");
    print_2d_tensor(&noise);

    // Statistiques sur les générateurs
    println!("\n📈 Validation statistique:");

    // Test de la loi des grands nombres
    let large_uniform = Tensor::uniform(0.0, 1.0, vec![10000], None).unwrap();
    let uniform_data = large_uniform.storage().to_vec_f64();
    let uniform_mean: f64 = uniform_data.iter().sum::<f64>() / uniform_data.len() as f64;
    println!(
        "Uniform [0,1) mean (n=10000): {:.4} (expected: 0.5)",
        uniform_mean
    );

    // Test multinomial sur grand échantillon
    let equal_weights = Tensor::from_data(&[1.0f64, 1.0, 1.0, 1.0], vec![4], None);
    let many_samples = equal_weights.multinomial(1000, true).unwrap();
    let sample_data = many_samples.storage().to_vec_f64();

    println!("Multinomial equal weights (n=1000):");
    for i in 0..4 {
        let count = sample_data.iter().filter(|&&x| x == i as f64).count();
        println!("  Category {}: {} (expected: ~250)", i, count);
    }

    println!("\n✅ Démonstration des générateurs aléatoires terminée !");
    println!("📦 Générateurs implémentés:");
    println!("   • randn() - Distribution normale standard N(0,1)");
    println!("   • normal() - Distribution normale N(μ,σ²)");
    println!("   • randint() - Entiers aléatoires dans un intervalle");
    println!("   • bernoulli() - Distribution de Bernoulli");
    println!("   • uniform() - Distribution uniforme continue");
    println!("   • multinomial() - Échantillonnage multinomial");
    println!("   • Support pour tous les types de données");
    println!("   • Validation statistique et cas d'usage ML");
}

/// Helper function to print 2D tensor in matrix format
fn print_2d_tensor(tensor: &Tensor) {
    let shape = tensor.shape();
    if shape.len() != 2 {
        println!("Cannot print non-2D tensor");
        return;
    }

    let data = tensor.storage().to_vec_f64();
    let rows = shape[0];
    let cols = shape[1];

    for r in 0..rows {
        print!("    [");
        for c in 0..cols {
            let val = data[r * cols + c];
            if c > 0 {
                print!(", ");
            }
            print!("{:6.3}", val);
        }
        println!("]");
    }
}

/// Helper function to print 2D boolean tensor
fn print_2d_bool_tensor(tensor: &Tensor) {
    let shape = tensor.shape();
    if shape.len() != 2 {
        println!("Cannot print non-2D tensor");
        return;
    }

    let data = tensor.storage().to_vec_f64();
    let rows = shape[0];
    let cols = shape[1];

    for r in 0..rows {
        print!("    [");
        for c in 0..cols {
            let val = data[r * cols + c];
            if c > 0 {
                print!(", ");
            }
            print!("{}", if val != 0.0 { "T" } else { "F" });
        }
        println!("]");
    }
}
