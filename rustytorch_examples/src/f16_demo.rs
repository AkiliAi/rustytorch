// rustytorch_examples/src/f16_demo.rs
// Démonstration du support F16 (half precision)

use half::f16;
use rustytorch_core::{DType, TensorOptions};
use rustytorch_tensor::{
    f16_support::{F16Arithmetic, F16Conversions, F16Ops, F16Utils, MixedPrecisionOps},
    Tensor,
};

pub fn run_f16_demo() {
    println!("🔢 Démonstration du support F16 (Half Precision) RustyTorch\n");

    // === Création de tenseurs F16 ===
    println!("📊 Création de tenseurs F16:");

    // Tenseur F16 zeros
    let zeros_f16 = Tensor::zeros_f16(vec![2, 3]);
    println!(
        "Zeros F16 - shape: {:?}, dtype: {:?}",
        zeros_f16.shape(),
        zeros_f16.dtype()
    );

    // Tenseur F16 ones
    let ones_f16 = Tensor::ones_f16(vec![3, 2]);
    println!(
        "Ones F16 - shape: {:?}, dtype: {:?}",
        ones_f16.shape(),
        ones_f16.dtype()
    );

    // Tenseur F16 avec valeur personnalisée
    let custom_f16 = Tensor::full_f16(vec![2, 2], f16::from_f32(3.14));
    println!("Custom F16 (π) - shape: {:?}", custom_f16.shape());

    // === Conversions de types ===
    println!("\n🔄 Conversions de types:");

    // F32 vers F16
    let f32_tensor = Tensor::from_data(&[1.0f32, 2.5, 3.7, -4.2], vec![4], None);
    let f16_converted = f32_tensor.to_f16().unwrap();
    println!(
        "F32 → F16 conversion: dtype avant={:?}, après={:?}",
        f32_tensor.dtype(),
        f16_converted.dtype()
    );

    // Vérifier la précision
    let f32_data = vec![1.23456789f32, 9.87654321f32];
    let f16_data = F16Conversions::f32_to_f16(&f32_data);
    let f32_back = F16Conversions::f16_to_f32(&f16_data);
    println!("\nTest précision F32→F16→F32:");
    for i in 0..2 {
        println!(
            "  Original: {:.8}, Après conversion: {:.8}, Erreur: {:.8}",
            f32_data[i],
            f32_back[i],
            (f32_data[i] - f32_back[i]).abs()
        );
    }

    // === Arithmétique F16 ===
    println!("\n➕ Arithmétique F16:");

    let a = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
    let b = vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)];

    // Addition
    let sum = F16Arithmetic::add_f16(&a, &b).unwrap();
    println!(
        "Addition F16: [1,2,3] + [4,5,6] = [{},{},{}]",
        sum[0].to_f32(),
        sum[1].to_f32(),
        sum[2].to_f32()
    );

    // Multiplication
    let prod = F16Arithmetic::mul_f16(&a, &b).unwrap();
    println!(
        "Multiplication F16: [1,2,3] * [4,5,6] = [{},{},{}]",
        prod[0].to_f32(),
        prod[1].to_f32(),
        prod[2].to_f32()
    );

    // Réductions
    let sum_val = F16Arithmetic::sum_f16(&a);
    let mean_val = F16Arithmetic::mean_f16(&a);
    println!(
        "Sum F16: {}, Mean F16: {}",
        sum_val.to_f32(),
        mean_val.to_f32()
    );

    // === Matrix multiplication F16 ===
    println!("\n🔢 Multiplication matricielle F16:");

    let mat_a = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    let mat_b = vec![
        f16::from_f32(5.0),
        f16::from_f32(6.0),
        f16::from_f32(7.0),
        f16::from_f32(8.0),
    ];

    let result = F16Arithmetic::matmul_f16(&mat_a, &mat_b, 2, 2, 2).unwrap();
    println!("Matmul F16 (2x2):");
    println!(
        "  [{:.1} {:.1}]   [{:.1} {:.1}]   [{:.1} {:.1}]",
        mat_a[0].to_f32(),
        mat_a[1].to_f32(),
        mat_b[0].to_f32(),
        mat_b[1].to_f32(),
        result[0].to_f32(),
        result[1].to_f32()
    );
    println!(
        "  [{:.1} {:.1}] × [{:.1} {:.1}] = [{:.1} {:.1}]",
        mat_a[2].to_f32(),
        mat_a[3].to_f32(),
        mat_b[2].to_f32(),
        mat_b[3].to_f32(),
        result[2].to_f32(),
        result[3].to_f32()
    );

    // === Mixed Precision ===
    println!("\n🎯 Mixed Precision (F16/F32):");

    // Comparaison précision pure F16 vs mixed
    let large_a = vec![f16::from_f32(0.001); 100];
    let large_b = vec![f16::from_f32(0.001); 100];

    // Pure F16
    let pure_f16 = F16Arithmetic::matmul_f16(&large_a, &large_b, 10, 10, 10).unwrap();

    // Mixed precision (calcul en F32, stockage en F16)
    let mixed = MixedPrecisionOps::mixed_matmul(&large_a, &large_b, 10, 10, 10).unwrap();

    println!("Matmul 10x10 (valeurs 0.001):");
    println!("  Pure F16 result[0,0]: {}", pure_f16[0].to_f32());
    println!("  Mixed precision[0,0]: {}", mixed[0].to_f32());
    println!("  Différence: {}", (pure_f16[0] - mixed[0]).to_f32().abs());

    // === Valeurs spéciales F16 ===
    println!("\n🌟 Valeurs spéciales F16:");

    println!("  Epsilon: {}", F16Utils::epsilon().to_f32());
    println!("  Infinity: {}", F16Utils::infinity().to_f32());
    println!("  -Infinity: {}", F16Utils::neg_infinity().to_f32());
    println!("  Min positive: {}", f16::MIN_POSITIVE.to_f32());
    println!("  Max: {}", f16::MAX.to_f32());

    // Test overflow/underflow
    let big = f16::from_f32(60000.0);
    let tiny = f16::from_f32(0.00001);
    println!("\nGestion overflow/underflow:");
    println!(
        "  60000.0 → F16: {} ({})",
        big.to_f32(),
        if big.is_infinite() {
            "overflow to inf"
        } else {
            "ok"
        }
    );
    println!(
        "  0.00001 → F16: {} ({})",
        tiny.to_f32(),
        if tiny == f16::from_f32(0.0) {
            "underflow to 0"
        } else {
            "ok"
        }
    );

    // === Cas d'usage pratiques ===
    println!("\n💡 Cas d'usage pratiques:");

    // 1. Économie mémoire
    let f32_size = 1000 * 1000 * 4; // 1M éléments * 4 bytes
    let f16_size = 1000 * 1000 * 2; // 1M éléments * 2 bytes
    println!("• Économie mémoire pour 1M éléments:");
    println!("  F32: {} MB", f32_size as f64 / (1024.0 * 1024.0));
    println!("  F16: {} MB", f16_size as f64 / (1024.0 * 1024.0));
    println!("  Économie: 50%");

    // 2. Gradient accumulation
    println!("\n• Gradient accumulation en mixed precision:");
    println!("  1. Forward pass en F16 (rapide)");
    println!("  2. Gradients calculés en F32 (précis)");
    println!("  3. Poids mis à jour en F32");
    println!("  4. Poids stockés en F16");

    // 3. Dynamic loss scaling
    println!("\n• Dynamic loss scaling pour éviter underflow:");
    let loss = f16::from_f32(0.0001);
    let scale = 1024.0;
    let scaled_loss = f16::from_f32(loss.to_f32() * scale);
    println!("  Loss original: {}", loss.to_f32());
    println!("  Scale factor: {}", scale);
    println!("  Scaled loss: {}", scaled_loss.to_f32());

    // === Patterns d'utilisation avancés ===
    println!("\n🏗️ Patterns avancés:");

    // AMP (Automatic Mixed Precision) helper
    println!("• AMP helper pour opérations complexes:");
    let input = vec![f16::from_f32(1.0), f16::from_f32(4.0), f16::from_f32(9.0)];
    let sqrt_result =
        MixedPrecisionOps::amp_operation(&input, |data| data.iter().map(|&x| x.sqrt()).collect());
    println!(
        "  sqrt([1,4,9]) en mixed precision: [{},{},{}]",
        sqrt_result[0].to_f32(),
        sqrt_result[1].to_f32(),
        sqrt_result[2].to_f32()
    );

    // === Benchmarks théoriques ===
    println!("\n📈 Performance théorique:");
    println!("• Avantages F16:");
    println!("  - 2x moins de mémoire");
    println!("  - 2x plus de bandwidth");
    println!("  - 2-8x plus rapide sur GPU avec Tensor Cores");
    println!("• Limitations:");
    println!("  - Range limité: ±65504");
    println!("  - Précision: ~3-4 décimales");
    println!("  - Risque underflow/overflow");

    // === Recommandations ===
    println!("\n📌 Recommandations d'utilisation:");
    println!("• Utiliser F16 pour:");
    println!("  - Inférence de modèles entraînés");
    println!("  - Forward pass pendant l'entraînement");
    println!("  - Stockage de grandes matrices d'embeddings");
    println!("• Utiliser F32/Mixed pour:");
    println!("  - Accumulation de gradients");
    println!("  - Optimiseurs (Adam, SGD)");
    println!("  - Batch normalization stats");

    println!("\n✅ Démonstration F16 terminée !");
    println!("📦 Support F16 implémenté:");
    println!("   • Conversions F16↔F32↔F64");
    println!("   • Arithmétique native F16");
    println!("   • Mixed precision operations");
    println!("   • Création de tenseurs F16");
    println!("   • Gestion valeurs spéciales");
    println!("   • Helpers pour AMP");
    println!("   • Préparation pour GPU Tensor Cores");
}
