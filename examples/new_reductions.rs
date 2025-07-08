// examples/new_reductions.rs
// Test rapide des nouvelles fonctionnalités de réduction

use rustytorch_tensor::Tensor;

fn main() {
    println!("🧪 Test des nouvelles réductions dans RustyTorch\n");

    // Test cumsum
    println!("📊 Test cumsum:");
    let tensor = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], None);
    println!("Tensor original: {:?}", tensor.storage().to_vec_f64());
    
    let cumsum_result = tensor.cumsum(0).unwrap();
    println!("Cumsum result: {:?}", cumsum_result.storage().to_vec_f64());
    // Attendu: [1.0, 3.0, 6.0, 10.0]
    
    // Test cumprod
    println!("\n📈 Test cumprod:");
    let cumprod_result = tensor.cumprod(0).unwrap();
    println!("Cumprod result: {:?}", cumprod_result.storage().to_vec_f64());
    // Attendu: [1.0, 2.0, 6.0, 24.0]
    
    // Test norm L2 (par défaut)
    println!("\n📏 Test norm L2:");
    let norm_result = tensor.norm(None, None, false).unwrap();
    println!("L2 norm: {:?}", norm_result.storage().get_f64(0).unwrap());
    // Attendu: sqrt(1²+2²+3²+4²) = sqrt(30) ≈ 5.48
    
    // Test norm L1
    println!("\n📐 Test norm L1:");
    let norm_l1 = tensor.norm(Some(1.0), None, false).unwrap();
    println!("L1 norm: {:?}", norm_l1.storage().get_f64(0).unwrap());
    // Attendu: |1|+|2|+|3|+|4| = 10.0
    
    // Test norm Linf (max)
    println!("\n🎯 Test norm L-infinity:");
    let norm_inf = tensor.norm(Some(f64::INFINITY), None, false).unwrap();
    println!("L∞ norm: {:?}", norm_inf.storage().get_f64(0).unwrap());
    // Attendu: max(|1|,|2|,|3|,|4|) = 4.0
    
    // Test avec tenseur 2D
    println!("\n🔲 Test sur tenseur 2D:");
    let tensor_2d = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None);
    println!("Tensor 2D: {:?} shape: {:?}", tensor_2d.storage().to_vec_f64(), tensor_2d.shape());
    
    // Cumsum le long de l'axe 0
    let cumsum_2d = tensor_2d.cumsum(0).unwrap();
    println!("Cumsum axis 0: {:?}", cumsum_2d.storage().to_vec_f64());
    // Attendu: [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
    
    // Norm Frobenius
    let frob_norm = tensor_2d.frobenius_norm().unwrap();
    println!("Frobenius norm: {:?}", frob_norm.storage().get_f64(0).unwrap());
    // Attendu: sqrt(1²+2²+3²+4²+5²+6²) = sqrt(91) ≈ 9.54
    
    println!("\n✅ Tous les tests des nouvelles réductions sont complétés !");
    println!("📦 Les nouvelles fonctionnalités ajoutées:");
    println!("   • cumsum() - Somme cumulative le long d'un axe");
    println!("   • cumprod() - Produit cumulatif le long d'un axe");
    println!("   • norm() - Calcul de normes (L1, L2, Lp, L∞)");
    println!("   • frobenius_norm() - Norme de Frobenius");
}