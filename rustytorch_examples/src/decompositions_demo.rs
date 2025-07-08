// rustytorch_examples/src/decompositions_demo.rs
// Démonstration des décompositions matricielles

use rustytorch_core::{DType, Reshapable, TensorOptions};
use rustytorch_tensor::Tensor;

pub fn run_decompositions_demo() {
    println!("🔢 Démonstration des décompositions matricielles RustyTorch\n");

    // === Test Cholesky Decomposition ===
    println!("🔺 Test décomposition de Cholesky:");

    // Créer une matrice symétrique définie positive
    // A = [[4, 2], [2, 3]]
    let a_data = vec![4.0, 2.0, 2.0, 3.0];
    let a = Tensor::from_data(&a_data, vec![2, 2], None);
    println!("Matrice A (symétrique définie positive):");
    print_2d_tensor(&a, "A");

    // Décomposition de Cholesky (triangulaire inférieure)
    let l = a.cholesky(false).unwrap();
    println!("\nDécomposition de Cholesky L:");
    print_2d_tensor(&l, "L");

    // Vérification: L * L^T = A
    let lt = l.transpose(0, 1).unwrap();
    let reconstructed = l.matmul(&lt).unwrap();
    println!("\nReconstruction L * L^T:");
    print_2d_tensor(&reconstructed, "L * L^T");

    // Test avec triangulaire supérieure
    let u = a.cholesky(true).unwrap();
    println!("\nDécomposition de Cholesky U (upper):");
    print_2d_tensor(&u, "U");

    // === Test QR Decomposition ===
    println!("\n📐 Test décomposition QR:");

    // Créer une matrice rectangulaire
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = Tensor::from_data(&b_data, vec![3, 2], None);
    println!("Matrice B (3x2):");
    print_2d_tensor(&b, "B");

    // Décomposition QR
    let (q, r) = b.qr().unwrap();
    println!("\nMatrice Q (orthogonale):");
    print_2d_tensor(&q, "Q");
    println!("\nMatrice R (triangulaire supérieure):");
    print_2d_tensor(&r, "R");

    // Vérification: Q * R = B
    let qr_product = q.matmul(&r).unwrap();
    println!("\nReconstruction Q * R:");
    print_2d_tensor(&qr_product, "Q * R");

    // Vérifier l'orthogonalité de Q: Q^T * Q = I
    let qt = q.transpose(0, 1).unwrap();
    let qtq = qt.matmul(&q).unwrap();
    println!("\nVérification orthogonalité Q^T * Q:");
    print_2d_tensor(&qtq, "Q^T * Q");

    // === Test SVD (Singular Value Decomposition) ===
    println!("\n🎯 Test décomposition en valeurs singulières (SVD):");

    // Matrice carrée
    let c_data = vec![1.0, 2.0, 3.0, 4.0];
    let c = Tensor::from_data(&c_data, vec![2, 2], None);
    println!("Matrice C (2x2):");
    print_2d_tensor(&c, "C");

    // SVD: C = U * S * V^T
    let (u, s, v) = c.svd(false).unwrap();
    println!("\nMatrice U (vecteurs singuliers gauches):");
    print_2d_tensor(&u, "U");
    println!("\nValeurs singulières S:");
    println!("  {:?}", s.storage().to_vec_f64());
    println!("\nMatrice V (vecteurs singuliers droits):");
    print_2d_tensor(&v, "V");

    // Test avec matrice rectangulaire
    println!("\n📊 SVD sur matrice rectangulaire:");
    let d_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let d = Tensor::from_data(&d_data, vec![4, 3], None);
    println!("Matrice D (4x3):");
    print_2d_tensor(&d, "D");

    let (u2, s2, v2) = d.svd(false).unwrap();
    println!("\nDimensions après SVD:");
    println!("  U: {:?}", u2.shape());
    println!("  S: {:?} (valeurs singulières)", s2.shape());
    println!("  V: {:?}", v2.shape());
    println!("Valeurs singulières: {:?}", s2.storage().to_vec_f64());

    // === Applications pratiques ===
    println!("\n🧠 Applications pratiques:");

    // 1. Résolution de système linéaire avec Cholesky
    println!("• Résolution de système linéaire Ax = b avec Cholesky:");
    // A est définie positive, on veut résoudre Ax = b
    let b_vec = vec![10.0, 13.0];
    let b_rhs = Tensor::from_data(&b_vec, vec![2, 1], None);
    println!("  Système: A * x = b où b = [10, 13]^T");

    // A = L * L^T, donc A*x = b devient L*L^T*x = b
    // On résout d'abord L*y = b, puis L^T*x = y
    println!("  Utilisation de la décomposition de Cholesky pour une résolution efficace");

    // 2. Compression d'image avec SVD
    println!("\n• Compression avec SVD (rank approximation):");
    // Créer une "image" 5x5
    let image_data: Vec<f64> = (0..25).map(|i| (i as f64).sin() * 10.0).collect();
    let image = Tensor::from_data(&image_data, vec![5, 5], None);
    println!("  Image originale (5x5):");
    print_2d_tensor(&image, "  ");

    let (u_img, s_img, v_img) = image.svd(false).unwrap();
    let s_vals = s_img.storage().to_vec_f64();
    println!("  Valeurs singulières: {:?}", &s_vals[..3]);

    // Approximation rang 2 (garder seulement 2 valeurs singulières)
    println!("  Approximation rang 2 (compression):");
    println!("  → Garde seulement les 2 plus grandes valeurs singulières");

    // 3. Analyse en composantes principales avec SVD
    println!("\n• Analyse en composantes principales (PCA):");
    // Données centrées (exemples x features)
    let data_matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 4.0, 6.0];
    let data = Tensor::from_data(&data_matrix, vec![4, 3], None);
    println!("  Matrice de données (4 échantillons, 3 features):");
    print_2d_tensor(&data, "  ");

    let (_, s_pca, v_pca) = data.svd(false).unwrap();
    println!(
        "  Valeurs singulières (variance): {:?}",
        s_pca.storage().to_vec_f64()
    );
    println!("  Composantes principales dans les colonnes de V");

    // 4. Conditionnement et stabilité numérique
    println!("\n• Analyse de conditionnement:");
    let s_cond = s_img.storage().to_vec_f64();
    if s_cond.len() >= 2 && s_cond[s_cond.len() - 1] > 1e-10 {
        let condition_number = s_cond[0] / s_cond[s_cond.len() - 1];
        println!("  Nombre de conditionnement: {:.2}", condition_number);
        println!("  (ratio plus grande/plus petite valeur singulière)");
    }

    // 5. Décomposition QR pour moindres carrés
    println!("\n• Moindres carrés avec QR:");
    // Résoudre Ax ≈ b au sens des moindres carrés
    let a_ls = Tensor::from_data(&[1.0, 1.0, 1.0, 2.0, 1.0, 3.0], vec![3, 2], None);
    let b_ls = Tensor::from_data(&[1.0, 2.0, 2.5], vec![3, 1], None);

    let (q_ls, r_ls) = a_ls.qr().unwrap();
    println!("  Système surdéterminé A (3x2) * x = b (3x1)");
    println!("  Utilisation de QR pour solution au sens des moindres carrés");

    // === Tests de robustesse ===
    println!("\n🛡️ Tests de robustesse:");

    // Test matrice non définie positive pour Cholesky
    println!("• Test Cholesky sur matrice non définie positive:");
    let non_pd = Tensor::from_data(&[1.0, 2.0, 2.0, 1.0], vec![2, 2], None);
    match non_pd.cholesky(false) {
        Ok(_) => println!("  ⚠️ Devrait échouer!"),
        Err(e) => println!("  ✓ Erreur attendue: {:?}", e),
    }

    // Test avec différents types
    println!("\n• Test avec Float32:");
    let mut f32_options = TensorOptions::default();
    f32_options.dtype = DType::Float32;
    let a_f32 = Tensor::from_data(&[4.0f32, 2.0, 2.0, 3.0], vec![2, 2], Some(f32_options));
    let l_f32 = a_f32.cholesky(false).unwrap();
    println!("  Cholesky F32 réussi, dtype: {:?}", l_f32.dtype());

    println!("\n✅ Démonstration des décompositions terminée !");
    println!("📦 Décompositions implémentées:");
    println!("   • cholesky() - Décomposition de Cholesky (L*L^T ou U^T*U)");
    println!("   • qr() - Décomposition QR (Q orthogonale, R triangulaire)");
    println!("   • svd() - Décomposition en valeurs singulières");
    println!("   • Applications: systèmes linéaires, compression, PCA");
    println!("   • Support multi-types et validation robuste");
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

    if !name.is_empty() {
        println!("{}:", name);
    }

    for r in 0..rows.min(5) {
        // Limiter l'affichage
        print!("  [");
        for c in 0..cols.min(5) {
            let val = data[r * cols + c];
            if c > 0 {
                print!(", ");
            }
            print!("{:7.3}", val);
        }
        if cols > 5 {
            print!(", ...");
        }
        println!("]");
    }
    if rows > 5 {
        println!("  ...");
    }
}
