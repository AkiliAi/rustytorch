// rustytorch_examples/src/advanced_linalg.rs
// Démonstration des opérations d'algèbre linéaire avancées

use rustytorch_core::Reshapable;
use rustytorch_tensor::Tensor;

pub fn run_advanced_linalg_demo() {
    println!("🧮 Démonstration d'algèbre linéaire avancée RustyTorch\n");

    // Test tensordot - produit tensoriel généralisé
    println!("🔀 Test tensordot:");
    let matrix_a = Tensor::from_data(&[1.0f64, 2.0, 3.0, 4.0], vec![2, 2], None);
    let matrix_b = Tensor::from_data(&[5.0f64, 6.0, 7.0, 8.0], vec![2, 2], None);

    println!("Matrice A (2x2): {:?}", matrix_a.storage().to_vec_f64());
    println!("Matrice B (2x2): {:?}", matrix_b.storage().to_vec_f64());

    // tensordot avec axes (1,0) - équivalent à la multiplication matricielle
    let tensordot_result = matrix_a.tensordot(&matrix_b, (vec![1], vec![0])).unwrap();
    println!(
        "Tensordot A⊗B axes([1],[0]): {:?}",
        tensordot_result.storage().to_vec_f64()
    );
    println!("Shape: {:?}", tensordot_result.shape());
    // Attendu: matmul(A, B) = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

    // Test outer product - produit extérieur
    println!("\n⊗ Test outer product:");
    let vec_u = Tensor::from_data(&[1.0f64, 2.0, 3.0], vec![3], None);
    let vec_v = Tensor::from_data(&[4.0f64, 5.0], vec![2], None);

    println!("Vecteur u: {:?}", vec_u.storage().to_vec_f64());
    println!("Vecteur v: {:?}", vec_v.storage().to_vec_f64());

    let outer_result = vec_u.outer(&vec_v).unwrap();
    println!("Outer product u⊗v:");
    print_2d_tensor(&outer_result);
    // Attendu: [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]] = [[4, 5], [8, 10], [12, 15]]

    // Test diagonal extraction
    println!("\n📐 Test diagonal extraction:");
    let matrix_c = Tensor::from_data(
        &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
        None,
    );

    println!("Matrice C (3x3):");
    print_2d_tensor(&matrix_c);

    // Diagonale principale
    let main_diag = matrix_c.diagonal(0, None, None).unwrap();
    println!(
        "Diagonale principale: {:?}",
        main_diag.storage().to_vec_f64()
    );
    // Attendu: [1, 5, 9]

    // Diagonale supérieure (offset +1)
    let upper_diag = matrix_c.diagonal(1, None, None).unwrap();
    println!(
        "Diagonale supérieure (+1): {:?}",
        upper_diag.storage().to_vec_f64()
    );
    // Attendu: [2, 6]

    // Diagonale inférieure (offset -1)
    let lower_diag = matrix_c.diagonal(-1, None, None).unwrap();
    println!(
        "Diagonale inférieure (-1): {:?}",
        lower_diag.storage().to_vec_f64()
    );
    // Attendu: [4, 8]

    // Test trace
    println!("\n🎯 Test trace (somme de la diagonale):");
    let trace_c = matrix_c.trace().unwrap();
    println!("Trace de C: {}", trace_c);
    // Attendu: 1 + 5 + 9 = 15

    // Test avec matrice rectangulaire
    let rect_matrix = Tensor::from_data(
        &[
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![3, 4],
        None,
    );

    println!("\nMatrice rectangulaire (3x4):");
    print_2d_tensor(&rect_matrix);

    let rect_diag = rect_matrix.diagonal(0, None, None).unwrap();
    println!("Diagonale: {:?}", rect_diag.storage().to_vec_f64());
    // Attendu: [1, 6, 11] (min(3,4) = 3 éléments)

    // Applications pratiques
    println!("\n🧠 Applications pratiques:");

    // 1. Calcul de la norme de Frobenius avec trace
    println!("• Calcul de norme:");
    let small_matrix = Tensor::from_data(&[3.0f64, 4.0, 0.0, 0.0], vec![2, 2], None);

    // A^T @ A pour obtenir la matrice de Gram
    let at = small_matrix.transpose(0, 1).unwrap();
    let gram = at.matmul(&small_matrix).unwrap();
    let trace_gram = gram.trace().unwrap();
    let frobenius_norm = trace_gram.sqrt();

    println!("  Matrice: {:?}", small_matrix.storage().to_vec_f64());
    println!(
        "  Norme de Frobenius via trace(A^T@A): {:.3}",
        frobenius_norm
    );

    // 2. Création de matrice diagonale à partir d'un vecteur
    println!("\n• Construction de matrice diagonale:");
    let diag_values = vec![2.0f64, 3.0, 5.0];
    let identity = Tensor::from_data(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![3, 3],
        None,
    );

    // Simuler la création d'une matrice diagonale (en réalité on multiplierait chaque colonne)
    println!("  Valeurs diagonales: {:?}", diag_values);
    println!("  (Construction de matrice diagonale - à implémenter)");
    // 3. Produit vectoriel via outer product
    println!("• Produit de rang 1 via outer product:");
    let u = Tensor::from_data(&[1.0f64, 0.0], vec![2], None);
    let v = Tensor::from_data(&[0.0f64, 1.0], vec![2], None);
    let rank1 = u.outer(&v).unwrap();

    println!("  u = {:?}", u.storage().to_vec_f64());
    println!("  v = {:?}", v.storage().to_vec_f64());
    println!("  u⊗v (matrice de rang 1):");
    print_2d_tensor(&rank1);

    println!("\n✅ Démonstration d'algèbre linéaire avancée terminée !");
    println!("📦 Nouvelles fonctionnalités implémentées:");
    println!("   • tensordot() - Produit tensoriel généralisé");
    println!("   • outer() - Produit extérieur de tenseurs");
    println!("   • diagonal() - Extraction de diagonales avec offset");
    println!("   • trace() - Somme des éléments diagonaux");
    println!("   • Support pour matrices rectangulaires et décalages");
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
        print!("  [");
        for c in 0..cols {
            let val = data[r * cols + c];
            if c > 0 {
                print!(", ");
            }
            print!("{:5.1}", val);
        }
        println!("]");
    }
}
