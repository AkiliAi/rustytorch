// rustytorch_examples/src/padding_demo.rs
// Démonstration des opérations de padding et cropping

use rustytorch_tensor::{
    padding::{PaddingMode, PaddingSpec},
    Tensor,
};

pub fn run_padding_demo() {
    println!("🖼️  Démonstration des opérations de padding et cropping\n");

    // Test avec un tenseur 1D
    println!("📏 Test padding 1D:");
    let tensor_1d = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], None);
    println!("Tensor original: {:?}", tensor_1d.storage().to_vec_f64());

    // Zero padding
    let padded_1d = tensor_1d.zero_pad(vec![(2, 1)]).unwrap(); // 2 avant, 1 après
    println!("Zero padded [2,1]: {:?}", padded_1d.storage().to_vec_f64());
    // Attendu: [0, 0, 1, 2, 3, 0]

    // Constant padding avec valeur
    let const_padded = tensor_1d.constant_pad(vec![(1, 2)], -1.0).unwrap();
    println!(
        "Constant padded [-1]: {:?}",
        const_padded.storage().to_vec_f64()
    );
    // Attendu: [-1, 1, 2, 3, -1, -1]

    // Test avec tenseur 2D (image 3x3)
    println!("\n🖼️  Test padding 2D (image 3x3):");
    let image = Tensor::from_data(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
        None,
    );

    println!("Image originale 3x3:");
    print_2d_tensor(&image);

    // Padding uniforme
    let padded_image = image.zero_pad(vec![(1, 1), (1, 1)]).unwrap(); // 1 pixel sur tous les côtés
    println!("\nAprès zero padding (1 pixel partout) -> 5x5:");
    print_2d_tensor(&padded_image);

    // Padding asymétrique
    let asym_padded = image.zero_pad(vec![(2, 0), (0, 3)]).unwrap(); // 2 en haut, 3 à droite
    println!("\nPadding asymétrique (2 en haut, 3 à droite) -> 5x6:");
    print_2d_tensor(&asym_padded);

    // Test cropping
    println!("\n✂️  Test cropping:");

    // Créer une image plus grande
    let big_image = Tensor::from_data(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
        ],
        vec![5, 5],
        None,
    );

    println!("Image originale 5x5:");
    print_2d_tensor(&big_image);

    // Crop manuel
    let cropped = big_image.crop(&[1, 1], &[4, 4]).unwrap(); // Extraire région 3x3 au centre
    println!("\nAprès crop [1:4, 1:4] -> 3x3:");
    print_2d_tensor(&cropped);

    // Center crop
    let center_cropped = big_image.center_crop(&[3, 3]).unwrap();
    println!("\nAprès center crop 3x3:");
    print_2d_tensor(&center_cropped);

    // Test des différents modes de padding (API étendue)
    println!("\n🎨 Test des différents modes de padding:");

    let small_tensor = Tensor::from_data(&[1.0f32, 2.0], vec![2], None);
    println!("Tensor test: {:?}", small_tensor.storage().to_vec_f64());

    // Mode constant avec différentes valeurs
    let spec_constant = PaddingSpec::constant(vec![(1, 1)], 5.0);
    let padded_const = small_tensor.pad(&spec_constant).unwrap();
    println!(
        "Constant padding (5.0): {:?}",
        padded_const.storage().to_vec_f64()
    );

    // Utilisations typiques
    println!("\n🧠 Cas d'usage typiques:");
    println!("• Zero padding: Préparation pour convolutions");
    println!("• Constant padding: Valeurs de remplissage personnalisées");
    println!("• Center crop: Extraction de région d'intérêt");
    println!("• Padding asymétrique: Ajustements de taille spécifiques");

    // Test avec des tenseurs plus larges
    println!("\n📊 Test avec tenseurs plus larges:");

    // Simuler une image plus grande
    let large_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let large_tensor = Tensor::from_data(&large_data, vec![4, 4], None); // 4x4

    println!("Tensor 4x4 original:");
    print_2d_tensor(&large_tensor);

    // Padding avec pattern complexe
    let complex_padded = large_tensor.zero_pad(vec![(2, 1), (1, 2)]).unwrap(); // 2+1 hauteur, 1+2 largeur
    println!("\nAprès padding complexe (2,1)x(1,2) -> 7x7:");
    print_2d_tensor(&complex_padded);

    println!("\n✅ Démonstration padding/cropping terminée !");
    println!("📦 Fonctionnalités implémentées:");
    println!("   • zero_pad() - Padding avec zéros");
    println!("   • constant_pad() - Padding avec valeur constante");
    println!("   • crop() - Découpage manuel avec coordonnées");
    println!("   • center_crop() - Découpage centré");
    println!("   • PaddingSpec - Spécification avancée de padding");
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
        print!("[");
        for c in 0..cols {
            let val = data[r * cols + c];
            if c > 0 {
                print!(", ");
            }
            print!("{:4.0}", val);
        }
        println!("]");
    }
}
