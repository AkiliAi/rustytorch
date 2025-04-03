use rustytorch_core::{Reshapable,};


fn main() {
    println!("Hello, world! RustyTorch!");

    println!("RustyTorch - Exemple de base de tenseurs");
    //
    // // Créer un tenseur à partir de données
    // let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // let tensor = Tensor::from_data(&data, vec![2, 3], None);
    // println!("Tenseur initial - shape: {:?}", tensor.shape());
    //
    // // Créer des tenseurs avec des valeurs prédéfinies
    // let zeros = Tensor::zeros(vec![2, 2], None);
    // println!("Tenseur de zéros - shape: {:?}", zeros.shape());
    //
    // let ones = Tensor::ones(vec![3, 2], None);
    // println!("Tenseur de uns - shape: {:?}", ones.shape());
    //
    // // Créer un tenseur avec des valeurs aléatoires
    // let random = Tensor::rand(vec![2, 3], None);
    // println!("Tenseur aléatoire - shape: {:?}", random.shape());
    //
    // // Opérations de transformation
    // let reshaped = tensor.reshape(&[3, 2]);
    // println!("Tenseur après reshape - shape: {:?}", reshaped.shape());
    //
    // let flattened = tensor.flatten();
    // println!("Tenseur aplati - shape: {:?}", flattened.shape());
    //
    // let transposed = tensor.transpose(0, 1);
    // println!("Tenseur transposé - shape: {:?}", transposed.shape());
}