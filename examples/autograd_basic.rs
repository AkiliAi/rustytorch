// rustytorch_examples/src/bin/autograd_example.rs

use rustytorch_tensor::Tensor;
use rustytorch_autograd::{Variable, no_grad, Operation};

fn main() {
    println!("RustyTorch - Exemple de différentiation automatique\n");

    // ====== Exemple 1: Opérations simples avec différentiation ======
    println!("Exemple 1: Opérations simples avec différentiation");

    // Créer des variables avec suivi de gradient
    let tensor_a = Tensor::from_data(&[2.0], vec![1], None);
    let tensor_b = Tensor::from_data(&[3.0], vec![1], None);

    let mut var_a = Variable::from_tensor(tensor_a, true);
    let mut var_b = Variable::from_tensor(tensor_b, true);

    // Effectuer des opérations: c = a * b
    let mut var_c = var_a.mul(&var_b);

    println!("a = 2.0, b = 3.0");
    println!("c = a * b = {}", extract_scalar(&var_c.tensor));

    // Calculer les gradients
    var_c.backward();

    // Afficher les gradients
    println!("dc/da = {}", extract_scalar(&var_a.grad.as_ref().unwrap()));
    println!("dc/db = {}", extract_scalar(&var_b.grad.as_ref().unwrap()));

    // ====== Exemple 2: Expression plus complexe ======
    println!("\nExemple 2: Expression plus complexe");

    // Fonction: f(x, y) = (x + 2*y) * (x^2)
    let tensor_x = Tensor::from_data(&[3.0], vec![1], None);
    let tensor_y = Tensor::from_data(&[4.0], vec![1], None);

    let mut var_x = Variable::from_tensor(tensor_x, true);
    let mut var_y = Variable::from_tensor(tensor_y, true);

    // Calculer 2*y
    let two = Variable::from_tensor(Tensor::from_data(&[2.0], vec![1], None), false);
    let two_y = two.mul(&var_y);

    // Calculer x + 2*y
    let x_plus_2y = var_x.add(&two_y);

    // Calculer x^2
    let x_squared = var_x.mul(&var_x);

    // Calculer le résultat final: (x + 2*y) * (x^2)
    let mut result = x_plus_2y.mul(&x_squared);

    println!("x = 3.0, y = 4.0");
    println!("f(x, y) = (x + 2*y) * (x^2) = {}", extract_scalar(&result.tensor));

    // Propager les gradients
    result.backward();

    // Afficher les gradients
    println!("df/dx = {}", extract_scalar(&var_x.grad.as_ref().unwrap()));
    println!("df/dy = {}", extract_scalar(&var_y.grad.as_ref().unwrap()));

    // ====== Exemple 3: Utilisation de no_grad ======
    println!("\nExemple 3: Utilisation de no_grad");

    {
        let _guard = no_grad();

        // Ces opérations ne seront pas suivies pour le calcul du gradient
        let var_p = Variable::from_tensor(Tensor::from_data(&[5.0], vec![1], None), true);
        let var_q = Variable::from_tensor(Tensor::from_data(&[6.0], vec![1], None), true);

        let var_r = var_p.add(&var_q);

        println!("p = 5.0, q = 6.0");
        println!("r = p + q = {}", extract_scalar(&var_r.tensor));
        println!("requires_grad = {}", var_r.requires_grad);
    }

    // ====== Exemple 4: Multiplication matricielle ======
    println!("\nExemple 4: Multiplication matricielle et rétropropagation");

    // Créer deux matrices
    let tensor_a = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], None);
    let tensor_b = Tensor::from_data(&[5.0, 6.0, 7.0, 8.0], vec![2, 2], None);

    let mut var_a = Variable::from_tensor(tensor_a, true);
    let mut var_b = Variable::from_tensor(tensor_b, true);

    // Effectuer la multiplication matricielle
    let mut var_c = var_a.matmul(&var_b);

    println!("Matrice A = [[1, 2], [3, 4]]");
    println!("Matrice B = [[5, 6], [7, 8]]");
    println!("C = A @ B = matrice 2x2");

    // Pour simplifier, on n'affiche pas la matrice complète ici

    // Propager les gradients (en utilisant une somme des éléments comme fonction de perte)
    // L = sum(C)
    let mut sum_c = var_c.sum();
    sum_c.backward();

    println!("dL/dA et dL/dB calculés (gradients des matrices)");

    println!("\nExemple de différentiation automatique terminé!");
}

// Fonction utilitaire pour extraire un scalaire d'un tenseur
fn extract_scalar(tensor: &Tensor) -> f64 {
    let storage = tensor.storage();
    match storage {
        rustytorch_tensor::storage::StorageType::F32(data) => {
            if data.len() >= 1 {
                data[0] as f64
            } else {
                f64::NAN
            }
        },
        rustytorch_tensor::storage::StorageType::F64(data) => {
            if data.len() >= 1 {
                data[0]
            } else {
                f64::NAN
            }
        },
        _ => f64::NAN,
    }
}

// Ajout de la méthode sum() à Variable pour l'exemple
impl Variable {
    fn sum(&self) -> Self {
        let result_tensor = self.tensor.sum();

        // Si le calcul du gradient est désactivé, retourner un résultat simple
        if !self.requires_grad {
            return Self::from_tensor(result_tensor, false);
        }

        // Pour la rétropropagation, le gradient de sum par rapport à chaque élément est 1
        let self_clone = self.clone();
        let grad_fn = Box::new(move |_grad_output: &Tensor| {
            // Pour sum(), le gradient par rapport à chaque élément de l'entrée est 1
            let ones = Tensor::ones(self_clone.tensor.shape().to_vec(), None);
            vec![ones]
        }) as Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>;

        // Créer la variable résultante
        Self::from_operation(
            result_tensor,
            Operation::None, // On pourrait ajouter un type d'opération Sum si nécessaire
            vec![self.clone()],
            Some(grad_fn),
        )
    }
}