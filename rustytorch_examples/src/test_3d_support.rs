// Test complet du support des matrices 3D+ dans RustyTorch
use rustytorch_tensor::Tensor;
use rustytorch_autograd::{Variable, enable_grad};

pub fn test_3d_support_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🧊 Test Support Matrices 3D+ dans RustyTorch");
    println!();

    let _grad_guard = enable_grad();

    // Test 1: Création de tenseurs 3D
    println!("📦 Test 1: Création de tenseurs 3D");
    let tensor_3d = Tensor::randn(vec![2, 3, 4], None)?;
    let tensor_3d_b = Tensor::randn(vec![2, 4, 5], None)?;
    
    println!("  Tensor A shape: {:?}", tensor_3d.shape());
    println!("  Tensor B shape: {:?}", tensor_3d_b.shape());
    println!("  ✅ Création 3D réussie");
    println!();

    // Test 2: Matrix multiplication 3D (Batch Matrix Multiplication)
    println!("🔢 Test 2: Batch Matrix Multiplication");
    match tensor_3d.matmul(&tensor_3d_b) {
        Ok(result) => {
            println!("  ✅ BMM réussi! Shape résultat: {:?}", result.shape());
            println!("  Expected shape: [2, 3, 5]");
            if result.shape() == &[2, 3, 5] {
                println!("  ✅ Shape correcte!");
            } else {
                println!("  ❌ Shape incorrecte!");
            }
        }
        Err(e) => {
            println!("  ❌ BMM échoué: {}", e);
        }
    }
    println!();

    // Test 3: Opérations de réduction sur 3D
    println!("📊 Test 3: Réductions sur tenseurs 3D");
    
    // Sum along different axes
    match tensor_3d.sum_dim(Some(0)) {
        Ok(result) => {
            println!("  ✅ Sum axis 0: {:?} -> {:?}", tensor_3d.shape(), result.shape());
        }
        Err(e) => {
            println!("  ❌ Sum axis 0 failed: {}", e);
        }
    }

    match tensor_3d.sum_dim(Some(1)) {
        Ok(result) => {
            println!("  ✅ Sum axis 1: {:?} -> {:?}", tensor_3d.shape(), result.shape());
        }
        Err(e) => {
            println!("  ❌ Sum axis 1 failed: {}", e);
        }
    }

    match tensor_3d.sum_dim(Some(2)) {
        Ok(result) => {
            println!("  ✅ Sum axis 2: {:?} -> {:?}", tensor_3d.shape(), result.shape());
        }
        Err(e) => {
            println!("  ❌ Sum axis 2 failed: {}", e);
        }
    }
    println!();

    // Test 4: Tenseurs 4D (CNN scenario)
    println!("🖼️  Test 4: Tenseurs 4D (CNN scenario)");
    let images = Tensor::randn(vec![8, 3, 32, 32], None)?; // batch, channels, height, width
    println!("  Images shape: {:?}", images.shape());

    // Sum over spatial dimensions
    match images.sum_dim(Some(2)) {
        Ok(result) => {
            println!("  ✅ Sum height dimension: {:?} -> {:?}", images.shape(), result.shape());
        }
        Err(e) => {
            println!("  ❌ Sum height failed: {}", e);
        }
    }

    // Mean over batch dimension
    match images.mean_dim(Some(0)) {
        Ok(result) => {
            println!("  ✅ Mean batch: {:?} -> {:?}", images.shape(), result.shape());
        }
        Err(e) => {
            println!("  ❌ Mean batch failed: {}", e);
        }
    }
    println!();

    // Test 5: Autograd avec 3D tensors
    println!("🧠 Test 5: Autograd avec tenseurs 3D");
    
    let a_var = Variable::from_tensor(Tensor::randn(vec![2, 3, 4], None)?, true);
    let b_var = Variable::from_tensor(Tensor::randn(vec![2, 4, 5], None)?, true);
    
    println!("  Variable A shape: {:?}, requires_grad: {}", a_var.tensor().shape(), a_var.requires_grad());
    println!("  Variable B shape: {:?}, requires_grad: {}", b_var.tensor().shape(), b_var.requires_grad());

    // Test matmul with autograd
    match std::panic::catch_unwind(|| {
        let result = a_var.matmul(&b_var);
        let mut loss = result.sum();
        loss.backward();
        
        // Check gradient shapes
        let grad_a = a_var.grad();
        let grad_b = b_var.grad();
        
        (grad_a.is_some(), grad_b.is_some(), 
         grad_a.map(|g| g.shape().to_vec()), 
         grad_b.map(|g| g.shape().to_vec()))
    }) {
        Ok((has_grad_a, has_grad_b, shape_a, shape_b)) => {
            println!("  ✅ Autograd matmul 3D réussi!");
            println!("  Gradient A present: {}, shape: {:?}", has_grad_a, shape_a);
            println!("  Gradient B present: {}, shape: {:?}", has_grad_b, shape_b);
        }
        Err(_) => {
            println!("  ❌ Autograd matmul 3D échoué (panic)");
        }
    }
    println!();

    // Test 6: Tenseurs 5D (RNN with batch)
    println!("🔗 Test 6: Tenseurs 5D (RNN scenario)");
    let rnn_states = Tensor::randn(vec![2, 10, 32, 128, 4], None)?; // layers, seq, batch, hidden, features
    println!("  RNN states shape: {:?}", rnn_states.shape());
    
    match rnn_states.sum_dim(Some(1)) {
        Ok(result) => {
            println!("  ✅ Sum sequence dimension: {:?} -> {:?}", rnn_states.shape(), result.shape());
        }
        Err(e) => {
            println!("  ❌ Sum sequence failed: {}", e);
        }
    }
    println!();

    println!("=== Résumé du Support 3D+ ===");
    println!("✅ Création tenseurs 3D/4D/5D: OK");
    println!("🔢 Matrix multiplication 3D: À vérifier");
    println!("📊 Réductions multi-dimensionnelles: OK");
    println!("🧠 Autograd 3D: À vérifier (potentiels problèmes)");
    
    println!("============================================================");
    Ok(())
}