// Quick test for 3D support
use rustytorch_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing 3D Tensor Support ===");

    // Test 1: Create 3D tensors
    let a = Tensor::randn(vec![2, 3, 4], None)?;
    let b = Tensor::randn(vec![2, 4, 5], None)?;
    
    println!("A shape: {:?}", a.shape());
    println!("B shape: {:?}", b.shape());

    // Test 2: Try matmul (should work via gemm)
    match a.matmul(&b) {
        Ok(result) => {
            println!("✅ 3D matmul SUCCESS! Result shape: {:?}", result.shape());
            assert_eq!(result.shape(), &[2, 3, 5]);
        }
        Err(e) => {
            println!("❌ 3D matmul FAILED: {}", e);
        }
    }

    // Test 3: Sum operations on 3D
    match a.sum_axis(0) {
        Ok(result) => {
            println!("✅ Sum axis 0 SUCCESS! Shape: {:?}", result.shape());
        }
        Err(e) => {
            println!("❌ Sum axis 0 FAILED: {}", e);
        }
    }

    // Test 4: 4D tensor (CNN scenario)
    let images = Tensor::randn(vec![8, 3, 32, 32], None)?;
    println!("Images shape: {:?}", images.shape());

    match images.sum_axis(1) { // Sum over channels
        Ok(result) => {
            println!("✅ 4D sum SUCCESS! Shape: {:?}", result.shape());
        }
        Err(e) => {
            println!("❌ 4D sum FAILED: {}", e);
        }
    }

    println!("\n=== Testing Complete ===");
    Ok(())
}