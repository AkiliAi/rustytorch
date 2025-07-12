use rustytorch_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing 3D Matrix Multiplication Support");

    // Test 1: Basic 3D tensor creation
    let a = Tensor::randn(vec![2, 3, 4], None)?;
    let b = Tensor::randn(vec![2, 4, 5], None)?;
    
    println!("Tensor A shape: {:?}", a.shape());
    println!("Tensor B shape: {:?}", b.shape());

    // Test 2: Matmul with 3D tensors (batch matrix multiplication)
    match a.matmul(&b) {
        Ok(result) => {
            println!("✅ 3D matmul succeeded!");
            println!("Result shape: {:?}", result.shape());
            assert_eq!(result.shape(), &[2, 3, 5]);
        }
        Err(e) => {
            println!("❌ 3D matmul failed: {}", e);
        }
    }

    // Test 3: 4D tensor for CNN
    let images = Tensor::randn(vec![32, 3, 224, 224], None)?;
    println!("\nCNN test - Image batch shape: {:?}", images.shape());

    // Test 4: Sum along batch dimension
    match images.sum_axis(0) {
        Ok(result) => {
            println!("✅ Sum along batch dimension succeeded!");
            println!("Result shape: {:?}", result.shape());
        }
        Err(e) => {
            println!("❌ Sum along batch dimension failed: {}", e);
        }
    }

    // Test 5: Mean across spatial dimensions
    match images.mean_axis(2) {
        Ok(result) => {
            println!("✅ Mean across height dimension succeeded!");
            println!("Result shape: {:?}", result.shape());
        }
        Err(e) => {
            println!("❌ Mean across height dimension failed: {}", e);
        }
    }

    Ok(())
}