//! Tests for core traits and types

#[cfg(test)]
mod tests {
    use crate::{CoreError, DType, Device, TensorMetadata, TensorOptions};

    #[test]
    fn test_dtype_properties() {
        // Test size in bytes
        assert_eq!(DType::Bool.size_in_bytes(), 1);
        assert_eq!(DType::Float16.size_in_bytes(), 2);
        assert_eq!(DType::Float32.size_in_bytes(), 4);
        assert_eq!(DType::Float64.size_in_bytes(), 8);
        assert_eq!(DType::Int32.size_in_bytes(), 4);
        assert_eq!(DType::UInt8.size_in_bytes(), 1);

        // Test type classification
        assert!(DType::Float32.is_floating_point());
        assert!(!DType::Int32.is_floating_point());
        assert!(DType::Int32.is_integer());
        assert!(!DType::Float32.is_integer());
        assert!(DType::Float32.is_signed());
        assert!(!DType::UInt32.is_signed());

        // Test string representation
        assert_eq!(DType::Float32.as_str(), "float32");
        assert_eq!(DType::Bool.as_str(), "bool");
    }

    #[test]
    fn test_device_properties() {
        let cpu = Device::Cpu;
        let cuda0 = Device::Cuda(0);
        let metal1 = Device::Metal(1);

        assert!(cpu.is_cpu());
        assert!(!cuda0.is_cpu());
        assert!(cuda0.is_gpu());
        assert!(metal1.is_gpu());

        assert_eq!(cpu.index(), None);
        assert_eq!(cuda0.index(), Some(0));
        assert_eq!(metal1.index(), Some(1));

        assert_eq!(cpu.device_type(), "cpu");
        assert_eq!(cuda0.device_type(), "cuda");
        assert_eq!(metal1.device_type(), "metal");

        assert_eq!(format!("{}", cuda0), "cuda:0");
        assert_eq!(format!("{}", metal1), "metal:1");
    }

    #[test]
    fn test_tensor_options() {
        let default_opts = TensorOptions::default();
        assert_eq!(default_opts.dtype, DType::Float32);
        assert_eq!(default_opts.requires_grad, false);
        assert_eq!(default_opts.device, Device::Cpu);

        let custom_opts = TensorOptions::new()
            .dtype(DType::Float64)
            .requires_grad(true)
            .device(Device::cuda(0));

        assert_eq!(custom_opts.dtype, DType::Float64);
        assert_eq!(custom_opts.requires_grad, true);
        assert_eq!(custom_opts.device, Device::Cuda(0));
    }

    #[test]
    fn test_tensor_metadata() {
        // Test 2D tensor metadata
        let meta = TensorMetadata::from_shape(vec![3, 4]);
        assert_eq!(meta.shape, vec![3, 4]);
        assert_eq!(meta.strides, vec![4, 1]); // Row-major
        assert_eq!(meta.numel, 12);
        assert_eq!(meta.ndim, 2);
        assert!(meta.is_contiguous);
        assert!(meta.is_matrix());

        // Test 1D tensor metadata
        let meta = TensorMetadata::from_shape(vec![5]);
        assert_eq!(meta.shape, vec![5]);
        assert_eq!(meta.strides, vec![1]);
        assert_eq!(meta.numel, 5);
        assert!(meta.is_vector());

        // Test scalar metadata
        let meta = TensorMetadata::from_shape(vec![]);
        assert_eq!(meta.shape, vec![]);
        assert_eq!(meta.strides, vec![]);
        assert_eq!(meta.numel, 1);
        assert_eq!(meta.ndim, 0);
        assert!(meta.is_scalar());

        // Test dimension access
        let meta = TensorMetadata::from_shape(vec![2, 3, 4]);
        assert_eq!(meta.size(0), Some(2));
        assert_eq!(meta.size(1), Some(3));
        assert_eq!(meta.size(2), Some(4));
        assert_eq!(meta.size(3), None);
        assert_eq!(meta.stride(0), Some(12));
        assert_eq!(meta.stride(1), Some(4));
        assert_eq!(meta.stride(2), Some(1));
    }

    #[test]
    fn test_core_errors() {
        // Test error creation and display
        let err = CoreError::shape_mismatch(vec![2, 3], vec![3, 2], "matmul");
        assert_eq!(
            err.to_string(),
            "Shape mismatch in matmul: expected [2, 3], got [3, 2]"
        );

        let err = CoreError::dim_out_of_bounds(3, 2, "transpose");
        assert_eq!(
            err.to_string(),
            "Dimension 3 out of bounds for 2-dimensional tensor in transpose"
        );

        let err = CoreError::broadcast_error(vec![3, 1], vec![1, 4], "incompatible shapes");
        assert!(err.to_string().contains("Cannot broadcast"));
    }
}

#[cfg(test)]
mod trait_tests {
    use super::*;
    use crate::errors::Result;
    use crate::traits::*;

    // Mock implementation for testing traits
    struct MockTensor {
        shape: Vec<usize>,
        data: Vec<f32>,
    }

    impl NumericOps for MockTensor {
        type Output = MockTensor;

        fn add(self, _rhs: Self) -> Result<Self::Output> {
            Ok(self)
        }

        fn sub(self, _rhs: Self) -> Result<Self::Output> {
            Ok(self)
        }

        fn mul(self, _rhs: Self) -> Result<Self::Output> {
            Ok(self)
        }

        fn div(self, _rhs: Self) -> Result<Self::Output> {
            Ok(self)
        }

        fn neg(self) -> Result<Self::Output> {
            Ok(self)
        }

        fn abs(self) -> Result<Self::Output> {
            Ok(self)
        }

        fn pow(self, _exponent: Self) -> Result<Self::Output> {
            Ok(self)
        }

        fn rem(self, _rhs: Self) -> Result<Self::Output> {
            Ok(self)
        }
    }

    #[test]
    fn test_numeric_ops_trait() {
        let tensor1 = MockTensor {
            shape: vec![2, 3],
            data: vec![1.0; 6],
        };
        let tensor2 = MockTensor {
            shape: vec![2, 3],
            data: vec![2.0; 6],
        };

        // Test that operations compile and return Ok
        assert!(tensor1.add(tensor2).is_ok());
    }
}
