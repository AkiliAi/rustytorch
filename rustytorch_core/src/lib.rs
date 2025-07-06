//! RustyTorch Core - Fundamental traits and types for tensor operations
//!
//! This crate provides the core abstractions used throughout RustyTorch:
//! - Mathematical operation traits
//! - Type system for tensors
//! - Device abstractions
//! - Error handling types

pub mod errors;
pub mod traits;
pub mod types;
pub mod device_ext;

#[cfg(test)]
mod tests;

// Re-export commonly used items
pub use errors::{CoreError, Result};
pub use traits::{
    Broadcasting, Comparable, Differentiable, Indexable, NumericOps, Reduction, Reshapable,
    Serializable,
};
pub use types::{DType, Device, TensorMetadata, TensorOptions};
pub use device_ext::{
    DeviceManager, DeviceInfo, DeviceType, DeviceContext, DeviceAllocator,
    current_device, set_device, available_devices, synchronize,
    cuda_is_available, metal_is_available, rocm_is_available, device_count,
};
