//! RustyTorch Core - Fundamental traits and types for tensor operations
//!
//! This crate provides the core abstractions used throughout RustyTorch:
//! - Mathematical operation traits
//! - Type system for tensors
//! - Device abstractions
//! - Error handling types

pub mod device_ext;
pub mod errors;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export commonly used items
pub use device_ext::{
    available_devices, cuda_is_available, current_device, device_count, metal_is_available,
    rocm_is_available, set_device, synchronize, DeviceAllocator, DeviceContext, DeviceInfo,
    DeviceManager, DeviceType,
};
pub use errors::{CoreError, Result};
pub use traits::{
    Broadcasting, Comparable, Differentiable, Indexable, NumericOps, Reduction, Reshapable,
    Serializable,
};
pub use types::{DType, Device, TensorMetadata, TensorOptions};
