//! Extended device functionality for heterogeneous computing
//!
//! This module provides extended device capabilities including:
//! - Device discovery and enumeration
//! - Memory management per device
//! - Device synchronization
//! - Multi-device support

use crate::{CoreError, Device, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Device information structure
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub device_type: DeviceType,
    pub index: usize,
    pub total_memory: usize,
    pub available_memory: usize,
    pub compute_capability: Option<ComputeCapability>,
    pub is_available: bool,
}

/// Device types with extended information
#[derive(Clone, Debug, PartialEq)]
pub enum DeviceType {
    Cpu,
    CudaGpu,
    MetalGpu,
    RocmGpu,
    XpuDevice,
}

/// Compute capability for GPUs
#[derive(Clone, Debug)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

/// Device manager for handling multiple devices
pub struct DeviceManager {
    devices: Arc<Mutex<HashMap<Device, DeviceInfo>>>,
    current_device: Arc<Mutex<Device>>,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Self {
        let mut devices = HashMap::new();

        // Always register CPU device
        devices.insert(
            Device::Cpu,
            DeviceInfo {
                name: "CPU".to_string(),
                device_type: DeviceType::Cpu,
                index: 0,
                total_memory: Self::get_system_memory(),
                available_memory: Self::get_available_system_memory(),
                compute_capability: None,
                is_available: true,
            },
        );

        Self {
            devices: Arc::new(Mutex::new(devices)),
            current_device: Arc::new(Mutex::new(Device::Cpu)),
        }
    }

    /// Discover and register available devices
    pub fn discover_devices(&mut self) -> Result<Vec<DeviceInfo>> {
        let mut discovered = Vec::new();
        let mut devices = self.devices.lock().unwrap();

        // Discover CUDA devices
        if let Ok(cuda_devices) = self.discover_cuda_devices() {
            for (idx, info) in cuda_devices.into_iter().enumerate() {
                devices.insert(Device::Cuda(idx), info.clone());
                discovered.push(info);
            }
        }

        // Discover Metal devices (macOS)
        #[cfg(target_os = "macos")]
        if let Ok(metal_devices) = self.discover_metal_devices() {
            for (idx, info) in metal_devices.into_iter().enumerate() {
                devices.insert(Device::Metal(idx), info.clone());
                discovered.push(info);
            }
        }

        // Discover ROCm devices (AMD)
        if let Ok(rocm_devices) = self.discover_rocm_devices() {
            for (idx, info) in rocm_devices.into_iter().enumerate() {
                devices.insert(Device::Rocm(idx), info.clone());
                discovered.push(info);
            }
        }

        Ok(discovered)
    }

    /// Get information about a specific device
    pub fn get_device_info(&self, device: &Device) -> Option<DeviceInfo> {
        self.devices.lock().unwrap().get(device).cloned()
    }

    /// Get the current device
    pub fn current_device(&self) -> Device {
        self.current_device.lock().unwrap().clone()
    }

    /// Set the current device
    pub fn set_current_device(&self, device: Device) -> Result<()> {
        let devices = self.devices.lock().unwrap();
        if !devices.contains_key(&device) {
            return Err(CoreError::invalid_op(
                "set_device",
                &format!("Device {} is not available", device),
            ));
        }

        *self.current_device.lock().unwrap() = device;
        Ok(())
    }

    /// Get all available devices
    pub fn available_devices(&self) -> Vec<Device> {
        self.devices
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, info)| info.is_available)
            .map(|(device, _)| device.clone())
            .collect()
    }

    /// Synchronize a device (wait for all operations to complete)
    pub fn synchronize(&self, device: &Device) -> Result<()> {
        match device {
            Device::Cpu => Ok(()), // CPU is always synchronized
            Device::Cuda(_) => self.cuda_synchronize(device),
            Device::Metal(_) => self.metal_synchronize(device),
            Device::Rocm(_) => self.rocm_synchronize(device),
            Device::Xpu(_) => self.xpu_synchronize(device),
        }
    }

    // Private helper methods

    fn get_system_memory() -> usize {
        // Simplified - in practice would use system calls
        8 * 1024 * 1024 * 1024 // 8GB default
    }

    fn get_available_system_memory() -> usize {
        // Simplified - in practice would use system calls
        4 * 1024 * 1024 * 1024 // 4GB default
    }

    fn discover_cuda_devices(&self) -> Result<Vec<DeviceInfo>> {
        // In a real implementation, this would use CUDA runtime API
        // For now, return mock data if CUDA_VISIBLE_DEVICES is set
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            Ok(vec![DeviceInfo {
                name: "NVIDIA GeForce RTX 3090".to_string(),
                device_type: DeviceType::CudaGpu,
                index: 0,
                total_memory: 24 * 1024 * 1024 * 1024, // 24GB
                available_memory: 20 * 1024 * 1024 * 1024, // 20GB
                compute_capability: Some(ComputeCapability { major: 8, minor: 6 }),
                is_available: true,
            }])
        } else {
            Ok(vec![])
        }
    }

    #[cfg(target_os = "macos")]
    fn discover_metal_devices(&self) -> Result<Vec<DeviceInfo>> {
        // In a real implementation, this would use Metal API
        Ok(vec![DeviceInfo {
            name: "Apple M1 GPU".to_string(),
            device_type: DeviceType::MetalGpu,
            index: 0,
            total_memory: 16 * 1024 * 1024 * 1024, // 16GB shared
            available_memory: 12 * 1024 * 1024 * 1024, // 12GB
            compute_capability: None,
            is_available: true,
        }])
    }

    #[cfg(not(target_os = "macos"))]
    fn discover_metal_devices(&self) -> Result<Vec<DeviceInfo>> {
        Ok(vec![])
    }

    fn discover_rocm_devices(&self) -> Result<Vec<DeviceInfo>> {
        // In a real implementation, this would use ROCm runtime API
        if std::env::var("ROCM_VISIBLE_DEVICES").is_ok() {
            Ok(vec![DeviceInfo {
                name: "AMD Radeon RX 7900 XTX".to_string(),
                device_type: DeviceType::RocmGpu,
                index: 0,
                total_memory: 24 * 1024 * 1024 * 1024, // 24GB
                available_memory: 20 * 1024 * 1024 * 1024, // 20GB
                compute_capability: None,
                is_available: true,
            }])
        } else {
            Ok(vec![])
        }
    }

    fn cuda_synchronize(&self, _device: &Device) -> Result<()> {
        // In real implementation: cudaDeviceSynchronize()
        Ok(())
    }

    fn metal_synchronize(&self, _device: &Device) -> Result<()> {
        // In real implementation: Metal command buffer waitUntilCompleted
        Ok(())
    }

    fn rocm_synchronize(&self, _device: &Device) -> Result<()> {
        // In real implementation: hipDeviceSynchronize()
        Ok(())
    }

    fn xpu_synchronize(&self, _device: &Device) -> Result<()> {
        // In real implementation: Intel XPU synchronization
        Ok(())
    }
}

/// Device memory allocator trait
pub trait DeviceAllocator: Send + Sync {
    /// Allocate memory on the device
    fn allocate(&self, size: usize) -> Result<*mut u8>;

    /// Deallocate memory on the device
    fn deallocate(&self, ptr: *mut u8);

    /// Copy memory from host to device
    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()>;

    /// Copy memory from device to host
    fn copy_to_host(&self, dst: &mut [u8], src: *const u8, size: usize) -> Result<()>;

    /// Copy memory between devices
    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) -> Result<()>;
}

/// CPU memory allocator
pub struct CpuAllocator;

impl DeviceAllocator for CpuAllocator {
    fn allocate(&self, size: usize) -> Result<*mut u8> {
        let layout = std::alloc::Layout::from_size_align(size, 64)
            .map_err(|_| CoreError::memory_error("Invalid allocation size"))?;

        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(CoreError::memory_error("Failed to allocate memory"));
        }
        Ok(ptr)
    }

    fn deallocate(&self, ptr: *mut u8) {
        // In practice, would need to track size
        // For now, this is a placeholder
        // Would use std::alloc::dealloc(ptr, layout)
        let _ = ptr; // Suppress warning
    }

    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], src: *const u8, size: usize) -> Result<()> {
        if dst.len() < size {
            return Err(CoreError::invalid_op(
                "copy",
                "Destination buffer too small",
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), size);
        }
        Ok(())
    }

    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
        Ok(())
    }
}

/// Device context for managing device-specific operations
pub struct DeviceContext {
    device: Device,
    allocator: Box<dyn DeviceAllocator>,
}

impl DeviceContext {
    /// Create a new device context
    pub fn new(device: Device) -> Self {
        let allocator: Box<dyn DeviceAllocator> = match &device {
            Device::Cpu => Box::new(CpuAllocator),
            // In real implementation, would have GPU allocators
            _ => Box::new(CpuAllocator), // Placeholder
        };

        Self { device, allocator }
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the allocator
    pub fn allocator(&self) -> &dyn DeviceAllocator {
        self.allocator.as_ref()
    }
}

// Global device manager instance
lazy_static::lazy_static! {
    static ref DEVICE_MANAGER: DeviceManager = {
        let mut manager = DeviceManager::new();
        let _ = manager.discover_devices();
        manager
    };
}

/// Get the global device manager
pub fn device_manager() -> &'static DeviceManager {
    &DEVICE_MANAGER
}

/// Convenience functions

/// Get the current device
pub fn current_device() -> Device {
    device_manager().current_device()
}

/// Set the current device
pub fn set_device(device: Device) -> Result<()> {
    device_manager().set_current_device(device)
}

/// Get all available devices
pub fn available_devices() -> Vec<Device> {
    device_manager().available_devices()
}

/// Synchronize the current device
pub fn synchronize() -> Result<()> {
    let device = current_device();
    device_manager().synchronize(&device)
}

/// Check if CUDA is available
pub fn cuda_is_available() -> bool {
    available_devices()
        .iter()
        .any(|d| matches!(d, Device::Cuda(_)))
}

/// Check if Metal is available
pub fn metal_is_available() -> bool {
    available_devices()
        .iter()
        .any(|d| matches!(d, Device::Metal(_)))
}

/// Check if ROCm is available
pub fn rocm_is_available() -> bool {
    available_devices()
        .iter()
        .any(|d| matches!(d, Device::Rocm(_)))
}

/// Get the number of available GPUs of a specific type
pub fn device_count(device_type: &str) -> usize {
    available_devices()
        .iter()
        .filter(|d| match device_type {
            "cuda" => matches!(d, Device::Cuda(_)),
            "metal" => matches!(d, Device::Metal(_)),
            "rocm" => matches!(d, Device::Rocm(_)),
            "xpu" => matches!(d, Device::Xpu(_)),
            _ => false,
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager() {
        let manager = DeviceManager::new();

        // CPU should always be available
        let cpu_info = manager.get_device_info(&Device::Cpu).unwrap();
        assert_eq!(cpu_info.device_type, DeviceType::Cpu);
        assert!(cpu_info.is_available);

        // Current device should be CPU by default
        assert_eq!(manager.current_device(), Device::Cpu);
    }

    #[test]
    fn test_device_discovery() {
        let mut manager = DeviceManager::new();
        let devices = manager.discover_devices().unwrap();

        // Should discover at least CPU
        assert!(!devices.is_empty());
    }

    #[test]
    fn test_cpu_allocator() {
        let allocator = CpuAllocator;

        // Test allocation
        let size = 1024;
        let ptr = allocator.allocate(size).unwrap();
        assert!(!ptr.is_null());

        // Test copy from host
        let data = vec![42u8; size];
        allocator.copy_from_host(ptr, &data).unwrap();

        // Test copy to host
        let mut result = vec![0u8; size];
        allocator.copy_to_host(&mut result, ptr, size).unwrap();
        assert_eq!(result, data);

        // Clean up
        allocator.deallocate(ptr);
    }

    #[test]
    fn test_device_context() {
        let context = DeviceContext::new(Device::Cpu);
        assert_eq!(context.device(), &Device::Cpu);

        // Test allocator through context
        let size = 256;
        let ptr = context.allocator().allocate(size).unwrap();
        assert!(!ptr.is_null());
        context.allocator().deallocate(ptr);
    }

    #[test]
    fn test_convenience_functions() {
        // Test current device
        assert_eq!(current_device(), Device::Cpu);

        // Test available devices
        let devices = available_devices();
        assert!(!devices.is_empty());
        assert!(devices.contains(&Device::Cpu));

        // Test synchronize
        synchronize().unwrap();
    }
}
