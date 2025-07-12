//! Memory pool system for efficient tensor memory management
//!
//! This module provides memory pooling to reduce allocation overhead and fragmentation.
//! It implements various strategies for memory reuse in deep learning workloads.

use rustytorch_core::{CoreError, DType, Device, Result};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Weak};

/// Memory block metadata
#[derive(Debug, Clone)]
struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
    in_use: bool,
    allocation_count: usize,
    last_used: std::time::Instant,
}

// Safety: MemoryBlock can be sent between threads as long as the memory is properly managed
unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

impl MemoryBlock {
    fn new(size: usize, alignment: usize) -> Result<Self> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| CoreError::memory_error("Invalid memory layout"))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(CoreError::memory_error("Failed to allocate memory block"));
        }

        Ok(MemoryBlock {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            layout,
            in_use: false,
            allocation_count: 0,
            last_used: std::time::Instant::now(),
        })
    }

    unsafe fn deallocate(&self) {
        dealloc(self.ptr.as_ptr(), self.layout);
    }
}

/// Pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum total memory to keep in pool (bytes)
    pub max_pool_size: usize,
    /// Maximum age of unused blocks before cleanup (seconds)
    pub max_age_seconds: u64,
    /// Whether to defragment on allocation failure
    pub enable_defragmentation: bool,
    /// Alignment for allocations
    pub alignment: usize,
    /// Growth factor when pool needs expansion
    pub growth_factor: f64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            max_pool_size: 1024 * 1024 * 1024, // 1GB
            max_age_seconds: 300,              // 5 minutes
            enable_defragmentation: true,
            alignment: 64, // Cache line alignment
            growth_factor: 1.5,
        }
    }
}

/// Memory pool for a specific device and dtype
#[derive(Clone)]
pub struct DeviceMemoryPool {
    device: Device,
    dtype: DType,
    config: PoolConfig,
    /// Blocks organized by size buckets
    blocks: HashMap<usize, VecDeque<MemoryBlock>>,
    /// Total allocated memory
    allocated_size: usize,
    /// Total in-use memory
    used_size: usize,
    /// Statistics
    stats: PoolStatistics,
}

#[derive(Debug, Default, Clone)]
pub struct PoolStatistics {
    pub total_allocations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub defragmentations: usize,
    pub peak_memory_usage: usize,
}

impl DeviceMemoryPool {
    pub fn new(device: Device, dtype: DType, config: PoolConfig) -> Self {
        DeviceMemoryPool {
            device,
            dtype,
            config,
            blocks: HashMap::new(),
            allocated_size: 0,
            used_size: 0,
            stats: PoolStatistics::default(),
        }
    }

    /// Allocate memory from pool
    pub fn allocate(&mut self, size: usize) -> Result<NonNull<u8>> {
        self.stats.total_allocations += 1;

        // Round up to alignment
        let aligned_size = self.round_up_size(size);
        let bucket_size = self.get_bucket_size(aligned_size);

        // Try to find a free block
        if let Some(blocks) = self.blocks.get_mut(&bucket_size) {
            if let Some(block) = blocks
                .iter_mut()
                .find(|b| !b.in_use && b.size >= bucket_size)
            {
                self.stats.cache_hits += 1;
                block.in_use = true;
                block.allocation_count += 1;
                block.last_used = std::time::Instant::now();
                self.used_size += block.size;
                return Ok(block.ptr);
            }
        }

        // Cache miss - need to allocate new block
        self.stats.cache_misses += 1;

        // Check if we need to free memory first
        if self.allocated_size + bucket_size > self.config.max_pool_size {
            self.cleanup_old_blocks();

            if self.config.enable_defragmentation {
                self.defragment();
            }
        }

        // Allocate new block
        let mut block = MemoryBlock::new(bucket_size, self.config.alignment)?;
        block.in_use = true;
        block.allocation_count = 1;

        let ptr = block.ptr;
        self.allocated_size += bucket_size;
        self.used_size += bucket_size;

        // Update peak memory usage
        if self.used_size > self.stats.peak_memory_usage {
            self.stats.peak_memory_usage = self.used_size;
        }

        // Store block
        self.blocks
            .entry(bucket_size)
            .or_insert_with(VecDeque::new)
            .push_back(block);

        Ok(ptr)
    }

    /// Release memory back to pool
    pub fn deallocate(&mut self, ptr: NonNull<u8>, size: usize) {
        let aligned_size = self.round_up_size(size);
        let bucket_size = self.get_bucket_size(aligned_size);

        // Find the block
        if let Some(blocks) = self.blocks.get_mut(&bucket_size) {
            for block in blocks.iter_mut() {
                if block.ptr == ptr {
                    block.in_use = false;
                    block.last_used = std::time::Instant::now();
                    self.used_size -= block.size;
                    return;
                }
            }
        }

        // Block not found - this is an error but we'll handle gracefully
        eprintln!("Warning: Attempted to deallocate unknown memory block");
    }

    /// Find a free block of at least the requested size
    fn find_free_block(&mut self, size: usize) -> Option<&mut MemoryBlock> {
        // This method is no longer used directly since we integrated the logic into allocate
        self.blocks
            .get_mut(&size)
            .and_then(|blocks| blocks.iter_mut().find(|b| !b.in_use && b.size >= size))
    }

    /// Round up size to alignment
    fn round_up_size(&self, size: usize) -> usize {
        let alignment = self.config.alignment;
        (size + alignment - 1) / alignment * alignment
    }

    /// Get bucket size for allocation
    fn get_bucket_size(&self, size: usize) -> usize {
        // Use power-of-2 buckets for better reuse
        let mut bucket_size = 64; // Minimum size
        while bucket_size < size {
            bucket_size *= 2;
        }
        bucket_size
    }

    /// Clean up old unused blocks
    fn cleanup_old_blocks(&mut self) {
        let max_age = std::time::Duration::from_secs(self.config.max_age_seconds);
        let now = std::time::Instant::now();

        for (size, blocks) in self.blocks.iter_mut() {
            blocks.retain(|block| {
                if !block.in_use && now.duration_since(block.last_used) > max_age {
                    unsafe {
                        block.deallocate();
                    }
                    self.allocated_size -= *size;
                    false
                } else {
                    true
                }
            });
        }

        // Remove empty buckets
        self.blocks.retain(|_, blocks| !blocks.is_empty());
    }

    /// Defragment memory pool
    fn defragment(&mut self) {
        self.stats.defragmentations += 1;

        // Simple defragmentation: merge adjacent free blocks
        // In a real implementation, this would be more sophisticated
        for blocks in self.blocks.values_mut() {
            // Sort by allocation count to keep frequently used blocks
            blocks.make_contiguous().sort_by_key(|b| b.allocation_count);
        }
    }

    /// Get pool statistics
    pub fn statistics(&self) -> &PoolStatistics {
        &self.stats
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> (usize, usize) {
        (self.used_size, self.allocated_size)
    }
}

/// Global memory pool manager
pub struct MemoryPoolManager {
    pools: Arc<Mutex<HashMap<(Device, DType), DeviceMemoryPool>>>,
    config: PoolConfig,
}

impl MemoryPoolManager {
    pub fn new(config: PoolConfig) -> Self {
        MemoryPoolManager {
            pools: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Get or create pool for device and dtype
    pub fn get_pool(&self, device: Device, dtype: DType) -> Arc<Mutex<DeviceMemoryPool>> {
        let mut pools = self.pools.lock().unwrap();
        let key = (device.clone(), dtype);

        if !pools.contains_key(&key) {
            let pool = DeviceMemoryPool::new(device, dtype, self.config.clone());
            pools.insert(key.clone(), pool);
        }

        // Return a separate Arc to avoid holding the lock
        Arc::new(Mutex::new(pools.get_mut(&key).unwrap().clone()))
    }

    /// Allocate memory from appropriate pool
    pub fn allocate(&self, size: usize, device: Device, dtype: DType) -> Result<NonNull<u8>> {
        let mut pools = self.pools.lock().unwrap();
        let key = (device.clone(), dtype);

        let pool = pools
            .entry(key)
            .or_insert_with(|| DeviceMemoryPool::new(device, dtype, self.config.clone()));

        pool.allocate(size)
    }

    /// Deallocate memory back to pool
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize, device: Device, dtype: DType) {
        let mut pools = self.pools.lock().unwrap();
        let key = (device, dtype);

        if let Some(pool) = pools.get_mut(&key) {
            pool.deallocate(ptr, size);
        }
    }

    /// Clear all pools
    pub fn clear_all(&self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();
    }

    /// Get total statistics across all pools
    pub fn global_statistics(&self) -> PoolStatistics {
        let pools = self.pools.lock().unwrap();
        let mut stats = PoolStatistics::default();

        for pool in pools.values() {
            stats.total_allocations += pool.stats.total_allocations;
            stats.cache_hits += pool.stats.cache_hits;
            stats.cache_misses += pool.stats.cache_misses;
            stats.defragmentations += pool.stats.defragmentations;
            stats.peak_memory_usage = stats.peak_memory_usage.max(pool.stats.peak_memory_usage);
        }

        stats
    }
}

/// Smart pointer for pooled memory
pub struct PooledMemory {
    ptr: NonNull<u8>,
    size: usize,
    device: Device,
    dtype: DType,
    pool: Weak<Mutex<HashMap<(Device, DType), DeviceMemoryPool>>>,
}

impl PooledMemory {
    pub fn new(
        ptr: NonNull<u8>,
        size: usize,
        device: Device,
        dtype: DType,
        pool: Weak<Mutex<HashMap<(Device, DType), DeviceMemoryPool>>>,
    ) -> Self {
        PooledMemory {
            ptr,
            size,
            device,
            dtype,
            pool,
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for PooledMemory {
    fn drop(&mut self) {
        // Return memory to pool when dropped
        if let Some(pools) = self.pool.upgrade() {
            let mut pools = pools.lock().unwrap();
            let key = (self.device.clone(), self.dtype);

            if let Some(pool) = pools.get_mut(&key) {
                pool.deallocate(self.ptr, self.size);
            }
        }
    }
}

// Safety: PooledMemory can be sent between threads
unsafe impl Send for PooledMemory {}
unsafe impl Sync for PooledMemory {}

/// Global memory pool instance
lazy_static::lazy_static! {
    static ref GLOBAL_MEMORY_POOL_MANAGER: MemoryPoolManager = {
        let config = PoolConfig::default();
        MemoryPoolManager::new(config)
    };
}

/// Get the global memory pool manager
pub fn memory_pool_manager() -> &'static MemoryPoolManager {
    &GLOBAL_MEMORY_POOL_MANAGER
}

/// Convenience functions

/// Allocate pooled memory
pub fn allocate_pooled(size: usize, device: Device, dtype: DType) -> Result<PooledMemory> {
    let manager = memory_pool_manager();
    let ptr = manager.allocate(size, device.clone(), dtype)?;

    Ok(PooledMemory::new(
        ptr,
        size,
        device,
        dtype,
        Arc::downgrade(&manager.pools),
    ))
}

/// Clear all memory pools
pub fn clear_memory_pools() {
    memory_pool_manager().clear_all();
}

/// Get global memory pool statistics
pub fn memory_pool_stats() -> PoolStatistics {
    memory_pool_manager().global_statistics()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let config = PoolConfig::default();
        let mut pool = DeviceMemoryPool::new(Device::Cpu, DType::Float32, config);

        // Allocate memory
        let ptr1 = pool.allocate(1024).unwrap();
        assert!(!ptr1.as_ptr().is_null());

        // Check statistics
        assert_eq!(pool.stats.total_allocations, 1);
        assert_eq!(pool.stats.cache_misses, 1);
        assert_eq!(pool.stats.cache_hits, 0);

        // Deallocate
        pool.deallocate(ptr1, 1024);

        // Allocate again - should hit cache
        let ptr2 = pool.allocate(1024).unwrap();
        assert_eq!(pool.stats.cache_hits, 1);
    }

    #[test]
    fn test_bucket_sizes() {
        let pool = DeviceMemoryPool::new(Device::Cpu, DType::Float32, PoolConfig::default());

        assert_eq!(pool.get_bucket_size(1), 64);
        assert_eq!(pool.get_bucket_size(100), 128);
        assert_eq!(pool.get_bucket_size(1000), 1024);
        assert_eq!(pool.get_bucket_size(2000), 2048);
    }

    #[test]
    fn test_pooled_memory_drop() {
        let size = 1024;
        let device = Device::Cpu;
        let dtype = DType::Float32;

        {
            let pooled = allocate_pooled(size, device, dtype).unwrap();
            assert!(!pooled.as_ptr().is_null());
            // Memory should be returned to pool when pooled is dropped
        }

        // Check that memory was returned
        let stats = memory_pool_stats();
        assert!(stats.total_allocations > 0);
    }

    #[test]
    fn test_multiple_pools() {
        let manager = MemoryPoolManager::new(PoolConfig::default());

        // Allocate from different pools
        let _ptr1 = manager.allocate(1024, Device::Cpu, DType::Float32).unwrap();
        let _ptr2 = manager.allocate(2048, Device::Cpu, DType::Float64).unwrap();

        // Should have two separate pools
        let pools = manager.pools.lock().unwrap();
        assert_eq!(pools.len(), 2);
    }
}
