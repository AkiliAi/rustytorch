//! Core types and enumerations used throughout RustyTorch

use std::fmt;

/// Data types supported by tensors
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    // Floating point types
    Float16,   // Half precision
    Float32,   // Single precision  
    Float64,   // Double precision
    
    // Signed integer types
    Int8,
    Int16,
    Int32,
    Int64,
    
    // Unsigned integer types
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    
    // Boolean type
    Bool,
    
    // Complex types
    Complex64,
    Complex128,
}

impl DType {
    /// Get the size of the data type in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::UInt8 => 1,
            DType::Float16 | DType::Int16 | DType::UInt16 => 2,
            DType::Float32 | DType::Int32 | DType::UInt32 => 4,
            DType::Float64 | DType::Int64 | DType::UInt64 => 8,
            DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }
    
    /// Check if this is a floating point type
    pub fn is_floating_point(&self) -> bool {
        matches!(self, DType::Float16 | DType::Float32 | DType::Float64 | 
                       DType::Complex64 | DType::Complex128)
    }
    
    /// Check if this is an integer type (signed or unsigned)
    pub fn is_integer(&self) -> bool {
        matches!(self, 
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 |
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64
        )
    }
    
    /// Check if this is a signed type
    pub fn is_signed(&self) -> bool {
        matches!(self,
            DType::Float16 | DType::Float32 | DType::Float64 |
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }
    
    /// Get string representation for display
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Bool => "bool",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Computation device types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU device
    Cpu,
    
    /// NVIDIA CUDA GPU device
    Cuda(usize),
    
    /// Apple Metal GPU device
    Metal(usize),
    
    /// AMD ROCm GPU device  
    Rocm(usize),
    
    /// Intel XPU device
    Xpu(usize),
}

impl Device {
    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
    
    /// Check if this is any GPU device
    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }
    
    /// Get the device index (returns None for CPU)
    pub fn index(&self) -> Option<usize> {
        match self {
            Device::Cpu => None,
            Device::Cuda(idx) | Device::Metal(idx) | Device::Rocm(idx) | Device::Xpu(idx) => Some(*idx),
        }
    }
    
    /// Get device type as string
    pub fn device_type(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "metal",
            Device::Rocm(_) => "rocm",
            Device::Xpu(_) => "xpu",
        }
    }
    
    /// Create a CUDA device with the given index
    pub fn cuda(index: usize) -> Self {
        Device::Cuda(index)
    }
    
    /// Create a Metal device with the given index
    pub fn metal(index: usize) -> Self {
        Device::Metal(index)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{}", idx),
            Device::Metal(idx) => write!(f, "metal:{}", idx),
            Device::Rocm(idx) => write!(f, "rocm:{}", idx),
            Device::Xpu(idx) => write!(f, "xpu:{}", idx),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

/// Options for tensor creation
#[derive(Clone, Debug, PartialEq)]
pub struct TensorOptions {
    /// Data type of the tensor
    pub dtype: DType,
    
    /// Whether to track gradients for this tensor
    pub requires_grad: bool,
    
    /// Device where the tensor is stored
    pub device: Device,
    
    // Memory layout (future extension)
    // pub layout: Layout,
}

impl TensorOptions {
    /// Create new tensor options
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the data type
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
    
    /// Set gradient tracking
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }
    
    /// Set the device
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

impl Default for TensorOptions {
    fn default() -> Self {
        Self {
            dtype: DType::Float32,
            requires_grad: false,
            device: Device::Cpu,
        }
    }
}

/// Metadata about a tensor's structure
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMetadata {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    
    /// Strides for each dimension
    pub strides: Vec<usize>,
    
    /// Total number of elements
    pub numel: usize,
    
    /// Number of dimensions
    pub ndim: usize,
    
    /// Whether the tensor is contiguous in memory
    pub is_contiguous: bool,
}

impl TensorMetadata {
    /// Create metadata from shape
    pub fn from_shape(shape: Vec<usize>) -> Self {
        let ndim = shape.len();
        let numel = shape.iter().product();
        
        // Calculate strides for row-major (C-style) layout
        let strides = if ndim == 0 {
            vec![]
        } else {
            let mut strides = vec![1; ndim];
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            strides
        };
        
        Self {
            shape,
            strides,
            numel,
            ndim,
            is_contiguous: true,
        }
    }
    
    /// Check if the tensor is a scalar (0-dimensional)
    pub fn is_scalar(&self) -> bool {
        self.ndim == 0
    }
    
    /// Check if the tensor is a vector (1-dimensional)
    pub fn is_vector(&self) -> bool {
        self.ndim == 1
    }
    
    /// Check if the tensor is a matrix (2-dimensional)
    pub fn is_matrix(&self) -> bool {
        self.ndim == 2
    }
    
    /// Get the size of a specific dimension
    pub fn size(&self, dim: usize) -> Option<usize> {
        self.shape.get(dim).copied()
    }
    
    /// Get the stride of a specific dimension
    pub fn stride(&self, dim: usize) -> Option<usize> {
        self.strides.get(dim).copied()
    }
}