
import { OptimizationLevel, OptimizationStep } from './types';

export const SYSTEM_INSTRUCTION = `You are a Senior CUDA Optimization Engineer. 
The user is asking about optimizing a TPC-H Lineitem table query (Sum Aggregation) on a GPU.
The dataset size is roughly 600 million tuples (TPC-H SF100, ~4.5GB).
Focus on concepts like:
1. CPU baseline comparison (memory bandwidth bound).
2. Memory Coalescing (Structure of Arrays vs Array of Structures).
3. Grid-Stride Loops to handle arbitrary input sizes.
4. Warp Shuffle reduction for high performance within blocks.
5. Avoiding Atomic contention.
6. Solving the Memory Wall via Compression (FP16/Int8).
Provide concise, technical, and accurate C++ CUDA code snippets when asked.`;

export const HARDWARE_SPECS = {
  cpu: {
    model: "Intel Xeon Gold 6248",
    clock: "2.50 GHz (Base)",
    cores: "20 Cores / 40 Threads",
    bandwidth: "130 GB/s (Peak)",
    bandwidthVal: 130
  },
  gpu: {
    model: "NVIDIA A100-SXM4",
    clock: "1410 MHz (Boost)",
    cores: "6912 CUDA Cores",
    memory: "40GB HBM2e",
    bandwidth: "1555 GB/s (Peak)",
    bandwidthVal: 1555
  }
};

// Data Size: 600M doubles = 600 * 10^6 * 8 bytes = 4.8 GB
const DATA_SIZE_GB = 4.8;

export const MEMORY_BOUND_STEPS: OptimizationStep[] = [
  {
    id: OptimizationLevel.CPU,
    title: "0. CPU Baseline (Serial)",
    description: "Latency Bound. Single-thread performance is limited by clock speed and instruction latency, not memory bandwidth. It cannot feed the CPU's memory controller fast enough.",
    performanceMetric: 1.0,
    executionTime: 1250.0,
    effectiveBandwidth: DATA_SIZE_GB / 1.250, // ~3.8 GB/s
    code: `// Host Code (C++)
#include <iostream>
#include <vector>
#include <chrono>

void sum_cpu(const std::vector<double>& h_input) {
    auto start = std::chrono::high_resolution_clock::now();
    
    double sum = 0.0;
    // Standard serial loop
    // Execution Time: ~1.25s for 600M items
    // Limited by instruction latency
    for (double val : h_input) {
        sum += val;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    printf("Time: %f ms\\n", ms.count());
}`
  },
  {
    id: OptimizationLevel.CPU_MULTICORE,
    title: "0.5 CPU Parallel (OpenMP)",
    description: "Bandwidth Bound (CPU). Using all 40 threads saturates the DDR4 memory channels. We hit ~96 GB/s, close to the theoretical limit of 130 GB/s. Adding more cores won't help.",
    performanceMetric: 25.0, 
    executionTime: 50.0,
    effectiveBandwidth: DATA_SIZE_GB / 0.050, // ~96 GB/s
    code: `// Host Code (C++) with OpenMP
#include <omp.h>
#include <vector>
#include <iostream>

void sum_cpu_omp(const std::vector<double>& h_input) {
    double total_sum = 0.0;
    
    // Saturation of Memory Bus
    // Execution Time: ~50ms
    #pragma omp parallel for reduction(+:total_sum)
    for (size_t i = 0; i < h_input.size(); i++) {
        total_sum += h_input[i];
    }
    
    printf("Total Sum: %f\\n", total_sum);
}`
  },
  {
    id: OptimizationLevel.NAIVE,
    title: "1. Naive Global Atomics",
    description: "Contention Bound. Terrible performance. 600M threads serialize on a single atomic lock. GPU utilization is effectively 0%.",
    performanceMetric: 0.03, 
    executionTime: 45000.0,
    effectiveBandwidth: DATA_SIZE_GB / 45.0, 
    code: `__global__ void sum_naive(const double* input, double* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Massive Contention
        // 600 Million threads fighting for one lock
        atomicAdd(output, input[idx]);
    }
}`
  },
  {
    id: OptimizationLevel.SHARED_MEM,
    title: "2. Shared Memory Reduction",
    description: "Compute/Latency Bound. Much faster, but we spend too much time loading from global memory into shared memory without full coalescence optimization. ~218 GB/s.",
    performanceMetric: 56.8, 
    executionTime: 22.0,
    effectiveBandwidth: DATA_SIZE_GB / 0.022, 
    code: `__global__ void sum_shared(const double* input, double* output, size_t n) {
    __shared__ double sdata[256];
    unsigned int tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    double sum = 0.0;
    while (i < n) {
        sum += input[i];
        i += gridDim.x * blockDim.x; 
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in Shared Mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(output, sdata[0]);
}`
  },
  {
    id: OptimizationLevel.WARP_SHUFFLE,
    title: "3. Warp Shuffle (State of the Art)",
    description: "Bandwidth Bound (GPU). We hit the Memory Wall. We are reading at ~1500 GB/s, matching the A100's physical limit. We cannot go faster with standard doubles.",
    performanceMetric: 390.6,
    executionTime: 3.2,
    effectiveBandwidth: DATA_SIZE_GB / 0.0032, // ~1500 GB/s
    code: `__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void sum_warp_shuffle(const double* input, double* output, size_t n) {
    double sum = 0.0;
    // Grid-Stride Loop ensures Memory Coalescing
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Warp Reduction (No Shared Mem needed)
    sum = warpReduceSum(sum);

    // Atomic add only once per warp (or block)
    if ((threadIdx.x % 32) == 0) {
        atomicAdd(output, sum);
    }
}`
  },
  {
    id: OptimizationLevel.COMPRESSION,
    title: "5. Compression (FP16 Solution)",
    description: "Breaking the Wall. Since we can't increase bandwidth, we decrease data size. Converting Double (64-bit) to Half (16-bit) reduces traffic by 4x. Throughput effectively quadruples to ~6000 GB/s (virtual).",
    performanceMetric: 1562.5,
    executionTime: 0.8,
    effectiveBandwidth: DATA_SIZE_GB / 0.0008, // ~6000 GB/s
    code: `#include <cuda_fp16.h>

__global__ void sum_compressed(const half* __restrict__ input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. Vectorized Load (Crucial for Bandwidth)
    // We cast the input pointer to float4* to load 128 bits (16 bytes) in a single instruction.
    // Since sizeof(half) = 2 bytes, this fetches 8 FP16 values at once.
    // This reduces the number of memory transactions by 8x compared to loading individually.
    float4 packed_data = reinterpret_cast<const float4*>(input)[idx];
    
    // 2. Reinterpret as Packed Half-Precision Vectors
    // We treat the loaded 16 bytes as an array of 4 'half2' vectors.
    // 'half2' is a SIMD type containing two 16-bit floats packed together.
    half2* halves = reinterpret_cast<half2*>(&packed_data);
    
    float sum = 0.0f;

    // 3. Unpack and Accumulate
    // We use #pragma unroll to force the compiler to inline these additions.
    #pragma unroll
    for(int k=0; k<4; k++) {
        // __half2float promotes the 16-bit float to a 32-bit float.
        // This is essential to maintain precision when summing millions of values.
        sum += __half2float(halves[k].x); // Lower 16 bits
        sum += __half2float(halves[k].y); // Upper 16 bits
    }
    
    // ... Standard Warp Reduction & Atomic Add follows ...
}`
  }
];

export const DB_FILTER_STEPS: OptimizationStep[] = [
    {
      id: OptimizationLevel.CPU,
      title: "0. CPU Baseline (Serial)",
      description: "Branch Misprediction Hell. The 'if (val > threshold)' check causes frequent pipeline flushes on the CPU when data is unsorted. Execution is noticeably slower (1.6s) than unconditional sum (1.25s).",
      performanceMetric: 1.0,
      executionTime: 1600.0, 
      effectiveBandwidth: DATA_SIZE_GB / 1.6, 
      code: `// Host Code (C++) - DB Filter
#include <vector>
#include <iostream>

void filter_sum_cpu(const std::vector<double>& h_input, double threshold) {
    double sum = 0.0;
    // PREDICATE: Branch Prediction struggles here if data is random
    // This causes CPU pipeline stalls (~15-20 cycle penalty per miss)
    for (double val : h_input) {
        if (val > threshold) {
            sum += val;
        }
    }
    printf("Filter Sum: %f\\n", sum);
}`
    },
    {
      id: OptimizationLevel.CPU_MULTICORE,
      title: "0.5 CPU Parallel (OpenMP)",
      description: "Bandwidth + Branch penalty. While we use all cores, the branch misprediction logic consumes cycles. Effective bandwidth drops to ~60 GB/s (vs 96 GB/s for simple sum).",
      performanceMetric: 20.0, 
      executionTime: 80.0, // 80ms (Slower than 50ms)
      effectiveBandwidth: DATA_SIZE_GB / 0.08, 
      code: `// Host Code (C++) - OpenMP
#include <omp.h>
#include <vector>
#include <iostream>

void filter_sum_omp(const std::vector<double>& h_input, double threshold) {
    double total_sum = 0.0;

    #pragma omp parallel for reduction(+:total_sum)
    for (size_t i = 0; i < h_input.size(); i++) {
        // CPU Branch Prediction misses cost ~15-20 cycles
        // Performance drops due to pipeline flushes
        if (h_input[i] > threshold) {
            total_sum += h_input[i];
        }
    }
    printf("Total Sum: %f\\n", total_sum);
}`
    },
    {
      id: OptimizationLevel.WARP_SHUFFLE,
      title: "1. GPU Optimized (Predicate)",
      description: "Resilient Speedup (~24x). GPU uses 'Predicated Execution' (calculating both paths and masking). It handles branches much better than CPU, avoiding pipeline flushes. We remain saturated at HBM limits.",
      performanceMetric: 484.8, 
      executionTime: 3.3, // 3.3ms (Barely slower than 3.2ms)
      effectiveBandwidth: DATA_SIZE_GB / 0.0033, 
      code: `__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void filter_sum_gpu(const double* input, double* output, size_t n, double threshold) {
    double sum = 0.0;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        double val = input[i];
        
        // GPU handles this efficiently via Predication
        // The compiler converts this to masked add instructions
        // No pipeline flush involved
        if (val > threshold) {
            sum += val;
        }
    }

    // Standard Warp Reduction
    sum = warpReduceSum(sum);
    
    if ((threadIdx.x % 32) == 0) {
        atomicAdd(output, sum);
    }
}`
    }
  ];
