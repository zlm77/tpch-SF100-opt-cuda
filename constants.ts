
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
6. CPU Branch Misprediction vs GPU Predicated Execution.
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

void sum_cpu(const std::vector<double>& h_input) {
    double sum = 0.0;
    // Standard serial loop
    // ~1.25s for 600M items
    for (double val : h_input) {
        sum += val;
    }
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

void sum_cpu_omp(const std::vector<double>& h_input) {
    double total_sum = 0.0;
    
    // Saturation of Memory Bus
    #pragma omp parallel for reduction(+:total_sum)
    for (size_t i = 0; i < h_input.size(); i++) {
        total_sum += h_input[i];
    }
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
    size_t idx = ...;
    if (idx < n) {
        // Massive Contention
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
    // ... Load & Reduce in Shared Mem ...
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
    code: `__global__ void sum_warp_shuffle(...) {
    // Highly optimized Grid-Stride Loop
    // Saturated HBM2e bandwidth
    for (int i = ...; i < n; i += stride) {
        sum += input[i];
    }
    sum = warpReduceSum(sum);
    // ...
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
    // Load 128 bits (8 x FP16 values) at once
    // Vectorized loads maximize memory efficiency
    float4 packed_data = reinterpret_cast<const float4*>(input)[idx];
    
    // Unpack and accumulate in registers (fast)
    half2* halves = reinterpret_cast<half2*>(&packed_data);
    
    float sum = 0.0f;
    #pragma unroll
    for(int k=0; k<4; k++) {
        sum += __half2float(halves[k].x) + __half2float(halves[k].y);
    }
    
    // Standard reduction follows...
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
void filter_sum_cpu(const std::vector<double>& h_input, double threshold) {
    double sum = 0.0;
    // PREDICATE: Branch Prediction struggles here if data is random
    // This causes CPU pipeline stalls (~15-20 cycle penalty per miss)
    for (double val : h_input) {
        if (val > threshold) {
            sum += val;
        }
    }
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
#pragma omp parallel for reduction(+:total_sum)
for (size_t i = 0; i < h_input.size(); i++) {
    // CPU Branch Prediction misses cost ~15-20 cycles
    if (h_input[i] > threshold) {
        total_sum += h_input[i];
    }
}`
    },
    {
      id: OptimizationLevel.WARP_SHUFFLE,
      title: "1. GPU Optimized (Predicate)",
      description: "Resilient Speedup (~24x). GPU uses 'Predicated Execution' (calculating both paths and masking). It handles branches much better than CPU, avoiding pipeline flushes. We remain saturated at HBM limits.",
      performanceMetric: 484.8, 
      executionTime: 3.3, // 3.3ms (Barely slower than 3.2ms)
      effectiveBandwidth: DATA_SIZE_GB / 0.0033, 
      code: `__global__ void filter_sum_gpu(..., double threshold) {
    // ...
    double val = input[i];
    
    // GPU handles this efficiently via Predication
    // The compiler converts this to a masked add instructions
    // No pipeline flush involved
    if (val > threshold) {
        sum += val;
    }
    
    // ... Warp Reduction ...
}`
    }
  ];
