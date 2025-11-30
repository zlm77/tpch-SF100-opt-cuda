
export enum OptimizationLevel {
  CPU = 'CPU',
  CPU_MULTICORE = 'CPU_MULTICORE',
  NAIVE = 'NAIVE',
  SHARED_MEM = 'SHARED_MEM',
  WARP_SHUFFLE = 'WARP_SHUFFLE',
  THRUST = 'THRUST',
  COMPRESSION = 'COMPRESSION'
}

export type WorkloadType = 'MEMORY_BOUND' | 'DB_FILTER';

export interface OptimizationStep {
  id: OptimizationLevel;
  title: string;
  description: string;
  code: string;
  performanceMetric: number; // Speedup vs CPU
  executionTime: number; // Actual runtime in milliseconds
  effectiveBandwidth: number; // GB/s processed
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}
