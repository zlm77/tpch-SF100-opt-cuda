
import React, { useState, useMemo } from 'react';
import { MEMORY_BOUND_STEPS, DB_FILTER_STEPS, HARDWARE_SPECS } from '../constants';
import CodeBlock from './CodeBlock';
import PerformanceChart from './PerformanceChart';
import { OptimizationStep, WorkloadType } from '../types';

const OptimizationGuide: React.FC = () => {
  const [workload, setWorkload] = useState<WorkloadType>('MEMORY_BOUND');
  
  const currentSteps = useMemo(() => {
    return workload === 'MEMORY_BOUND' ? MEMORY_BOUND_STEPS : DB_FILTER_STEPS;
  }, [workload]);

  const [activeStep, setActiveStep] = useState<OptimizationStep>(currentSteps[0]);

  // Reset active step when workload changes
  React.useEffect(() => {
    setActiveStep(currentSteps[0]);
  }, [currentSteps]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-80px)]">
      
      {/* Sidebar Navigation */}
      <div className="lg:col-span-4 flex flex-col gap-4 overflow-y-auto pr-2 custom-scrollbar pb-6">
        
        {/* Workload Toggle */}
        <div className="bg-slate-800 rounded-lg p-1 border border-slate-700 flex text-sm font-semibold mb-2 shrink-0">
          <button
            onClick={() => setWorkload('MEMORY_BOUND')}
            className={`flex-1 py-2 rounded transition-all ${
              workload === 'MEMORY_BOUND' 
                ? 'bg-blue-600 text-white shadow-lg' 
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Simple Sum
          </button>
          <button
            onClick={() => setWorkload('DB_FILTER')}
            className={`flex-1 py-2 rounded transition-all ${
              workload === 'DB_FILTER' 
                ? 'bg-purple-600 text-white shadow-lg' 
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            DB Filter (Where &gt;)
          </button>
        </div>

        {/* Hardware Specs Card */}
        <div className="bg-slate-800 rounded-lg p-5 border border-slate-700 shadow-lg shrink-0">
          <h3 className="text-slate-200 font-bold text-sm mb-3 flex items-center gap-2 uppercase tracking-wider">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-slate-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
            </svg>
            System Configuration
          </h3>
          <div className="space-y-4">
            {/* CPU Config */}
            <div className="bg-slate-900/50 rounded p-3 border border-slate-700">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-bold text-yellow-500 bg-yellow-900/20 px-1.5 py-0.5 rounded">CPU</span>
                <span className="text-sm font-semibold text-slate-200">{HARDWARE_SPECS.cpu.model}</span>
              </div>
              <div className="grid grid-cols-2 gap-y-1 gap-x-2 text-xs text-slate-400">
                <span className="text-slate-500">Clock:</span>
                <span>{HARDWARE_SPECS.cpu.clock}</span>
                <span className="text-slate-500">Cores:</span>
                <span>{HARDWARE_SPECS.cpu.cores}</span>
              </div>
            </div>
            
            {/* GPU Config */}
            <div className="bg-slate-900/50 rounded p-3 border border-slate-700">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-bold text-green-500 bg-green-900/20 px-1.5 py-0.5 rounded">GPU</span>
                <span className="text-sm font-semibold text-slate-200">{HARDWARE_SPECS.gpu.model}</span>
              </div>
              <div className="grid grid-cols-2 gap-y-1 gap-x-2 text-xs text-slate-400">
                <span className="text-slate-500">Cores:</span>
                <span>{HARDWARE_SPECS.gpu.cores}</span>
                <span className="text-slate-500">Memory:</span>
                <span>{HARDWARE_SPECS.gpu.memory}</span>
                <span className="text-slate-500">Bandwidth:</span>
                <span>{HARDWARE_SPECS.gpu.bandwidth}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Dynamic Analysis Card */}
        {workload === 'MEMORY_BOUND' ? (
          <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-500/30 shrink-0">
            <h4 className="text-blue-300 font-bold text-xs uppercase mb-2">Analysis: Simple Sum</h4>
            <div className="text-xs text-blue-200 leading-relaxed space-y-2">
               <p>
                 <span className="font-bold text-white">Scenario:</span> Unconditional Sum.
               </p>
               <p>
                 <span className="font-bold text-white">Bottleneck:</span> Memory Bandwidth.
               </p>
               <div className="bg-slate-900/50 p-2 rounded border border-blue-500/20 mt-1">
                 <p className="text-slate-400">Insight:</p>
                 <p>Both CPU and GPU process data faster than RAM provides it.</p>
                 <p>Speedup (~12x) reflects the bandwidth ratio (1555 vs 130 GB/s).</p>
               </div>
            </div>
          </div>
        ) : (
          <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/30 shrink-0">
            <h4 className="text-purple-300 font-bold text-xs uppercase mb-2">Analysis: Filter Predicate</h4>
            <div className="text-xs text-purple-200 leading-relaxed space-y-2">
               <p>
                 <span className="font-bold text-white">Scenario:</span> <code className="bg-slate-900 px-1 rounded">WHERE val &gt; threshold</code>
               </p>
               <p>
                 <span className="font-bold text-white">Bottleneck:</span> Branch Prediction (CPU) vs Bandwidth (GPU).
               </p>
               <div className="bg-slate-900/50 p-2 rounded border border-purple-500/20 mt-1">
                 <p className="text-slate-400">Insight:</p>
                 <p><span className="text-yellow-400">CPU</span> slows down due to Branch Misprediction (pipeline flushes).</p>
                 <p><span className="text-green-400">GPU</span> uses Predication (masking) and stays fast.</p>
                 <p className="font-bold border-t border-slate-700 mt-1 pt-1">Gap Widens: ~25x Speedup</p>
               </div>
            </div>
          </div>
        )}

        {/* Steps List */}
        <div className="bg-slate-800 rounded-lg p-5 border border-slate-700 shrink-0">
          <h2 className="text-xl font-bold text-white mb-2">Experiment Steps</h2>
          <p className="text-slate-400 text-sm mb-4">
            Goal: Process 600,000,000 tuples.
          </p>
          <div className="space-y-2">
            {currentSteps.map((step) => (
              <button
                key={step.id}
                onClick={() => setActiveStep(step)}
                className={`w-full text-left p-3 rounded-md transition-all border ${
                  activeStep.id === step.id
                    ? 'bg-blue-600/20 border-blue-500 text-blue-200 shadow-md shadow-blue-900/20'
                    : 'bg-slate-700/50 border-transparent text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-sm">{step.title}</span>
                  {activeStep.id === step.id && (
                    <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Explanation & Chart */}
        <div className="bg-slate-800 rounded-lg p-5 border border-slate-700 flex flex-col min-h-[500px]">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            Experiment Results
          </h3>
          <p className="text-slate-300 leading-relaxed text-sm mb-4">
            {activeStep.description}
          </p>
          
          <div className="mt-2 pt-4 border-t border-slate-700 flex-1">
            <PerformanceChart data={currentSteps} activeId={activeStep.id} />
          </div>
        </div>
      </div>

      {/* Main Code Area */}
      <div className="lg:col-span-8 h-full flex flex-col">
        <CodeBlock code={activeStep.code} title={`${activeStep.title} - Code Snippet`} />
      </div>

    </div>
  );
};

export default OptimizationGuide;
