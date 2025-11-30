
import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
  ReferenceLine
} from 'recharts';
import { OptimizationStep, OptimizationLevel } from '../types';
import { HARDWARE_SPECS } from '../constants';

interface PerformanceChartProps {
  data: OptimizationStep[];
  activeId: string;
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ data, activeId }) => {
  const [metric, setMetric] = useState<'time' | 'speedup' | 'bandwidth'>('time');

  // Reset to 'time' when data changes significantly (e.g. switching workloads)
  useEffect(() => {
    // Optional: could reset metric here if needed, but keeping user preference is usually better
  }, [data]);

  const chartData = data.map(item => ({
    ...item,
    displayValue: metric === 'time' 
      ? item.executionTime 
      : metric === 'speedup' 
        ? item.performanceMetric 
        : Math.round(item.effectiveBandwidth)
  }));

  const formatValue = (value: number) => {
    if (metric === 'time') {
        if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
        return `${value} ms`;
    }
    if (metric === 'speedup') return `${value.toFixed(1)}x`;
    return `${value} GB/s`;
  };

  return (
    <div className="h-96 w-full bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-lg flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-slate-300 text-sm font-bold truncate pr-2">
          {metric === 'time' ? 'Execution Time' 
           : metric === 'speedup' ? 'Speedup vs Serial CPU'
           : 'Effective Bandwidth'}
        </h3>
        <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-700 shrink-0">
          <button
            onClick={() => setMetric('time')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
              metric === 'time' 
                ? 'bg-blue-600 text-white shadow' 
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Time
          </button>
          <button
            onClick={() => setMetric('speedup')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
              metric === 'speedup' 
                ? 'bg-blue-600 text-white shadow' 
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Speedup
          </button>
          <button
            onClick={() => setMetric('bandwidth')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
              metric === 'bandwidth' 
                ? 'bg-blue-600 text-white shadow' 
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Bandwidth
          </button>
        </div>
      </div>

      <div className="flex-1 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 5, right: 35, top: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={true} vertical={true} />
            <XAxis 
              type="number" 
              stroke="#94a3b8" 
              fontSize={12}
              domain={[0, 'auto']}
              hide={true} 
            />
            <YAxis 
              dataKey="id" 
              type="category" 
              width={90} 
              stroke="#94a3b8" 
              fontSize={11}
              tickFormatter={(value) => {
                  const map: Record<string, string> = {
                      'CPU': 'CPU Serial',
                      'CPU_MULTICORE': 'CPU OpenMP',
                      'NAIVE': 'GPU Naive',
                      'SHARED_MEM': 'Shared Mem',
                      'WARP_SHUFFLE': 'GPU Warp',
                      'THRUST': 'Thrust',
                      'COMPRESSION': 'FP16'
                  };
                  return map[value] || value;
              }}
            />
            <Tooltip 
              cursor={{fill: '#334155', opacity: 0.4}}
              contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }}
              formatter={(value: number) => [formatValue(value), metric === 'time' ? 'Time' : metric === 'speedup' ? 'Speedup' : 'Bandwidth']}
            />
            <Bar dataKey="displayValue" radius={[0, 4, 4, 0]} barSize={24}>
              {chartData.map((entry, index) => {
                let color = '#64748b'; // default slate
                if (entry.id === activeId) color = '#3b82f6'; // active blue
                else if (entry.id === OptimizationLevel.CPU) color = '#eab308'; // yellow
                else if (entry.id === OptimizationLevel.CPU_MULTICORE) color = '#f97316'; // orange
                else if (entry.id === OptimizationLevel.NAIVE) color = '#ef4444'; // red
                else if (entry.id === OptimizationLevel.COMPRESSION) color = '#10b981'; // emerald
                else if (metric === 'bandwidth' && entry.displayValue > 1000) color = '#8b5cf6'; // purple for high bandwidth
                
                return <Cell key={`cell-${index}`} fill={color} />;
              })}
              <LabelList 
                dataKey="displayValue" 
                position="right" 
                fill="#cbd5e1" 
                fontSize={11} 
                formatter={(val: number) => formatValue(val)}
              />
            </Bar>
            
            {/* Show Bandwidth Hardware Limits */}
            {metric === 'bandwidth' && (
              <>
                <ReferenceLine x={HARDWARE_SPECS.cpu.bandwidthVal} stroke="#eab308" strokeDasharray="3 3" />
                <ReferenceLine x={HARDWARE_SPECS.gpu.bandwidthVal} stroke="#3b82f6" strokeDasharray="3 3" />
              </>
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
      {metric === 'bandwidth' && (
        <div className="mt-2 text-xs flex justify-center gap-6">
             <div className="flex items-center gap-1 text-yellow-500">
                <span className="w-4 h-0.5 bg-yellow-500 border-t border-dashed"></span>
                CPU Limit ({HARDWARE_SPECS.cpu.bandwidthVal})
             </div>
             <div className="flex items-center gap-1 text-blue-500">
                <span className="w-4 h-0.5 bg-blue-500 border-t border-dashed"></span>
                GPU Limit ({HARDWARE_SPECS.gpu.bandwidthVal})
             </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceChart;
