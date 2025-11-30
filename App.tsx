import React from 'react';
import OptimizationGuide from './components/OptimizationGuide';
import AiAssistant from './components/AiAssistant';

function App() {
  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans selection:bg-blue-500/30">
      
      {/* Navigation Bar */}
      <nav className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-tr from-blue-600 to-indigo-500 w-8 h-8 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
                </svg>
              </div>
              <span className="font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                CUDA<span className="text-blue-500">Opt</span>
              </span>
            </div>
            <div className="flex items-center gap-4">
               <span className="text-xs font-mono text-slate-500 border border-slate-800 px-2 py-1 rounded">TPC-H SF100</span>
               <a href="#" className="text-sm text-slate-400 hover:text-white transition-colors">Documentation</a>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <OptimizationGuide />
      </main>

      {/* Floating AI Assistant */}
      <AiAssistant />
    </div>
  );
}

export default App;