import React from 'react';

interface CodeBlockProps {
  code: string;
  title: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, title }) => {
  return (
    <div className="rounded-lg overflow-hidden border border-slate-700 bg-[#0d1117] flex flex-col h-full shadow-lg">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
        <span className="text-xs font-mono text-blue-400 font-bold">{title}</span>
        <div className="flex space-x-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/50"></div>
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500/50"></div>
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/20 border border-green-500/50"></div>
        </div>
      </div>
      <div className="p-4 overflow-auto flex-1 text-sm font-mono leading-relaxed">
        <pre className="text-slate-300">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
};

export default CodeBlock;
