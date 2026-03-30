// Web Worker: runs WASM alignment in a background thread.
// Communicates with the main page via postMessage.

let wasm = null;

// Override console.log to capture WASM log output and forward to main thread
const originalLog = console.log;
console.log = function(...args) {
  const msg = args.map(a => String(a)).join(' ');
  postMessage({ type: 'log', text: msg });
  originalLog.apply(console, args);
};

self.onmessage = async function(e) {
  const { type, data } = e.data;

  if (type === 'init') {
    try {
      // Import and initialize the WASM module
      const { default: init, align_wasm_full } = await import('../pkg/rammap.js');
      await init();
      wasm = { align_wasm_full };
      postMessage({ type: 'ready' });
    } catch (err) {
      postMessage({ type: 'error', text: `Failed to load WASM: ${err.message || err}` });
    }
    return;
  }

  if (type === 'align') {
    const { refText, queryText, preset, outputSam, outputCigar } = data;
    try {
      const result = wasm.align_wasm_full(refText, queryText, preset, outputSam, outputCigar);
      const parts = result.split('\n---LOG---\n');
      postMessage({
        type: 'result',
        output: parts[0] || '',
        log: parts[1] || '',
      });
    } catch (err) {
      postMessage({ type: 'error', text: `Alignment error: ${err.message || err}` });
    }
    return;
  }
};
