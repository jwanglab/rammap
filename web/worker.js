// Web Worker: runs WASM alignment in a background thread.
// Communicates with the main page via postMessage.

let mod = null;
let wasmExports = null;
let ready = false;

function memBytes() {
  try { return wasmExports?.memory?.buffer?.byteLength ?? 0; } catch { return 0; }
}

// Forward console.log and console.error from inside WASM to the main thread.
const originalLog = console.log;
console.log = function(...args) {
  const msg = args.map(a => String(a)).join(' ');
  postMessage({ type: 'log', text: msg });
  originalLog.apply(console, args);
};
const originalError = console.error;
console.error = function(...args) {
  const msg = args.map(a => String(a)).join(' ');
  postMessage({ type: 'log', text: '[!] ' + msg });
  originalError.apply(console, args);
};

// 16 MB streaming chunks: balances chunk overhead vs WASM linear-memory
// pressure. Each chunk is parsed and the underlying bytes are dropped
// before the next chunk arrives.
const CHUNK_BYTES = 16 * 1024 * 1024;

self.onmessage = async function(e) {
  const { type, data } = e.data;

  if (type === 'init') {
    try {
      mod = await import('../pkg/rammap.js');
      wasmExports = await mod.default();
      // `initThreadPool` only exists in the threaded build (wasm-bindgen-rayon).
      // Its absence means this is a single-threaded build, so the thread count
      // is pinned to 1 regardless of what the UI requested.
      const multithreaded = !!mod.initThreadPool;
      let threads = 1;
      if (multithreaded) {
        threads = data?.numThreads || self.navigator?.hardwareConcurrency || 4;
        await mod.initThreadPool(threads);
      }
      ready = true;
      postMessage({ type: 'ready', threads, multithreaded });
      postMessage({ type: 'log', text: `[mem] post-init wasm linear memory: ${formatSize(memBytes())}` });
    } catch (err) {
      postMessage({ type: 'error', text: `Failed to load WASM: ${err.message || err}` });
    }
    return;
  }

  if (type === 'align') {
    if (!ready || !mod) {
      postMessage({ type: 'error', text: 'WASM not ready' });
      return;
    }
    try {
      await runAlignment(data);
    } catch (err) {
      const msg = String(err?.message || err);
      const memMB = (memBytes() / 1024 / 1024).toFixed(0);
      // Heuristic: a wasm32 OOM bubbles up as "unreachable executed" (the trap
      // emitted by the abort intrinsic when the allocator can't grow linear
      // memory). If the trap fires with linear memory at or near the 4 GB cap,
      // that's almost certainly what happened.
      const isUnreachable = /unreachable executed/i.test(msg);
      const nearCap = memBytes() > 3.5 * 1024 * 1024 * 1024;
      if (isUnreachable && nearCap) {
        postMessage({ type: 'error',
          text: `Out of memory: the WASM heap hit its 4 GB cap (current ${memMB} MB) ` +
                `before the index could finish. Genome-scale references won't fit in ` +
                `browser-WASM — use the native rammap CLI instead.` });
      } else {
        postMessage({ type: 'error',
          text: `Alignment error: ${msg} (wasm memory ${memMB} MB)` });
      }
    }
    return;
  }
};

async function runAlignment({ refFile, queryFile, preset, outputSam, outputCigar }) {
  if (!(refFile instanceof Blob) || !(queryFile instanceof Blob)) {
    throw new Error('refFile and queryFile must be Blob/File objects');
  }
  const session = new mod.AlignSession(preset, outputSam, outputCigar);

  // ─── Stream the reference ──────────────────────────────────────────────
  postMessage({ type: 'progress', stage: 'ref', message: `Streaming reference (${formatSize(refFile.size)})...` });
  const refGz = await isGzipped(refFile);
  // Reserve packed-buffer capacity. For plain FASTA the file size is a good
  // upper bound on total bases (a bit pessimistic, ~5% wasted for headers and
  // newlines). For gzipped input we don't know the decompressed size; 4× the
  // compressed size covers the typical 3-4× gzip ratio for genome data.
  const reserveHint = refGz ? refFile.size * 4 : refFile.size;
  session.reserve_ref_bases(BigInt(reserveHint));
  let refStream = refFile.stream();
  if (refGz) {
    refStream = refStream.pipeThrough(new DecompressionStream('gzip'));
  }
  let refBytesSeen = 0;
  let chunkIdx = 0;
  let lastMemLog = 0;
  try {
    for await (const chunk of streamChunks(refStream, CHUNK_BYTES)) {
      session.append_ref(chunk);
      refBytesSeen += chunk.byteLength;
      chunkIdx++;
      postMessage({ type: 'progress', stage: 'ref', bytes: refBytesSeen });
      // Log WASM memory growth every ~4 chunks (~64 MB processed) so we can
      // tell, when a crash happens, how close we are to the 4 GB cap.
      if (chunkIdx - lastMemLog >= 4) {
        postMessage({ type: 'log', text: `[mem] processed ${formatSize(refBytesSeen)} ref → wasm linear memory: ${formatSize(memBytes())}` });
        lastMemLog = chunkIdx;
      }
    }
    postMessage({ type: 'log', text: `[mem] ref streaming done → wasm linear memory: ${formatSize(memBytes())}` });
    session.finalize_ref();
    postMessage({ type: 'log', text: `[mem] finalize_ref done → wasm linear memory: ${formatSize(memBytes())}` });
  } catch (e) {
    postMessage({ type: 'log', text: `[mem] CRASH at ${formatSize(refBytesSeen)} ref-bytes processed → wasm linear memory: ${formatSize(memBytes())}` });
    throw e;
  }

  // ─── Stream the queries, aligning on the fly ───────────────────────────
  postMessage({ type: 'progress', stage: 'query', message: `Streaming queries (${formatSize(queryFile.size)})...` });
  let qStream = queryFile.stream();
  if (await isGzipped(queryFile)) {
    qStream = qStream.pipeThrough(new DecompressionStream('gzip'));
  }
  let qBytesSeen = 0;
  for await (const chunk of streamChunks(qStream, CHUNK_BYTES)) {
    // append_query returns the PAF/SAM for the reads completed in this chunk.
    // Forward it immediately so the page can render results as they stream in
    // rather than waiting for the whole query file to finish.
    const out = session.append_query(chunk);
    if (out) postMessage({ type: 'partial', output: out });
    qBytesSeen += chunk.byteLength;
    postMessage({ type: 'progress', stage: 'query', bytes: qBytesSeen });
  }
  const tail = session.finalize();
  const [trailingOut, logTail] = splitOnce(tail, '---LOG---\n');
  if (trailingOut) postMessage({ type: 'partial', output: trailingOut });

  postMessage({
    type: 'result',
    log: logTail || '',
  });
}

// ─── helpers ──────────────────────────────────────────────────────────────

async function isGzipped(blob) {
  if (blob.size < 2) return false;
  const head = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
  return head[0] === 0x1f && head[1] === 0x8b;
}

// Re-chunk a ReadableStream<Uint8Array> into chunks of approximately
// `targetBytes` each. Works with arbitrary upstream chunk sizes (gzip-
// decompression typically yields 16 KB - 64 KB chunks).
async function* streamChunks(stream, targetBytes) {
  const reader = stream.getReader();
  let buf = null;
  let bufLen = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (!buf) {
      buf = new Uint8Array(targetBytes);
      bufLen = 0;
    }
    let src = value;
    while (src.byteLength > 0) {
      const room = targetBytes - bufLen;
      const take = Math.min(room, src.byteLength);
      buf.set(src.subarray(0, take), bufLen);
      bufLen += take;
      src = src.subarray(take);
      if (bufLen === targetBytes) {
        yield buf.subarray(0, bufLen);
        buf = new Uint8Array(targetBytes);
        bufLen = 0;
      }
    }
  }
  if (buf && bufLen > 0) {
    yield buf.subarray(0, bufLen);
  }
}

function splitOnce(s, sep) {
  const i = s.indexOf(sep);
  if (i < 0) return [s, ''];
  return [s.slice(0, i), s.slice(i + sep.length)];
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
