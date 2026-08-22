const runList = document.getElementById('run-list');
const runCount = document.getElementById('run-count');
const runTitle = document.getElementById('run-title');
const runMeta = document.getElementById('run-meta');
const metricGrid = document.getElementById('metric-grid');
const rawMetrics = document.getElementById('raw-metrics');
const message = document.getElementById('message');
const iframe = document.getElementById('perfetto');
let runs = [];
let selectedIndex = -1;
let profileTotal = 0;
let nextOffset = 0;
let hasMoreRuns = false;
let perfettoReady = false;
let pendingTrace = null;
let pingTimer = null;
const pageSize = 50;
const formatNumber = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle('error', isError);
}

function formatValue(value, unit = '') {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${formatNumber.format(Number(value))}${unit ? ` ${unit}` : ''}`;
}

function currentRun() { return runs[selectedIndex]; }

function renderRunList() {
  runList.replaceChildren();
  if (!runs.length) {
    runList.innerHTML = '<p class="empty">No profile runs found.</p>';
    runCount.textContent = '0 runs';
    return;
  }
  runCount.textContent = profileTotal > runs.length
    ? `${runs.length} of ${profileTotal}`
    : `${profileTotal} ${profileTotal === 1 ? 'run' : 'runs'}`;
  runs.forEach((run, index) => {
    const button = document.createElement('button');
    button.className = 'run-button';
    button.type = 'button';
    button.setAttribute('aria-pressed', String(index === selectedIndex));
    const model = document.createElement('span');
    model.className = 'run-model';
    model.textContent = run.model;
    const name = document.createElement('span');
    name.className = 'run-name';
    name.textContent = run.name;
    button.append(model, name);
    button.addEventListener('click', () => selectRun(index));
    runList.appendChild(button);
  });
  if (hasMoreRuns) {
    const button = document.createElement('button');
    button.className = 'load-more';
    button.type = 'button';
    button.textContent = 'Load more';
    button.setAttribute('aria-controls', 'run-list');
    button.addEventListener('click', loadMoreRuns);
    runList.appendChild(button);
  }
}

async function loadMoreRuns(event) {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = 'Loading…';
  try {
    const response = await fetch(`/api/profiles?limit=${pageSize}&offset=${nextOffset}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    runs = runs.concat(payload.runs || []);
    profileTotal = payload.total || runs.length;
    nextOffset = runs.length;
    hasMoreRuns = Boolean(payload.has_more);
    renderRunList();
    setMessage(`${runs.length} of ${profileTotal} profile runs loaded.`);
  } catch (error) {
    button.disabled = false;
    button.textContent = 'Load more';
    setMessage(`Could not load more profiles: ${error.message}`, true);
  }
}

function renderMetrics(payload) {
  const metrics = payload.metrics || payload;
  const values = [
    [metrics.time_to_first_token, 's'],
    [metrics.prefill_tokens_per_second, 'tok/s'],
    [metrics.decode_tokens_per_second, 'tok/s'],
    [metrics.average_inter_token_latency, 's'],
    [metrics.total_seconds, 's'],
    [metrics.generated_tokens, ''],
    [metrics.prompt_tokens, ''],
    [metrics.decode_seconds, 's']
  ];
  metricGrid.querySelectorAll('.metric-value').forEach((element, index) => {
    element.textContent = formatValue(values[index][0], values[index][1]);
  });
  rawMetrics.textContent = JSON.stringify(payload, null, 2);
}

async function loadMetrics() {
  const run = currentRun();
  if (!run || !run.files.metrics) {
    rawMetrics.textContent = 'Metrics unavailable for this run.';
    setMessage('Metrics are unavailable for this run.');
    return;
  }
  setMessage('Loading metrics…');
  try {
    const response = await fetch(run.urls.metrics);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderMetrics(await response.json());
    setMessage('Metrics updated.');
  } catch (error) {
    setMessage(`Could not load metrics: ${error.message}`, true);
  }
}

async function selectRun(index) {
  selectedIndex = index;
  const run = currentRun();
  renderRunList();
  if (!run) return;
  runTitle.textContent = run.name;
  runMeta.textContent = `${run.model}  ·  ${run.id}`;
  document.getElementById('engine-button').disabled = !run.files.engine_trace;
  document.getElementById('torch-button').disabled = !run.files.torch_trace;
  await loadMetrics();
  if (run.files.engine_trace) await openTrace('engine');
}

function sendPing() {
  if (iframe.contentWindow) iframe.contentWindow.postMessage('PING', '*');
}

function startPerfettoHandshake() {
  if (perfettoReady || pingTimer !== null) return;
  sendPing();
  pingTimer = setInterval(sendPing, 100);
}

function stopPerfettoHandshake() {
  if (pingTimer !== null) {
    clearInterval(pingTimer);
    pingTimer = null;
  }
}

function sendPendingTrace() {
  if (!perfettoReady || !pendingTrace) return;
  const trace = pendingTrace;
  pendingTrace = null;
  iframe.contentWindow.postMessage({ perfetto: {
    buffer: trace.buffer,
    title: trace.title,
    fileName: trace.fileName,
    keepApiOpen: true
  } }, '*', [trace.buffer]);
  setMessage(`${trace.label} opened in Perfetto.`);
}

window.addEventListener('message', (event) => {
  if (event.source === iframe.contentWindow && event.data === 'PONG') {
    perfettoReady = true;
    stopPerfettoHandshake();
    sendPendingTrace();
  }
});

iframe.addEventListener('load', () => {
  perfettoReady = false;
  startPerfettoHandshake();
});
startPerfettoHandshake();

async function openTrace(kind) {
  const run = currentRun();
  if (!run || !run.files[kind + '_trace']) {
    setMessage('Trace unavailable for this run.', true);
    return;
  }
  setMessage(`Loading ${kind === 'engine' ? 'engine' : 'PyTorch'} trace…`);
  const response = await fetch(run.urls[kind + '_trace']);
  if (!response.ok) {
    setMessage(`Could not load trace: HTTP ${response.status}`, true);
    return;
  }
  const buffer = await response.arrayBuffer();
  pendingTrace = {
    buffer,
    label: kind === 'engine' ? 'Engine trace' : 'PyTorch trace',
    title: `${run.model} / ${run.name} / ${kind}`,
    fileName: `${run.name}.${kind}.trace.json`
  };
  if (perfettoReady) sendPendingTrace();
  else startPerfettoHandshake();
}

document.getElementById('metrics-button').addEventListener('click', loadMetrics);
document.getElementById('engine-button').addEventListener('click', () => openTrace('engine'));
document.getElementById('torch-button').addEventListener('click', () => openTrace('torch'));

fetch('/api/profiles').then((response) => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then((payload) => {
  runs = payload.runs || [];
  profileTotal = payload.total || runs.length;
  nextOffset = runs.length;
  hasMoreRuns = Boolean(payload.has_more);
  renderRunList();
  if (runs.length) selectRun(0);
  else {
    runTitle.textContent = 'No profile selected';
    setMessage('Run a profiled generation to see metrics.');
  }
}).catch((error) => {
  runList.innerHTML = '<p class="empty">Could not load profile runs.</p>';
  runCount.textContent = 'Unavailable';
  setMessage(`Could not load profiles: ${error.message}`, true);
});
