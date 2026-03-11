/* ── Charts ── */

function renderCharts() {
  if (typeof Chart === 'undefined') return;

  const runs = window.RUNS_DATA || [];
  const done = runs.filter(r => r.status === 'completed' && r.performance);
  if (!done.length) return;

  const labels = done.map(r => `Run ${r.run}`);

  Chart.defaults.color = '#8b949e';
  Chart.defaults.borderColor = '#21262d';

  const defaults = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, beginAtZero: true },
    },
  };

  // Conversion time (stacked)
  new Chart(document.getElementById('timeChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Phase 1', data: done.map(r => Math.round((r.performance.phase1_time_s || 0) / 60)), backgroundColor: '#388bfd' },
        { label: 'Phase 2', data: done.map(r => Math.round((r.performance.phase2_time_s || 0) / 60)), backgroundColor: '#bc8cff' },
      ],
    },
    options: {
      ...defaults,
      plugins: { legend: { display: true, labels: { color: '#8b949e', boxWidth: 12 } } },
      scales: {
        ...defaults.scales,
        x: { ...defaults.scales.x, stacked: true },
        y: { ...defaults.scales.y, stacked: true, title: { display: true, text: 'Minutes', color: '#8b949e' } },
      },
    },
  });

  // Output size
  new Chart(document.getElementById('sizeChart'), {
    type: 'bar',
    data: { labels, datasets: [{ data: done.map(r => r.performance.output_size_gb || 0), backgroundColor: '#3fb950', borderRadius: 3 }] },
    options: { ...defaults, scales: { ...defaults.scales, y: { ...defaults.scales.y, title: { display: true, text: 'GB', color: '#8b949e' } } } },
  });

  // Compression ratio
  new Chart(document.getElementById('ratioChart'), {
    type: 'bar',
    data: { labels, datasets: [{ data: done.map(r => r.performance.compression_ratio || 0), backgroundColor: '#d29922', borderRadius: 3 }] },
    options: { ...defaults, scales: { ...defaults.scales, y: { ...defaults.scales.y, title: { display: true, text: 'Ratio', color: '#8b949e' } } } },
  });

  // Phase 2 rate
  new Chart(document.getElementById('rateChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: done.map(r => r.performance.phase2_rate_batches_per_min || r.performance.phase2_rate_chunks_per_min || 0),
        backgroundColor: '#f78166', borderRadius: 3,
      }],
    },
    options: { ...defaults, scales: { ...defaults.scales, y: { ...defaults.scales.y, title: { display: true, text: 'Batches/min', color: '#8b949e' } } } },
  });
}

/* ── Card toggle ── */

function toggleBody(header) {
  const body = header.nextElementSibling;
  const chevron = header.querySelector('.chevron');
  body.classList.toggle('open');
  chevron.classList.toggle('open');
}

/* ── Live log polling ── */

const logPollers = {};

async function toggleLog(btn, runId) {
  const pre = document.getElementById(`log-${runId}`);

  if (pre.classList.contains('open')) {
    pre.classList.remove('open');
    btn.innerHTML = 'Log';
    if (logPollers[runId]) { clearInterval(logPollers[runId]); delete logPollers[runId]; }
    return;
  }

  async function fetchLog() {
    try {
      const resp = await fetch(`/api/logs/${runId}`);
      if (!resp.ok) return;
      const text = await resp.text();
      if (text !== pre.textContent) {
        const wasAtBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
        pre.textContent = text;
        if (wasAtBottom) pre.scrollTop = pre.scrollHeight;
      }
    } catch {}
  }

  btn.innerHTML = '…';
  await fetchLog();
  pre.classList.add('open');

  const runs = window.RUNS_DATA || [];
  const run = runs.find(r => r.run === runId);
  const isLive = run && (run.status === 'running' || run.status === 'phase2');

  if (isLive) {
    btn.innerHTML = 'Hide <span class="log-live"><span class="dot"></span>live</span>';
    logPollers[runId] = setInterval(fetchLog, 3000);
  } else {
    btn.innerHTML = 'Hide';
  }
}

/* ── Tabs ── */

document.getElementById('tabs').addEventListener('click', e => {
  const tab = e.target.closest('.tab');
  if (!tab) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  tab.classList.add('active');
  document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
});

/* ── Init ── */

renderCharts();
