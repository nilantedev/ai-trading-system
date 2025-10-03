(async function(){
  const $ = (s)=>document.querySelector(s);
  async function fetchJSON(url, opts){
    const headers = (opts && opts.headers) ? {...opts.headers} : {};
    if((opts && (opts.method||'GET') !== 'GET')){
      const csrf = document.cookie.split(';').map(c=>c.trim()).find(c=>c.startsWith('csrf_token='));
      if(csrf){ headers['X-CSRF-Token'] = decodeURIComponent(csrf.split('=')[1]); }
    }
    const r = await fetch(url, {credentials:'include', ...opts, headers});
    if(!r.ok) throw new Error(url+': '+r.status);
    const ct = r.headers.get('content-type')||'';
    return ct.includes('application/json') ? r.json() : r.text();
  }
  function render(id, data){ const el=document.getElementById(id); if(!el) return; el.textContent=typeof data==='string'?data:JSON.stringify(data,null,2); }
  function appendLog(el, line){ if(!el) return; el.textContent += (line+"\n"); el.scrollTop = el.scrollHeight; }

  async function loadStaticPanels(){
    try { render('system-summary-body', await fetchJSON('/admin/api/system/summary')); } catch(e){ render('system-summary-body',{error:e.message}); }
    try { render('trading-performance-body', await fetchJSON('/admin/api/trading/performance')); } catch(e){ render('trading-performance-body',{error:e.message}); }
    try { render('model-status-body', await fetchJSON('/admin/api/models/status')); } catch(e){ render('model-status-body',{error:e.message}); }
  }
  async function loadMetrics(){
    try { render('metrics-body', await fetchJSON('/admin/api/metrics/summary')); } catch(e){ /* ignore */ }
  }
  async function loadDriftSummary(){
    try { const d = await fetchJSON('/admin/api/drift/summary'); render('drift-summary-body', d); const sev = d.overall_severity; const el=document.getElementById('drift-summary-body'); if(el && typeof d==='object'){ el.innerHTML = `<div>Overall Severity: <span class="badge ${sev==='high'?'danger':sev==='medium'?'warning':'success'}">${sev}</span></div>` + '<pre class="log" style="max-height:200px;overflow:auto">'+JSON.stringify(d.models,null,2)+'</pre>'; } } catch(e){ render('drift-summary-body',{error:e.message}); }
  }
  async function loadBackfillJobs(){
    try { const d = await fetchJSON('/admin/api/backfill/jobs?limit=10'); const el=document.getElementById('backfill-jobs-body'); if(el){ if(d.jobs){ el.innerHTML = '<table class="tight"><thead><tr><th>ID</th><th>Status</th><th>Dataset</th><th>Symbols</th><th>Updated</th></tr></thead><tbody>'+d.jobs.map(j=>`<tr><td>${(j.job_id||'').slice(0,8)}</td><td>${j.status||j.publish_status||'?'}</td><td>${j.dataset||''}</td><td>${j.symbol_count||''}</td><td>${j.last_update||j.created_at||''}</td></tr>`).join('')+'</tbody></table>'; } else { el.textContent='No jobs'; } } } catch(e){ render('backfill-jobs-body',{error:e.message}); }
  }
  async function loadBackfillCompleteness(){
    try { const r = await fetch('/health/full',{credentials:'include'}); if(r.ok){ const d= await r.json(); const bc = d.components?.backfill_complete; const badge=document.getElementById('coverage-backfill-badge'); if(badge){ if(bc){ badge.textContent='backfill complete'; badge.className='badge success'; } else { badge.textContent='backfill partial'; badge.className='badge warning'; } } } } catch(_){ }
  }
  async function loadLatency(){ try { render('latency-body', await fetchJSON('/admin/api/latency/metrics')); } catch(e){ render('latency-body',{error:e.message}); } }
  async function loadPnlTs(){
    try {
      const data = await fetchJSON('/admin/api/pnl/timeseries');
      render('pnl-timeseries-body', data.summary ? {summary:data.summary} : data);
      if(data.points){ ChartUtils.renderLineChart('pnlChart', data.points.map(p=>({ts:p.ts, pnl:p.pnl})), { series:[{key:'pnl', color:'#d9480f', name:'PnL'}] }); }
    } catch(e){ render('pnl-timeseries-body',{error:e.message}); }
  }

  // Data Verification & Coverage Panel
  let lastVerification = null;
  function formatPct(v){ if(v==null) return '—'; return (v*100).toFixed(2)+'%'; }
  function classifyRow(r){
    if(!r || r.meets_target === false) return 'badge danger';
    if(r.recent_gap) return 'badge warning';
    return 'badge success';
  }
  function renderVerification(summary){
    const container = document.getElementById('verification-summary');
    if(!container) return;
    if(!summary){ container.textContent='No data'; return; }
    const targets = summary.targets||{};
    const rows = Object.keys(targets).map(name=>{
      const t = targets[name];
      const ratio = t.approx_trading_day_ratio;
      const cls = classifyRow(t);
      return `<tr><td>${name}</td><td>${t.row_count||'—'}</td><td>${t.distinct_trading_days||'—'}</td><td>${t.span_days||'—'}</td><td>${formatPct(ratio)}</td><td><span class="${cls}">${t.meets_target? 'OK':'Target'}</span></td><td>${t.recent_gap?'<span class="badge warning">gap</span>':'—'}</td></tr>`;
    }).join('');
    container.innerHTML = `
      <table class="tight">
        <thead><tr><th>Dataset</th><th>Rows</th><th>Days</th><th>Span(d)</th><th>Coverage</th><th>Status</th><th>Recent Gap</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
    const statusEl = document.getElementById('verification-status');
    if(statusEl){
      if(summary.meets_all_targets){ statusEl.textContent='all targets met'; statusEl.className='badge success'; }
      else { statusEl.textContent='attention'; statusEl.className='badge warning'; }
    }
    // Populate Data Quality aggregate
    const dq = document.getElementById('data-quality-body');
    if(dq){
      const problem = Object.values(targets).filter(t=>!t.meets_target || t.recent_gap);
      if(problem.length===0){ dq.innerHTML = '<span class="badge success">No issues detected</span>'; }
      else {
        dq.innerHTML = '<ul>'+problem.map(p=>`<li>${p.recent_gap?'<span class=badge warning>gap</span> ':''}${p.meets_target?'' : '<span class=badge danger>target</span> '} (${p.row_count||'rows?'})</li>`).join('')+'</ul>';
      }
    }
  }
  async function loadVerification(refresh=false){
    try {
      const url = '/admin/api/data/verification'+(refresh?'?refresh=true':'');
      const data = await fetchJSON(url);
      lastVerification = data;
      renderVerification(data.summary);
      const rawEl = document.getElementById('verification-raw');
      if(rawEl){ rawEl.textContent = JSON.stringify(data.raw, null, 2); }
    } catch(e){ render('verification-summary', {error:e.message}); }
  }

  // Heartbeat & Live Streaming panel
  function startHeartbeat(){
    const connEl = document.getElementById('heartbeat-connection');
    const lastEl = document.getElementById('heartbeat-last');
    const latEl = document.getElementById('heartbeat-latency');
    const metaEl = document.getElementById('heartbeat-meta');
    let lastTs = null; let es; let lastRecv = 0;
    function connect(){
      if(connEl){ connEl.textContent='connecting'; connEl.className='badge warning'; }
      es = new EventSource('/admin/api/heartbeat/stream',{withCredentials:true});
      es.addEventListener('heartbeat', ev=>{
        const now = performance.now();
        const data = JSON.parse(ev.data);
        const serverTs = new Date(data.ts).getTime();
        if(lastTs){
          const latency = now - lastRecv; // time since previous event arrival
          if(latEl) latEl.textContent = latency.toFixed(0);
        }
        lastRecv = now;
        lastTs = serverTs;
        if(lastEl) lastEl.textContent = data.ts;
        if(metaEl) metaEl.textContent = 'latency_metric_present='+data.latency_metric_present;
        if(connEl){ connEl.textContent='live'; connEl.className='badge success'; }
      });
      es.onerror = ()=>{
        if(connEl){ connEl.textContent='reconnecting'; connEl.className='badge warning'; }
        setTimeout(()=>{ try { es.close(); } catch(_){} connect(); }, 2000);
      };
    }
    connect();
  }

  // Ingestion Freshness (derived from verification last_timestamp timestamps)
  function computeFreshnessSeries(){
    if(!lastVerification) return null;
    const summary = lastVerification.summary;
    if(!summary) return null;
    const targets = summary.targets||{};
    const now = Date.now();
    const points = Object.entries(targets).map(([k,v])=>{
      if(!v.last_timestamp) return null;
      const ageSec = (now - new Date(v.last_timestamp).getTime())/1000;
      return { dataset: k, ageSec };
    }).filter(Boolean);
    return points;
  }
  function renderFreshness(){
    const pts = computeFreshnessSeries();
    const body = document.getElementById('freshness-body');
    if(!body){ return; }
    if(!pts || pts.length===0){ body.textContent='No freshness data'; return; }
    const rows = pts.map(p=>`<tr><td>${p.dataset}</td><td>${p.ageSec.toFixed(1)}</td><td>${p.ageSec<120?'<span class="badge success">fresh</span>': p.ageSec<600?'<span class="badge warning">warm</span>':'<span class="badge danger">stale</span>'}</td></tr>`).join('');
    body.innerHTML = `<table class="tight"><thead><tr><th>Dataset</th><th>Age(s)</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>`;
    // simple bar visualization: treat ageSec as value
    const chartData = pts.map(p=>({ts: Date.now(), [p.dataset]: p.ageSec}));
    // Flatten for ChartUtils expecting series structure
    const series = pts.map((p,i)=>({key:p.dataset, color: ChartUtils.palette ? ChartUtils.palette[i % ChartUtils.palette.length] : '#'+((Math.random()*0xffffff)|0).toString(16)}));
    try { ChartUtils.renderLineChart('freshnessChart', chartData, {series}); } catch(_){ }
  }
  function updateFreshnessLoop(){ renderFreshness(); setTimeout(updateFreshnessLoop, 15000); }

  function bindVerificationActions(){
    const refreshBtn = document.getElementById('btn-verification-refresh');
    const forceBtn = document.getElementById('btn-force-backfill');
    const bootstrapBtn = document.getElementById('btn-bootstrap-tables');
    if(refreshBtn){ refreshBtn.onclick = ()=>{ loadVerification(true).then(()=>renderFreshness()); }; }
    if(forceBtn){
      forceBtn.onclick = async ()=>{
        forceBtn.disabled = true; forceBtn.textContent='Forcing Backfill...';
        try {
          // Run admin task runner programmatically with backfill force argument
            await fetchJSON('/admin/api/tasks/run',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name:'backfill', args:['force=true']})});
        } catch(e){ alert('Backfill start failed: '+e.message); }
        finally { forceBtn.disabled=false; forceBtn.textContent='Force Backfill'; }
      };
    }
    if(bootstrapBtn){
      bootstrapBtn.onclick = async ()=>{
        if(bootstrapBtn.dataset.done === '1') return;
        bootstrapBtn.disabled = true; const orig=bootstrapBtn.textContent; bootstrapBtn.textContent='Bootstrapping...';
        try {
          await fetchJSON('/admin/api/tasks/run',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name:'bootstrap-questdb'})});
          bootstrapBtn.textContent='Bootstrapping (started)';
          bootstrapBtn.dataset.done='1';
          // Trigger verification refresh after short delay to pick up new tables
          setTimeout(()=>{ loadVerification(true).then(()=>renderFreshness()); }, 5000);
        } catch(e){
          alert('Bootstrap failed to start: '+e.message); bootstrapBtn.disabled=false; bootstrapBtn.textContent=orig;
        }
      };
    }
  }

  async function bindActions(){
    const btn = $('#btn-promotion-check');
    const form = $('#rollback-form');
    const resEl = $('#admin-action-result');
    if(btn){
      btn.addEventListener('click', async ()=>{
        btn.disabled = true; resEl.textContent = 'Running promotion check...';
        try {
          const result = await fetchJSON('/admin/api/models/promotion-check', {method:'POST'});
          resEl.textContent = JSON.stringify(result, null, 2);
        } catch(e){ resEl.textContent = 'Error: '+e.message; }
        finally { btn.disabled = false; }
      });
    }
    if(form){
      form.addEventListener('submit', async (ev)=>{
        ev.preventDefault();
        const modelId = document.getElementById('rollback-model-id').value.trim();
        if(!modelId){ resEl.textContent = 'Enter a model_id.'; return; }
        resEl.textContent = 'Submitting rollback...';
        try {
          const result = await fetchJSON(`/admin/api/models/${encodeURIComponent(modelId)}/rollback`, {method:'POST'});
          resEl.textContent = JSON.stringify(result, null, 2);
        } catch(e){ resEl.textContent = 'Error: '+e.message; }
      });
    }
  }

  // Live application logs via SSE
  function startLogsStream(){
    const logsEl = document.getElementById('logs-stream');
    const stopBtn = document.getElementById('logs-stream-stop');
    const clearBtn = document.getElementById('logs-stream-clear');
    const connEl = document.getElementById('logs-conn');
    let es;
    function connect(){
      if(connEl){ connEl.textContent = 'connecting…'; connEl.className = 'badge warning'; }
      es = new EventSource('/admin/api/logs/stream', { withCredentials: true });
      es.addEventListener('log', (ev)=>{ appendLog(logsEl, ev.data); });
      es.onopen = ()=>{ if(connEl){ connEl.textContent = 'connected'; connEl.className = 'badge success'; } };
      es.onerror = () => {
        // Auto-reconnect with backoff
        if(es) es.close();
        if(connEl){ connEl.textContent = 'reconnecting…'; connEl.className = 'badge warning'; }
        setTimeout(connect, 2000);
      };
    }
    connect();
    if(stopBtn){ stopBtn.onclick = ()=>{ if(es) es.close(); es = null; } }
    if(clearBtn){ clearBtn.onclick = ()=>{ if(logsEl) logsEl.textContent = ''; } }
  }

  // Task runner: submit and stream output
  function bindTaskRunner(){
    const form = document.getElementById('task-form');
    const nameSel = document.getElementById('task-name');
    const argsInput = document.getElementById('task-args');
    const statusEl = document.getElementById('task-status');
    const streamEl = document.getElementById('task-stream');
    const stopBtn = document.getElementById('task-stream-stop');
    const clearBtn = document.getElementById('task-stream-clear');
    const endStatusEl = document.getElementById('task-end-status');
    let es;

    async function openStream(taskId){
      if(es){ es.close(); }
      streamEl.textContent = '';
      endStatusEl.textContent = '';
      es = new EventSource(`/admin/api/tasks/${encodeURIComponent(taskId)}/stream`, { withCredentials: true });
      es.addEventListener('log', (ev)=>{ appendLog(streamEl, ev.data); });
      es.addEventListener('end', (ev)=>{ endStatusEl.textContent = `Task finished: ${ev.data}`; if(es){ es.close(); } });
      es.onerror = ()=>{ /* keep open; server sends end on completion */ };
    }

    if(form){
      form.addEventListener('submit', async (ev)=>{
        ev.preventDefault();
        const name = nameSel.value;
        const rawArgs = (argsInput.value||'').trim();
        const args = rawArgs ? rawArgs.split(',').map(s=>s.trim()).filter(Boolean) : [];
        statusEl.textContent = 'Starting task...';
        try {
          const body = JSON.stringify({ name, args });
          const result = await fetchJSON('/admin/api/tasks/run', { method:'POST', headers:{'Content-Type':'application/json'}, body });
          statusEl.textContent = `Started task ${result.task_id} (pid ${result.pid})`;
          await openStream(result.task_id);
        } catch(e){ statusEl.textContent = 'Error: '+e.message; }
      });
    }
    if(stopBtn){ stopBtn.onclick = ()=>{ if(es){ es.close(); es = null; } } }
    if(clearBtn){ clearBtn.onclick = ()=>{ streamEl.textContent = ''; endStatusEl.textContent=''; } }
  }

  await loadStaticPanels();
  await loadMetrics();
  loadDriftSummary(); loadBackfillJobs(); loadBackfillCompleteness();
  await bindActions();
  setInterval(loadMetrics, 10000);
  setInterval(loadDriftSummary, 60000); setInterval(loadBackfillJobs, 45000); setInterval(loadBackfillCompleteness, 60000);
  loadLatency(); setInterval(loadLatency, 15000);
  loadPnlTs(); setInterval(loadPnlTs, 30000);
  // New panels init
  await loadVerification(false); renderFreshness(); updateFreshnessLoop(); bindVerificationActions(); startHeartbeat();

  // Combined events stream (latency+pnl snapshots)
  (function startEvents(){
    try {
      const es = new EventSource('/admin/api/events/stream', { withCredentials: true });
      es.addEventListener('snapshot', (ev)=>{
        try {
          const data = JSON.parse(ev.data);
          // Optional: Could merge into existing panels or log somewhere
        } catch(_){}
      });
      es.onerror = ()=>{ /* silent retry via browser */ };
    } catch(_){ }
  })();
  // Initialize streams and runner
  startLogsStream();
  bindTaskRunner();
})();

// PnL chart now rendered via shared ChartUtils.renderLineChart (legend & tooltips)