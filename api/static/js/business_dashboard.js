(async function(){
  const $ = (s)=>document.querySelector(s);
  async function fetchJSON(url){
    const r=await fetch(url,{credentials:'include'});
    if(!r.ok) throw new Error(url+': '+r.status);
    return r.json();
  }
  function renderText(id, data){ const el=document.getElementById(id); if(el) el.textContent=JSON.stringify(data,null,2); }

  // Minimal chart renderer for sparkline using canvas
  function drawSparkline(canvas, series){
    if(!canvas || !series || series.length<2) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth || 800;
    const h = canvas.height = canvas.clientHeight || 240;
    const min = Math.min(...series), max = Math.max(...series);
    const pad = 10;
    const scaleX = (w - 2*pad) / (series.length - 1);
    const scaleY = (h - 2*pad) / (max - min || 1);
    ctx.clearRect(0,0,w,h);
    // grid
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
    for(let i=0;i<5;i++){ const y = pad + i*(h-2*pad)/4; ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(w-pad,y); ctx.stroke(); }
    // line
    ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2; ctx.beginPath();
    series.forEach((v,i)=>{
      const x = pad + i*scaleX;
      const y = h - pad - (v - min) * scaleY;
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
  }

  async function loadKpis(){ try { renderText('kpi-body', await fetchJSON('/business/api/kpis')); } catch(e){ renderText('kpi-body',{error:e.message}); } }

  async function loadCoverage(){
    try {
      const data = await fetchJSON('/business/api/coverage/summary');
      renderText('coverage-body', { timestamp: data.timestamp, ratios: data.ratios });
      if (data.history) {
        ChartUtils.renderLineChart('coverageChart', data.history.map(p=>({ts:p.ts,equities:p.equities,options:p.options})), {
          series:[
            {key:'equities', color:'#2b8a3e', name:'Equities'},
            {key:'options', color:'#1c7ed6', name:'Options'}
          ]
        });
      }
    } catch(e){ renderText('coverage-body',{error:e.message}); }
  }

  async function loadIngestion(){
    try { renderText('ingestion-body', await fetchJSON('/business/api/ingestion/health')); } catch(e){ renderText('ingestion-body',{error:e.message}); }
  }

  async function loadNewsSentiment(){
    try {
      const data = await fetchJSON('/business/api/news/sentiment');
      renderText('news-sentiment-body', {sentiment_1d_avg:data.sentiment_1d_avg, sentiment_7d_avg:data.sentiment_7d_avg, anomaly_delta:data.anomaly_delta});
      if(data.history){
        const points = data.history.map(h=>({ ts: h.date+'T00:00:00Z', avg: h.avg_sentiment }));
        ChartUtils.renderLineChart('sentimentChart', points, { series:[{key:'avg', color:'#e8590c', name:'Avg Sentiment'}] });
      }
    } catch(e){ renderText('news-sentiment-body',{error:e.message}); }
  }

  async function populateCompanies(){
    const select = $('#company-select');
    const link = $('#company-link');
    const listEl = $('#company-list');
    try {
      const data = await fetchJSON('/business/api/companies');
      const companies = data.companies || [];
      // render list
      listEl.textContent = companies.join(', ');
      // fill selector
      select.innerHTML = companies.map(s=>`<option value="${s}">${s}</option>`).join('');
      if(companies.length){
        select.value = companies[0];
        link.href = `/business/company/${companies[0]}`;
        await loadCompany(companies[0]);
      }
      select.addEventListener('change', async ()=>{
        const sym = select.value;
        link.href = `/business/company/${sym}`;
        await loadCompany(sym);
      });
    } catch(e){ listEl.textContent = `Error: ${e.message}`; }
  }

  async function loadCompany(symbol){
    const canvas = document.getElementById('sparkline-canvas');
    try {
      const spark = await fetchJSON(`/business/api/company/${symbol}/sparkline`);
      drawSparkline(canvas, spark.series || []);
    } catch(e){ /* ignore chart errors but keep UI responsive */ }
    try { renderText('forecast-body', await fetchJSON(`/business/api/company/${symbol}/forecast`)); } catch(e){ renderText('forecast-body',{error:e.message}); }
    try { renderText('report-body', await fetchJSON(`/business/api/company/${symbol}/report`)); } catch(e){ renderText('report-body',{error:e.message}); }
  }

  // initial loads
  await loadKpis();
  await populateCompanies();
  // periodic refresh for KPIs
  setInterval(loadKpis, 15000);
  loadCoverage(); setInterval(loadCoverage, 30000);
  loadIngestion(); setInterval(loadIngestion, 20000);
  loadNewsSentiment(); setInterval(loadNewsSentiment, 60000);
})();

// Coverage & sentiment charts now rendered via ChartUtils.renderLineChart (with legend & tooltips)