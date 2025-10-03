/* Shared lightweight chart utilities (no external deps)
 * Features: scales, axes, legend, tooltip hit detection, line rendering.
 * CSP-friendly: no inline scripts required; consumers pass canvas IDs.
 */
(function(global){
  // Shared palette for multi-series visuals
  const palette = ['#58a6ff','#2ea043','#d29922','#f85149','#a371f7','#79c0ff','#3fb950','#ffdd57','#ffa657','#ff7b72'];
  function createScales(points, seriesDefs, w, h, padding){
    const xs = points.map(p=> new Date(p.ts || p.date || p.d || p.time).getTime()).filter(n=>!isNaN(n));
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const allVals = [];
    seriesDefs.forEach(s=>{ points.forEach(p=>{ const v = p[s.key]; if(v!=null) allVals.push(v); }); });
    const minV = Math.min(...allVals), maxV = Math.max(...allVals);
    const xScale = t => padding.left + ((t - minX) / (maxX - minX || 1)) * (w - padding.left - padding.right);
    const yScale = v => (h - padding.bottom) - ((v - minV) / (maxV - minV || 1)) * (h - padding.top - padding.bottom);
    return {xScale, yScale, minX, maxX, minV, maxV};
  }

  function drawAxes(ctx, w, h, scales, padding){
    ctx.strokeStyle='#444'; ctx.lineWidth=1; ctx.beginPath();
    ctx.moveTo(padding.left, padding.top); ctx.lineTo(padding.left, h - padding.bottom); ctx.lineTo(w - padding.right, h - padding.bottom); ctx.stroke();
    ctx.fillStyle='#666'; ctx.font='10px sans-serif';
    ctx.fillText(scales.maxV.toFixed(2), 4, padding.top+8);
    ctx.fillText(scales.minV.toFixed(2), 4, h - padding.bottom - 2);
  }

  function drawSeries(ctx, points, seriesDefs, scales){
    seriesDefs.forEach(s=>{
      ctx.beginPath(); ctx.strokeStyle=s.color; ctx.lineWidth=2; let first=true;
      points.forEach(p=>{ const v=p[s.key]; if(v==null) return; const x=scales.xScale(new Date(p.ts||p.date||p.d||p.time).getTime()); const y=scales.yScale(v); if(first){ctx.moveTo(x,y); first=false;} else ctx.lineTo(x,y); });
      ctx.stroke();
    });
  }

  function hitTest(points, seriesDefs, scales, mx, my, radius){
    let nearest=null, bestDist=Infinity;
    points.forEach(p=>{
      const t = new Date(p.ts||p.date||p.d||p.time).getTime();
      const x = scales.xScale(t);
      seriesDefs.forEach(s=>{
        const v=p[s.key]; if(v==null) return; const y=scales.yScale(v);
        const dx=mx-x, dy=my-y; const dist=Math.sqrt(dx*dx+dy*dy);
        if(dist < radius && dist < bestDist){ bestDist=dist; nearest={point:p, series:s, x,y}; }
      });
    });
    return nearest;
  }

  function drawLegend(ctx, seriesDefs, w, padding){
    let x = padding.left; const y = 4; ctx.font='11px sans-serif';
    seriesDefs.forEach(s=>{ ctx.fillStyle=s.color; ctx.fillRect(x,y,10,10); ctx.fillStyle='#ccc'; ctx.fillText(s.name||s.key, x+14, y+10); x += ctx.measureText((s.name||s.key)+'   ').width + 28; });
  }

  function renderLineChart(canvasId, points, cfg){
    const canvas=document.getElementById(canvasId); if(!canvas||!points||!points.length) return;
    const ctx=canvas.getContext('2d');
    const w=canvas.width = canvas.clientWidth || 800;
    const h=canvas.height = canvas.clientHeight || 180;
    const padding={left:40,right:10,top:16,bottom:24};
    ctx.clearRect(0,0,w,h);
    const seriesDefs = cfg.series;
    const scales = createScales(points, seriesDefs, w, h, padding);
    drawAxes(ctx, w, h, scales, padding);
    drawSeries(ctx, points, seriesDefs, scales);
    drawLegend(ctx, seriesDefs, w, padding);

    // Tooltip layer (simple redraw on move)
    function handleMove(ev){
      const rect=canvas.getBoundingClientRect();
      const mx=ev.clientX-rect.left, my=ev.clientY-rect.top;
      ctx.clearRect(0,0,w,h);
      drawAxes(ctx,w,h,scales,padding); drawSeries(ctx,points,seriesDefs,scales); drawLegend(ctx,seriesDefs,w,padding);
      const hit = hitTest(points, seriesDefs, scales, mx, my, 8);
      if(hit){
        ctx.fillStyle='#222a'; ctx.strokeStyle='#fff'; ctx.beginPath(); ctx.arc(hit.x,hit.y,5,0,Math.PI*2); ctx.fill(); ctx.stroke();
        const tooltip = `${hit.series.name||hit.series.key}: ${hit.point[hit.series.key]}`;
        const tw = ctx.measureText(tooltip).width + 8;
        const tx = Math.min(w - tw - 4, Math.max(4, hit.x + 10));
        const ty = Math.max(18, hit.y - 24);
        ctx.fillStyle='#000c'; ctx.fillRect(tx, ty-12, tw, 16);
        ctx.fillStyle='#fff'; ctx.font='11px sans-serif'; ctx.fillText(tooltip, tx+4, ty);
      }
    }
    canvas.onmousemove = handleMove;
    canvas.onmouseleave = ()=>{ ctx.clearRect(0,0,w,h); drawAxes(ctx,w,h,scales,padding); drawSeries(ctx,points,seriesDefs,scales); drawLegend(ctx,seriesDefs,w,padding); };
  }

  global.ChartUtils = { renderLineChart, palette };
})(window);
