(async function(){
  const symbol = window.location.pathname.split('/').pop();
  async function fetchJSON(url){const r=await fetch(url,{credentials:'include'}); if(!r.ok) throw new Error(url+': '+r.status); return r.json(); }
  function render(id,data){const el=document.getElementById(id); if(el) el.textContent=JSON.stringify(data,null,2); }
  try { render('forecast-body', await fetchJSON(`/business/api/company/${symbol}/forecast`)); } catch(e){ render('forecast-body',{error:e.message}); }
  try { render('report-body', await fetchJSON(`/business/api/company/${symbol}/report`)); } catch(e){ render('report-body',{error:e.message}); }
})();