const $ = id => document.getElementById(id);
let project = null, frame = 0, image = null, savedMasks = [], points = [], draft = null, draftImage = null, draftMask = null;
let colorLabel = null;
let busy = false, dirty = false, positive = true, editing = null;
let displayMode = 'fit', displayZoom = 1;
try { const preference=JSON.parse(localStorage.getItem('labeler-display')); if(preference?.mode==='custom'&&Number.isFinite(preference.zoom)){displayMode='custom';displayZoom=Math.max(.01,Math.min(8,preference.zoom));} } catch {}
const palette = [['Mint','#68ddbc'],['Blue','#82b6ff'],['Orange','#f6bf70'],['Purple','#cc99ff'],['Pink','#ff91ac'],['Lime','#bbdc77'],['Red','#f46d6d'],['White','#ffffff']];
const canvas = $('canvas'), ctx = canvas.getContext('2d');

function status(text, error=false) { $('status').textContent = text; $('status').className = error ? 'error' : ''; }
function controls() {
  for (const el of document.querySelectorAll('button')) el.disabled = busy || (!project && !['open','choose'].includes(el.id));
  $('folder').disabled = busy; $('timeline').disabled = busy || !project;
  $('class').disabled = busy; $('object').disabled = busy;
  $('newlabel').disabled = busy || !project; $('zoom').disabled = busy || !image;
  for(const id of ['newcolor','editcolor','opacity'])$(id).disabled=busy || !project;
  for(const id of ['zoomout','zoomin','fit'])$(id).disabled=busy || !image;
  $('save').disabled = busy || !draft; $('undo').disabled = busy || !points.length;
  $('prev').disabled = busy || !project || frame === 0;
  $('next').disabled = busy || !project || frame === project.frames.length-1;
}
async function task(fn) {
  if (busy) return;
  busy=true; controls();
  try { await fn(); } catch(e) { status(e.message, true); }
  finally { busy=false; controls(); }
}
async function api(path, body, method) {
  const multipart = body instanceof FormData;
  const response = await fetch(path, {method: method || (body ? 'POST' : 'GET'),
    headers: body && !multipart ? {'Content-Type':'application/json'} : {}, body: multipart ? body : body ? JSON.stringify(body) : undefined});
  if (!response.ok) {
    let detail; try { detail=(await response.json()).detail; } catch { detail='Backend error. See server log.'; }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return response.json();
}
function loadImage(src) {
  return new Promise((resolve,reject)=>{ const im=new Image(); im.onload=()=>resolve(im); im.onerror=()=>reject(new Error('Could not load image.')); im.src=src; });
}
function tint(mask, color) {
  const c=document.createElement('canvas'); c.width=mask.width;c.height=mask.height;
  const x=c.getContext('2d'); x.drawImage(mask,0,0); const pixels=x.getImageData(0,0,c.width,c.height);
  const rgb=color.match(/\w\w/g).map(v=>parseInt(v,16));
  for(let i=0;i<pixels.data.length;i+=4){ const a=pixels.data[i]>127?255:0; pixels.data[i]=rgb[0];pixels.data[i+1]=rgb[1];pixels.data[i+2]=rgb[2];pixels.data[i+3]=a; }
  x.putImageData(pixels,0,0);return c;
}
function draw() {
  if(!image)return;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);
  if($('overlays').checked){
    ctx.save();ctx.globalAlpha=Number($('opacity').value)/100;
    for(const m of savedMasks) if(m.id!==editing)ctx.drawImage(m.canvas,0,0);
    if(draftImage)ctx.drawImage(draftImage,0,0);
    ctx.restore();
  }
  const radius=5*canvas.width/Math.max(canvas.getBoundingClientRect().width,1);
  for(const p of points){ctx.beginPath();ctx.arc(p.x,p.y,radius,0,2*Math.PI);ctx.fillStyle=p.label?'#68ffba':'#ff657f';ctx.fill();ctx.strokeStyle='#10151c';ctx.lineWidth=radius/3;ctx.stroke();}
}
function resetDraft() { points=[];draft=null;draftImage=null;draftMask=null;dirty=false;editing=null;$('object').value='';$('score').textContent='Click inside an object to start.';draw(); }
function canDiscard() { return !dirty || confirm('Discard the unsaved object preview? Saved masks will stay on disk.'); }
function labelColor(name){return project?.label_colors?.[name]||'#68ddbc';}
function retintDraft(){draftImage=draftMask?tint(draftMask,labelColor($('class').value)):null;draw();}
function retintMasks(){for(const mask of savedMasks)mask.canvas=tint(mask.raw,labelColor(mask.label));retintDraft();}
function renderPalette(id,inputId){
  $(id).replaceChildren(...palette.map(([name,color])=>{
    const button=document.createElement('button');button.type='button';button.style.background=color;
    button.title=`${name} ${color}`;button.setAttribute('aria-label',`${id==='new-palette'?'New':'Existing'} label color ${name}`);
    button.setAttribute('aria-pressed',String($(inputId).value.toLowerCase()===color));
    button.onclick=()=>{if(busy||!project)return;$(inputId).value=color;renderPalette(id,inputId);controls();};return button;
  }));
}
function updateLabels() {
  const chosen=$('class').value, names=project.labels||[];
  const placeholder=document.createElement('option');placeholder.value='';placeholder.textContent='Select a label';
  $('class').replaceChildren(placeholder,...names.map(name=>{const option=document.createElement('option');option.value=name;option.textContent=name;return option;}));
  $('class').value=names.includes(chosen)?chosen:(names[0]||'');
  $('label-count').textContent=names.length;
  $('label-list').replaceChildren();
  if(!names.length){const hint=document.createElement('p');hint.className='hint';hint.textContent='No labels yet. Add a label above.';$('label-list').append(hint);}
  const annotations=Object.values(project.annotations).flat();
  for(const name of names){
    const entry=document.createElement('div');entry.className='label-entry';
    const swatch=document.createElement('button');swatch.className='label-color';swatch.style.background=labelColor(name);
    swatch.setAttribute('aria-label',`Change color for ${name}`);swatch.title=`Change ${name} color (${labelColor(name)})`;
    swatch.onclick=()=>{if(busy)return;colorLabel=name;$('color-label').textContent=`Color for ${name}`;$('editcolor').value=labelColor(name);$('color-editor').hidden=false;renderPalette('edit-palette','editcolor');controls();};
    const button=document.createElement('button');button.className='label-option';button.classList.toggle('selected',name===$('class').value);
    const label=document.createElement('span');label.textContent=name;
    const count=document.createElement('span');const total=annotations.filter(a=>a.class_name===name).length;count.textContent=`${total} saved`;
    button.append(label,count);button.onclick=()=>{if(busy)return;$('class').value=name;if(draft)dirty=true;updateLabels();controls();};
    entry.append(swatch,button);$('label-list').append(entry);
  }
  retintDraft();
}
function updateProject() {
  const annotations=project.annotations[String(frame)]||[];
  $('count').textContent=annotations.length;
  $('objects').replaceChildren();
  if(!annotations.length){const p=document.createElement('p');p.className='hint';p.textContent='No saved objects yet.';$('objects').append(p);}
  for(const a of annotations){
    const card=document.createElement('div');card.className='object';
    const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=labelColor(a.class_name);
    const title=document.createElement('span');title.textContent=`#${a.object_id} · ${a.class_name}`;
    const row=document.createElement('div');row.className='row';
    const edit=document.createElement('button');edit.textContent='Refine';edit.onclick=()=>{if(busy||!canDiscard())return;task(async()=>{editing=a.object_id;$('object').value=a.object_id;$('class').value=a.class_name;updateLabels();points=a.points.map(p=>({...p}));dirty=true;await predict();});};
    const remove=document.createElement('button');remove.textContent='Delete';remove.onclick=()=>{if(busy||!confirm(`Delete saved object #${a.object_id} on this frame?`))return;task(async()=>{project=await api(`/api/mask/${frame}/${a.object_id}?project=${project.id}`,null,'DELETE');if(editing===a.object_id)resetDraft();await refreshMasks();updateProject();status('Object deleted.');});};
    row.append(edit,remove);card.append(swatch,title,row);$('objects').append(card);
  }
  updateLabels();
  $('opacity').value=Math.round((project.overlay_opacity??.45)*100);$('opacity-value').textContent=`${$('opacity').value}%`;
  $('savepath').textContent=`Saved to ${project.save_directory}`;
}
async function refreshMasks() {
  savedMasks=await Promise.all((project.annotations[String(frame)]||[]).map(async a=>{
    const raw=await loadImage(`/api/mask/${frame}/${a.object_id}?project=${project.id}&t=${Date.now()}`);
    return {id:a.object_id,label:a.class_name,raw,canvas:tint(raw,labelColor(a.class_name))};
  }));
  draw();
}
async function go(index) {
  if(!project||index<0||index>=project.frames.length)return;
  status('Loading frame…');
  const nextImage=await loadImage(`/api/frame/${index}?project=${project.id}`);
  frame=index;image=nextImage;savedMasks=[];canvas.width=image.width;canvas.height=image.height;
  resetDraft();canvas.hidden=false;$('canvas-wrap').hidden=false;$('empty').hidden=true;applyDisplay();
  $('filename').textContent=project.frames[frame].name;$('dimensions').textContent=`${image.width} × ${image.height}`;
  $('timeline').max=project.frames.length-1;$('timeline').value=frame;$('framecount').textContent=`${frame+1} / ${project.frames.length}`;
  await refreshMasks();updateProject();draw();status('Ready. Click inside an object.');
  localStorage.setItem('labeler-frame',JSON.stringify({project:project.id,frame}));
}
function applyDisplay() {
  if(!image)return;
  const viewport=$('viewport');
  if(displayMode==='fit')displayZoom=Math.min(1,Math.max(.001,Math.min((viewport.clientWidth-26)/image.width,(viewport.clientHeight-26)/image.height)));
  canvas.style.width=`${image.width*displayZoom}px`;canvas.style.height=`${image.height*displayZoom}px`;
  $('zoom').value=Number((displayZoom*100).toFixed(1));
  $('fit').classList.toggle('selected',displayMode==='fit');draw();
}
function setZoom(value) {
  if(!image||busy)return;
  if(!Number.isFinite(value)||value<1||value>800){$('zoom').value=Number((displayZoom*100).toFixed(1));status('Display zoom must be between 1% and 800%.',true);return;}
  displayMode='custom';displayZoom=value/100;applyDisplay();
  localStorage.setItem('labeler-display',JSON.stringify({mode:displayMode,zoom:displayZoom}));
  status(`Display zoom ${value.toFixed(0)}%. SAM 2 and exported masks stay at ${image.width} × ${image.height}.`);
}
async function predict() {
  draft=null;draftImage=null;draftMask=null;draw();
  if(!points.some(p=>p.label===1)){status('Add a positive point inside the object.');$('score').textContent='No positive points.';return;}
  status('Generating mask… First prediction loads the model.');
  const result=await api('/api/predict',{project:project.id,frame,points});
  draftMask=await loadImage('data:image/png;base64,'+result.mask);
  draft=result.draft;retintDraft();
  $('score').textContent=`Model score ${(result.score*100).toFixed(1)}% · ${result.pixels.toLocaleString()} pixels`;
  status('Preview ready. Refine with more clicks, or save the object.');
}
function setMode(value){positive=value;$('positive').classList.toggle('selected',value);$('negative').classList.toggle('selected',!value);}
$('positive').onclick=()=>setMode(true);$('negative').onclick=()=>setMode(false);
async function activateProject(nextProject) {
  project=nextProject;image=null;savedMasks=[];resetDraft();colorLabel=null;$('color-editor').hidden=true;canvas.hidden=true;$('canvas-wrap').hidden=true;$('empty').hidden=false;
  let index=0;try{const last=JSON.parse(localStorage.getItem('labeler-frame'));if(last?.project===project.id)index=Math.min(last.frame,project.frames.length-1);}catch{}
  await go(index);
}
$('open').onclick=()=>{if(!canDiscard())return;task(async()=>{status('Opening folder…');const nextProject=await api('/api/open',{folder:$('folder').value});localStorage.setItem('labeler-folder',$('folder').value);await activateProject(nextProject);$('foldername').textContent=`${project.frames.length} images loaded from folder path.`;});};
$('choose').onclick=()=>{if(busy||!canDiscard())return;$('folderfiles').value='';$('folderfiles').click();};
$('folderfiles').onchange=()=>task(async()=>{
  const images=Array.from($('folderfiles').files).filter(file=>/\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name));
  if(!images.length)throw new Error('The selected folder contains no supported images.');
  const data=new FormData();for(const file of images)data.append('files',file,file.webkitRelativePath||file.name);
  const name=(images[0].webkitRelativePath||images[0].name).split('/')[0];
  status(`Reading ${images.length} images from ${name}…`);
  await activateProject(await api('/api/import',data));
  $('foldername').textContent=`${name} · ${project.frames.length} images`;
});
$('addlabel').onclick=()=>task(async()=>{
  const name=$('newlabel').value.trim();if(!name)throw new Error('Enter a new label name.');
  project=await api('/api/labels',{project:project.id,name,color:$('newcolor').value});
  updateLabels();$('class').value=project.labels[project.labels.length-1];updateLabels();$('newlabel').value='';
  if(draft)dirty=true;status('Label created and selected.');
});
$('newcolor').oninput=()=>{renderPalette('new-palette','newcolor');controls();};
$('editcolor').oninput=()=>{renderPalette('edit-palette','editcolor');controls();};
$('savecolor').onclick=()=>task(async()=>{
  if(!colorLabel)return;
  project=await api('/api/labels/color',{project:project.id,name:colorLabel,color:$('editcolor').value},'PATCH');
  retintMasks();updateProject();draw();$('color-editor').hidden=true;colorLabel=null;status('Label color saved. All objects with this label use the selected color.');
});
$('cancelcolor').onclick=()=>{colorLabel=null;$('color-editor').hidden=true;};
$('opacity').oninput=()=>{$('opacity-value').textContent=`${$('opacity').value}%`;draw();};
$('opacity').onchange=()=>task(async()=>{
  try{project=await api('/api/overlay',{project:project.id,opacity:Number($('opacity').value)/100},'PATCH');status('Overlay opacity saved.');}
  finally{$('opacity').value=Math.round((project.overlay_opacity??.45)*100);$('opacity-value').textContent=`${$('opacity').value}%`;draw();}
});
$('newlabel').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();$('addlabel').click();}};
$('zoom').onchange=()=>setZoom(Number($('zoom').value));
$('zoom').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();setZoom(Number($('zoom').value));}};
$('zoomout').onclick=()=>setZoom(Math.max(1,Math.round(displayZoom*100/1.25)));
$('zoomin').onclick=()=>setZoom(Math.min(800,Math.round(displayZoom*100*1.25)));
$('fit').onclick=()=>{displayMode='fit';applyDisplay();$('viewport').scrollTo(0,0);localStorage.setItem('labeler-display',JSON.stringify({mode:'fit'}));status('Image fitted to the view. Original image and export size are unchanged.');};
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{
  if(busy||!image||!project||![0,2].includes(e.button))return;
  e.preventDefault();const rect=canvas.getBoundingClientRect();
  points.push({x:Math.min(image.width-1,Math.max(0,(e.clientX-rect.left)*image.width/rect.width)),y:Math.min(image.height-1,Math.max(0,(e.clientY-rect.top)*image.height/rect.height)),label:e.button===2?0:(positive?1:0)});
  dirty=true;task(predict);
});
$('undo').onclick=()=>{if(!points.length)return;points.pop();dirty=true;task(predict);};
$('clear').onclick=()=>{points=[];draft=null;draftImage=null;draftMask=null;dirty=false;editing=null;draw();$('score').textContent='Click inside an object to start.';status('Clicks cleared.');controls();};
$('new').onclick=()=>{if(canDiscard()){resetDraft();controls();status('Ready for a new object.');}};
$('save').onclick=()=>task(async()=>{
  const label=$('class').value.trim();if(!label)throw new Error('Enter a class name before saving.');
  const oid=$('object').value?Number($('object').value):null;
  if(oid!==null&&(!Number.isInteger(oid)||oid<1||oid>2147483647))throw new Error('Object ID must be a positive integer.');
  if((project.annotations[String(frame)]||[]).some(a=>a.object_id===oid)&&editing!==oid&&!confirm(`Replace object #${oid} on this frame?`))return;
  project=await api('/api/save',{project:project.id,frame,draft,class_name:label,object_id:oid});
  resetDraft();await refreshMasks();updateProject();status('Object saved to disk. Add another object or move to the next frame.');
});
function navigate(index){if(busy||!canDiscard()){ $('timeline').value=frame;return; }task(()=>go(index));}
$('prev').onclick=()=>navigate(frame-1);$('next').onclick=()=>navigate(frame+1);
$('timeline').onchange=()=>navigate(Number($('timeline').value));$('overlays').onchange=draw;
$('class').onchange=()=>{if(draft)dirty=true;updateLabels();controls();};$('object').oninput=()=>{if(draft)dirty=true;};
$('export').onclick=()=>{if(dirty&&!confirm('Export saved masks only? The current preview has not been saved.'))return;window.location.href=`/api/export?project=${project.id}`;};
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
window.addEventListener('resize',applyDisplay);
window.addEventListener('keydown',e=>{if(busy||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;const ids={ArrowLeft:'prev',ArrowRight:'next',z:'undo',Z:'undo',Enter:'save'};if(ids[e.key]){e.preventDefault();$(ids[e.key]).click();}});
$('folder').value=localStorage.getItem('labeler-folder')||'';
renderPalette('new-palette','newcolor');renderPalette('edit-palette','editcolor');
controls();api('/api/status').then(s=>{$('device').textContent=s.device;}).catch(e=>status(e.message,true));
