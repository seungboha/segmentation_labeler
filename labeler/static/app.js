const $ = id => document.getElementById(id);
let project = null, frame = 0, image = null, savedMasks = [], history = [], draft = null, draftImage = null, draftMask = null;
let colorLabel = null, stroke = null;
let pendingCvat = null;
let tool = 'brush', samPoints = [], samBase = null, samBaseDirty = false, samBasePoints = [], samPending = false, samUsedPoints = [];
let busy = false, dirty = false, positive = true, editing = null;
let displayMode = 'fit', displayZoom = 1;
try { const preference=JSON.parse(localStorage.getItem('labeler-display')); if(preference?.mode==='custom'&&Number.isFinite(preference.zoom)){displayMode='custom';displayZoom=Math.max(.01,Math.min(8,preference.zoom));} } catch {}
const palette = [['Mint','#68ddbc'],['Blue','#82b6ff'],['Orange','#f6bf70'],['Purple','#cc99ff'],['Pink','#ff91ac'],['Lime','#bbdc77'],['Red','#f46d6d'],['White','#ffffff']];
const canvas = $('canvas'), ctx = canvas.getContext('2d');

function status(text, error=false) { $('status').textContent = text; $('status').className = error ? 'error' : ''; }
function controls() {
  for (const el of document.querySelectorAll('button')) el.disabled = busy || (!project && !['open','choose'].includes(el.id));
  $('folder').disabled = busy; $('timeline').disabled = busy || !project;
  $('brushsize').disabled = busy || !image || tool==='sam';
  $('loadmask').disabled = busy || !image;
  $('class').disabled = busy; $('object').disabled = busy;
  $('newlabel').disabled = busy || !project; $('zoom').disabled = busy || !image;
  for(const id of ['newcolor','editcolor','opacity'])$(id).disabled=busy || !project;
  for(const id of ['zoomout','zoomin','fit'])$(id).disabled=busy || !image;
  $('save').disabled = busy || !draft || !!stroke || samPending; $('undo').disabled = busy || !history.length;
  $('sam-operation').disabled=busy || !image;
  $('sam-undo').disabled=busy || !samPoints.length;
  $('sam-retry').disabled=busy || !samPoints.some(point=>point.label===1);
  $('sam-next').disabled=busy || !samPoints.length;
  $('prev').disabled = busy || !project || frame === 0;
  $('next').disabled = busy || !project || frame === project.frames.length-1;
  $('save-version').disabled = busy || !project || !!stroke;
  for(const id of ['version-name','version-note'])$(id).disabled=busy;
}
async function task(fn) {
  if (busy) return;
  busy=true; controls();
  try { await fn(); } catch(e) { status(e.message, true);if($('version-dialog').open)$('version-error').textContent=e.message;if($('cvat-dialog').open)$('cvat-error').textContent=e.message; }
  finally { busy=false; controls(); }
}
async function api(path, body, method) {
  const multipart = body instanceof FormData;
  const response = await fetch(path, {method: method || (body ? 'POST' : 'GET'),
    headers: body && !multipart ? {'Content-Type':'application/json'} : {}, body: multipart ? body : body ? JSON.stringify(body) : undefined});
  if (!response.ok) {
    let detail; try { detail=(await response.json()).detail; } catch { detail='Backend error. See server log.'; }
    // Validation errors can include the entire submitted PNG in `input`.
    // Display only the field and message, never the base64 mask payload.
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map(error=>`${(error.loc||[]).filter(part=>part!=='body').join('.')}: ${error.msg||'Invalid value'}`).join('; ')
      : 'Backend error. See server log.');
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
    if(tool==='sam'){
      const radius=5*canvas.width/Math.max(canvas.getBoundingClientRect().width,1);
      for(const point of samPoints){ctx.beginPath();ctx.arc(point.x,point.y,radius,0,Math.PI*2);ctx.fillStyle=point.label?'#68ffba':'#ff657f';ctx.fill();ctx.strokeStyle='#10151c';ctx.lineWidth=radius/3;ctx.stroke();}
    }
  }
}
function resetDraft() { samPoints=[];samBase=null;samPending=false;samUsedPoints=[];samBasePoints=[];history=[];draft=null;draftImage=null;draftMask=null;dirty=false;editing=null;$('object').value='';$('score').textContent='Load a mask, paint, or use SAM2 points.';draw(); }
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
    const edit=document.createElement('button');edit.textContent='Edit mask';edit.onclick=()=>{if(busy||!canDiscard())return;task(async()=>{resetDraft();editing=a.object_id;$('object').value=a.object_id;$('class').value=a.class_name;updateLabels();setDraft(savedMasks.find(m=>m.id===editing).raw);history=[];dirty=false;status('Mask ready. Drag to paint or erase, then save.');});};
    const remove=document.createElement('button');remove.textContent='Delete';remove.onclick=()=>{if(busy||!confirm(`Delete saved object #${a.object_id} on this frame?`))return;task(async()=>{project=await api(`/api/mask/${frame}/${a.object_id}?project=${project.id}`,null,'DELETE');if(editing===a.object_id)resetDraft();await refreshMasks();updateProject();status('Object deleted.');});};
    row.append(edit,remove);card.append(swatch,title,row);$('objects').append(card);
  }
  updateLabels();
  $('opacity').value=Math.round((project.overlay_opacity??.45)*100);$('opacity-value').textContent=`${$('opacity').value}%`;
  $('savepath').textContent=`Saved to ${project.save_directory}`;
  renderVersions();
}
function renderVersions(){
  const versions=project.mask_versions||[];
  $('version-count').textContent=versions.length;$('version-list').replaceChildren();
  if(!versions.length){const hint=document.createElement('p');hint.className='hint';hint.textContent='No versions saved yet.';$('version-list').append(hint);}
  for(const version of [...versions].reverse()){
    const card=document.createElement('div');card.className='version-card';
    const title=document.createElement('strong');title.textContent=`v${version.number} · ${version.name}`;
    const detail=document.createElement('p');detail.className='hint';detail.textContent=`${new Date(version.created_at).toLocaleString()} · ${version.mask_count} masks`;
    const note=document.createElement('p');note.className='hint version-note';note.textContent=version.note;
    const download=document.createElement('button');download.textContent='Export';download.setAttribute('aria-label',`Export version ${version.number}: ${version.name}`);
    download.onclick=()=>{if(busy)return;window.location.href=`/api/versions/${version.id}/export?project=${project.id}`;};
    card.append(title,detail);if(version.note)card.append(note);card.append(download);$('version-list').append(card);
  }
}
function versionNaming(){
  $('version-unsaved').hidden=true;$('version-form').hidden=false;$('version-error').textContent='';
  $('version-name').focus();
}
$('save-version').onclick=()=>{
  if(busy||!project||stroke)return;
  $('version-name').value='';$('version-note').value='';$('version-error').textContent='';
  $('version-unsaved').hidden=!dirty;$('version-form').hidden=dirty;
  $('version-dialog').showModal();if(!dirty)$('version-name').focus();
};
$('version-cancel').onclick=()=>{if(!busy)$('version-dialog').close();};
$('version-dialog').addEventListener('cancel',e=>{if(busy)e.preventDefault();});
$('version-discard').onclick=()=>{resetDraft();controls();versionNaming();};
$('version-save-object').onclick=async()=>{await task(async()=>{if(await saveCurrentObject())versionNaming();});if(!$('version-form').hidden)$('version-name').focus();};
$('version-form').onsubmit=e=>{
  e.preventDefault();if(busy||dirty||stroke)return;
  task(async()=>{
    const name=$('version-name').value.trim();if(!name)throw new Error('Enter a version name.');
    status('Saving the complete mask set…');
    project=await api('/api/versions',{project:project.id,name,note:$('version-note').value});
    renderVersions();$('version-dialog').close();status(`Version “${name}” saved for the entire mask set.`);
  });
};
$('load-cvat').onclick=()=>{if(busy||stroke||!project||!canDiscard())return;$('cvat-file').value='';$('cvat-file').click();};
$('cvat-file').onchange=()=>{
  const file=$('cvat-file').files[0];if(!file)return;
  task(async()=>{
    status('Reading CVAT annotations and matching source frames…');
    const data=new FormData();data.append('project',project.id);data.append('file',file);
    pendingCvat=await api('/api/import/cvat/preview',data);
    $('cvat-filename').textContent=pendingCvat.filename;
    $('cvat-summary').textContent=`${pendingCvat.mask_count} masks across ${pendingCvat.frame_count} frames`;
    $('cvat-labels').textContent=`Labels: ${pendingCvat.labels.join(', ')}`;
    const skipped=Object.entries(pendingCvat.skipped_shapes);
    $('cvat-skipped').textContent=skipped.length?`Not imported (not segmentation shapes): ${skipped.map(([type,count])=>`${count} ${type}`).join(', ')}.`:'';
    $('cvat-error').textContent='';$('cvat-dialog').showModal();status('CVAT preview ready. Review and import the masks.');
  });
};
$('cvat-cancel').onclick=()=>task(async()=>{
  if(pendingCvat)await api(`/api/import/cvat/${pendingCvat.token}?project=${project.id}`,null,'DELETE');
  pendingCvat=null;$('cvat-dialog').close();status('CVAT import cancelled.');
});
$('cvat-dialog').addEventListener('cancel',e=>{e.preventDefault();if(!busy)$('cvat-cancel').click();});
$('cvat-apply').onclick=()=>task(async()=>{
  if(!pendingCvat)return;
  status('Importing CVAT masks…');
  const result=await api('/api/import/cvat/apply',{project:project.id,token:pendingCvat.token});
  project=result.project;pendingCvat=null;$('cvat-dialog').close();resetDraft();
  await go(result.summary.first_frame);
  status(`Imported ${result.summary.mask_count} masks across ${result.summary.frame_count} frames. Select Edit mask to make corrections.`);
});
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
  await refreshMasks();updateProject();draw();status('Ready. Load a mask or paint a new object.');
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
  status(`Display zoom ${value.toFixed(0)}%. Source images and exported masks stay at ${image.width} × ${image.height}.`);
}
function setDraft(raw=null) {
  draftMask=document.createElement('canvas');draftMask.width=image.width;draftMask.height=image.height;
  const x=draftMask.getContext('2d');x.fillStyle='black';x.fillRect(0,0,image.width,image.height);
  if(raw)x.drawImage(raw,0,0);
  draft=true;retintDraft();$('score').textContent='Manual mask · save to commit changes.';
}
function finishSamRegion(){samPoints=[];samBase=null;samPending=false;draw();controls();}
function chooseTool(value){
  if(busy||stroke)return;
  if(value!==tool)finishSamRegion();tool=value;
  $('tool-sam').classList.toggle('selected',value==='sam');$('tool-brush').classList.toggle('selected',value==='brush');
  $('sam-controls').hidden=value!=='sam';setMode(positive);draw();controls();
  status(value==='sam'?'SAM2: include points inside one region, exclude points outside it.':'Brush editor: drag to paint; right drag erases.');
}
$('tool-sam').onclick=()=>chooseTool('sam');$('tool-brush').onclick=()=>chooseTool('brush');
function beginSamRegion(){
  if(samBase)return;
  if(!draftMask)setDraft();
  samBase=document.createElement('canvas');samBase.width=image.width;samBase.height=image.height;
  samBase.getContext('2d').drawImage(draftMask,0,0);samBaseDirty=dirty;samBasePoints=samUsedPoints.map(point=>({...point}));
}
async function predictSam(){
  if(!samPoints.some(point=>point.label===1)){samPending=true;draw();throw new Error('Add an include point inside the desired region first.');}
  samPending=true;status('SAM2 predicting… The first prediction loads the model.');draw();
  const result=await api('/api/predict',{project:project.id,frame,points:samPoints});
  const region=await loadImage('data:image/png;base64,'+result.mask);
  const combined=document.createElement('canvas');combined.width=image.width;combined.height=image.height;
  const x=combined.getContext('2d');x.drawImage(region,0,0);
  const pixels=x.getImageData(0,0,image.width,image.height), base=samBase.getContext('2d').getImageData(0,0,image.width,image.height);
  const operation=$('sam-operation').value;
  for(let i=0;i<pixels.data.length;i+=4){
    const original=base.data[i]>127,predicted=pixels.data[i]>127;
    const included=operation==='add'?(original||predicted):operation==='remove'?(original&&!predicted):predicted;
    pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=included?255:0;pixels.data[i+3]=255;
  }
  x.putImageData(pixels,0,0);checkpoint();setDraft(combined);
  samUsedPoints=samPoints.map(point=>({...point}));samPending=false;dirty=true;
  $('score').textContent=`SAM2 region score ${(result.score*100).toFixed(1)}% · ${result.pixels.toLocaleString()} predicted pixels (before combining)`;
  $('sam-status').textContent=`SAM2 ready on ${result.device}.`;
  status('SAM2 preview ready. Refine with points, add a new region, use the brush, or save.');
}
$('sam-retry').onclick=()=>task(predictSam);
$('sam-operation').onchange=()=>{if(samPoints.length)task(predictSam);};
$('sam-next').onclick=()=>{finishSamRegion();status('Click inside another region to combine it with this mask.');};
$('sam-undo').onclick=()=>{
  if(busy||!samPoints.length)return;samPoints.pop();
  if(samPoints.some(point=>point.label===1)){task(predictSam);return;}
  if(samBase){setDraft(samBase);dirty=samBaseDirty;samUsedPoints=samBasePoints.map(point=>({...point}));}
  samPending=samPoints.length>0;draw();controls();status(samPoints.length?'Add an include point first.':'SAM points undone. The mask from before this region is restored.');
};
function checkpoint(){
  if(!draftMask)setDraft();
  history.push(draftMask.getContext('2d').getImageData(0,0,image.width,image.height));
  if(history.length>20)history.shift();
}
function position(e){const r=canvas.getBoundingClientRect();return {x:Math.max(0,Math.min(image.width-1,(e.clientX-r.left)*image.width/r.width)),y:Math.max(0,Math.min(image.height-1,(e.clientY-r.top)*image.height/r.height))};}
function paint(p){
  const x=draftMask.getContext('2d');x.strokeStyle=x.fillStyle=stroke.erase?'black':'white';
  x.lineWidth=stroke.size;x.lineCap='round';x.lineJoin='round';
  x.beginPath();x.moveTo(stroke.last.x,stroke.last.y);x.lineTo(p.x,p.y);x.stroke();
  x.beginPath();x.arc(p.x,p.y,stroke.size/2,0,Math.PI*2);x.fill();
  stroke.last=p;dirty=true;retintDraft();
}
$('loadmask').onclick=()=>{if(busy||!image||!canDiscard())return;$('maskfile').value='';$('maskfile').click();};
$('maskfile').onchange=()=>{const file=$('maskfile').files[0];if(!file)return;task(async()=>{
  const data=new FormData();data.append('project',project.id);data.append('frame',frame);data.append('file',file);
  const result=await api('/api/load-mask',data);
  const raw=await loadImage('data:image/png;base64,'+result.mask);
  resetDraft();setDraft(raw);dirty=true;status(`Loaded ${file.name}. Edit the mask and save it with a label.`);
});};
function setMode(value){positive=value;$('positive').classList.toggle('selected',value);$('negative').classList.toggle('selected',!value);$('positive').textContent=tool==='sam'?'+ Include':'Brush';$('negative').textContent=tool==='sam'?'− Exclude':'Eraser';}
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
$('dental-labels').onclick=()=>task(async()=>{
  project=await api('/api/labels/dental-preset',{project:project.id});updateLabels();
  status('Five dental labels are available. Background is annotated explicitly; overlaps remain allowed.');
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
  if(busy||stroke||!image||!project||![0,2].includes(e.button))return;
  if(tool==='sam'){
    e.preventDefault();if(samPoints.length>=500){status('Maximum 500 SAM2 points per region. Start a new SAM region.',true);return;}
    beginSamRegion();samPoints.push({...position(e),label:e.button===2?0:(positive?1:0)});dirty=true;samPending=true;task(predictSam);return;
  }
  const size=Number($('brushsize').value);
  if(!Number.isFinite(size)||size<1||size>500){status('Brush diameter must be between 1 and 500 pixels.',true);return;}
  e.preventDefault();checkpoint();const p=position(e);
  stroke={pointer:e.pointerId,last:p,erase:e.button===2||!positive,size};
  canvas.setPointerCapture(e.pointerId);paint(p);controls();
});
canvas.addEventListener('pointermove',e=>{if(stroke?.pointer===e.pointerId)paint(position(e));});
function endStroke(e){if(stroke?.pointer!==e.pointerId)return;stroke=null;controls();}
canvas.addEventListener('pointerup',endStroke);canvas.addEventListener('pointercancel',endStroke);canvas.addEventListener('lostpointercapture',endStroke);
$('undo').onclick=()=>{if(!history.length||stroke)return;finishSamRegion();samUsedPoints=[];draftMask.getContext('2d').putImageData(history.pop(),0,0);dirty=true;retintDraft();controls();};
$('clear').onclick=()=>{if(!image||stroke)return;finishSamRegion();samUsedPoints=[];checkpoint();const x=draftMask.getContext('2d');x.fillStyle='black';x.fillRect(0,0,image.width,image.height);dirty=true;retintDraft();controls();};
$('new').onclick=()=>{if(canDiscard()){resetDraft();controls();status('Ready for a new object.');}};
function binaryMask(){
  const x=draftMask.getContext('2d'), pixels=x.getImageData(0,0,draftMask.width,draftMask.height);
  for(let i=0;i<pixels.data.length;i+=4){const value=pixels.data[i]>127?255:0;pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=value;pixels.data[i+3]=255;}
  const c=document.createElement('canvas');c.width=draftMask.width;c.height=draftMask.height;c.getContext('2d').putImageData(pixels,0,0);
  return c.toDataURL('image/png').split(',')[1];
}
async function saveCurrentObject(){
  if(samPending)throw new Error('Generate a valid SAM2 result or switch to Brush editor before saving.');
  const label=$('class').value.trim();if(!label)throw new Error('Enter a class name before saving.');
  const oid=$('object').value?Number($('object').value):null;
  if(oid!==null&&(!Number.isInteger(oid)||oid<1||oid>2147483647))throw new Error('Object ID must be a positive integer.');
  if((project.annotations[String(frame)]||[]).some(a=>a.object_id===oid)&&editing!==oid&&!confirm(`Replace object #${oid} on this frame?`))return false;
  project=await api('/api/save',{project:project.id,frame,mask:binaryMask(),class_name:label,object_id:oid,sam_points:samUsedPoints});
  resetDraft();await refreshMasks();updateProject();status('Object saved to disk. Add another object or move to the next frame.');
  return true;
}
$('save').onclick=()=>task(saveCurrentObject);
function navigate(index){if(busy||stroke||!canDiscard()){ $('timeline').value=frame;return; }task(()=>go(index));}
$('prev').onclick=()=>navigate(frame-1);$('next').onclick=()=>navigate(frame+1);
$('timeline').onchange=()=>navigate(Number($('timeline').value));$('overlays').onchange=draw;
$('class').onchange=()=>{if(draft)dirty=true;updateLabels();controls();};$('object').oninput=()=>{if(draft)dirty=true;};
$('export').onclick=()=>{if(dirty&&!confirm('Export saved masks only? The current preview has not been saved.'))return;window.location.href=`/api/export?project=${project.id}`;};
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
window.addEventListener('resize',applyDisplay);
window.addEventListener('keydown',e=>{if(busy||$('version-dialog').open||$('cvat-dialog').open||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;const ids={ArrowLeft:'prev',ArrowRight:'next',z:'undo',Z:'undo',Enter:'save'};if(ids[e.key]){e.preventDefault();$(ids[e.key]).click();}});
$('folder').value=localStorage.getItem('labeler-folder')||'';
renderPalette('new-palette','newcolor');renderPalette('edit-palette','editcolor');
controls();api('/api/status').then(s=>{$('device').textContent=s.device;$('sam-status').textContent=s.sam2?.message||'SAM2 status unavailable.';}).catch(e=>status(e.message,true));
