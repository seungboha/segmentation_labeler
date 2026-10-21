/* Local raster editing. No model requests are made by this class. */
class MaskBrush {
  constructor() {
    this.surface=document.createElement('canvas');this.overlay=document.createElement('canvas');
    this.context=this.surface.getContext('2d',{willReadFrequently:true});this.reset();
  }
  reset(){this.active=false;this.modified=false;this.undoStack=[];this.redoStack=[];this.before=null;this.previous=null;}
  blank(width,height){this.reset();this.surface.width=width;this.surface.height=height;this.overlay.width=width;this.overlay.height=height;this.active=true;}
  load(raw){
    this.blank(raw.width,raw.height);this.context.drawImage(raw,0,0);
    const pixels=this.context.getImageData(0,0,raw.width,raw.height);
    for(let i=0;i<pixels.data.length;i+=4){const a=pixels.data[i]>127&&pixels.data[i+3]>127?255:0;pixels.data[i]=255;pixels.data[i+1]=255;pixels.data[i+2]=255;pixels.data[i+3]=a;}
    this.context.putImageData(pixels,0,0);
  }
  colored(color){const x=this.overlay.getContext('2d');x.clearRect(0,0,this.overlay.width,this.overlay.height);x.globalCompositeOperation='source-over';x.drawImage(this.surface,0,0);x.globalCompositeOperation='source-in';x.fillStyle=color;x.fillRect(0,0,this.overlay.width,this.overlay.height);x.globalCompositeOperation='source-over';return this.overlay;}
  snapshot(){return this.context.getImageData(0,0,this.surface.width,this.surface.height);}
  limitHistory(){const limit=Math.max(1,Math.min(30,Math.floor(64*1024*1024/(this.surface.width*this.surface.height*4))));while(this.undoStack.length+this.redoStack.length>limit){if(this.undoStack.length)this.undoStack.shift();else this.redoStack.shift();}}
  begin(point,erase,size){this.before=this.snapshot();this.previous=point;this.erase=erase;this.size=size;this.stroke(point);}
  stroke(point){
    if(!this.before)return;
    const x=this.context,from=this.previous,r=this.size/2;
    x.globalCompositeOperation=this.erase?'destination-out':'source-over';x.fillStyle='#ffffff';x.strokeStyle='#ffffff';
    x.lineWidth=this.size;x.lineCap='round';x.lineJoin='round';x.beginPath();x.moveTo(from.x,from.y);x.lineTo(point.x,point.y);x.stroke();x.beginPath();x.arc(point.x,point.y,r,0,Math.PI*2);x.fill();
    x.globalCompositeOperation='source-over';this.previous=point;
  }
  end(){
    if(!this.before)return false;
    const after=this.snapshot();let changed=false;
    for(let i=3;i<after.data.length;i+=4){if(after.data[i]!==this.before.data[i]){changed=true;break;}}
    if(changed){this.undoStack.push(this.before);this.redoStack=[];this.modified=true;this.limitHistory();}
    this.before=null;this.previous=null;return changed;
  }
  undo(){if(!this.undoStack.length||this.before)return false;this.redoStack.push(this.snapshot());this.context.putImageData(this.undoStack.pop(),0,0);this.modified=true;return true;}
  redo(){if(!this.redoStack.length||this.before)return false;this.undoStack.push(this.snapshot());this.context.putImageData(this.redoStack.pop(),0,0);this.modified=true;return true;}
  png(){
    const pixels=this.snapshot();for(let i=0;i<pixels.data.length;i+=4){const v=pixels.data[i+3]>=128?255:0;pixels.data[i]=v;pixels.data[i+1]=v;pixels.data[i+2]=v;pixels.data[i+3]=255;}
    const output=document.createElement('canvas');output.width=this.surface.width;output.height=this.surface.height;output.getContext('2d').putImageData(pixels,0,0);return output.toDataURL('image/png').split(',')[1];
  }
}
window.MaskBrush=MaskBrush;
