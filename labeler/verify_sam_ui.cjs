// Real SAM2 browser check: requires Playwright and a SAM2-enabled disposable server.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const os=require('node:os');
const path=require('node:path');
const crypto=require('node:crypto');

(async()=>{
  const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'sam-ui-'));
  const browser=await chromium.launch({headless:true,channel:process.env.LABELER_TEST_BROWSER || 'chrome'});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    page.setDefaultTimeout(60000);
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto(process.env.LABELER_TEST_URL || 'http://127.0.0.1:8012');
    const png=await page.evaluate(nonce=>{
      const c=document.createElement('canvas');c.width=320;c.height=240;const x=c.getContext('2d');
      x.fillStyle='#dddde8';x.fillRect(0,0,320,240);x.fillStyle='#c84040';x.beginPath();x.ellipse(85,120,50,70,0,0,Math.PI*2);x.fill();
      x.fillStyle='#358050';x.fillRect(200,60,85,125);x.fillStyle='#222';x.fillText(nonce,2,235);
      return c.toDataURL().split(',')[1];
    },crypto.randomUUID());
    await fs.writeFile(path.join(temporary,'frame.png'),Buffer.from(png,'base64'));
    await page.locator('#folderfiles').setInputFiles(temporary);await page.locator('#canvas').waitFor({state:'visible'});
    await page.waitForFunction(()=>!busy);
    await page.locator('#dental-labels').click();await page.waitForFunction(()=>!busy);
    assert.equal(await page.locator('#label-count').innerText(),'5');
    await page.locator('#class').selectOption('teeth_gum');
    await page.locator('#zoom').fill('50');await page.locator('#zoom').dispatchEvent('change');
    await page.locator('#tool-sam').click();
    const click=async(x,y,button='left')=>{
      const rect=await page.locator('#canvas').boundingBox();
      await page.mouse.click(rect.x+x/320*rect.width,rect.y+y/240*rect.height,{button});
      await page.waitForFunction(()=>!busy);
    };
    const pixel=async(x,y)=>page.evaluate(([x,y])=>draftMask.getContext('2d').getImageData(x,y,1,1).data[0],[x,y]);
    await click(10,10,'right');assert.equal(await page.locator('#save').isDisabled(),true);
    await page.locator('#sam-undo').click();
    await click(85,120);assert.equal(await pixel(85,120),255);
    await click(10,10,'right');assert.equal(await pixel(85,120),255);
    await page.locator('#sam-undo').click();await page.waitForFunction(()=>!busy);
    await page.locator('#sam-next').click();
    await click(240,120);assert.equal(await pixel(85,120),255);assert.equal(await pixel(240,120),255);
    await page.locator('#sam-operation').selectOption('remove');await page.waitForFunction(()=>!busy);
    assert.equal(await pixel(85,120),255);assert.equal(await pixel(240,120),0);
    await page.locator('#sam-operation').selectOption('replace');await page.waitForFunction(()=>!busy);
    assert.equal(await pixel(85,120),0);assert.equal(await pixel(240,120),255);
    await page.locator('#sam-undo').click();assert.equal(await pixel(85,120),255);
    await page.locator('#sam-operation').selectOption('add');
    await page.route('**/api/predict',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'Test unavailable'})}));
    await click(240,120);assert.equal(await page.locator('#save').isDisabled(),true);
    assert.equal(await pixel(85,120),255);
    await page.unroute('**/api/predict');await page.locator('#sam-retry').click();await page.waitForFunction(()=>!busy);
    assert.equal(await pixel(85,120),255);assert.equal(await pixel(240,120),255);
    await page.locator('#tool-brush').click();
    await click(10,10);assert.equal(await pixel(10,10),255);
    await page.locator('#undo').click();assert.equal(await pixel(10,10),0);
    await click(10,10);await page.locator('#save').click();await page.waitForFunction(()=>!busy && !dirty);
    await page.getByRole('button',{name:'Edit mask',exact:true}).click();await page.waitForFunction(()=>!busy);
    assert.equal(await pixel(10,10),255);assert.equal(await pixel(85,120),255);assert.equal(await pixel(240,120),255);
    assert.equal(await page.evaluate(()=>samPoints.length),0);
    await page.locator('#save-version').click();await page.locator('#version-name').fill('SAM2 regions');
    await page.locator('#version-submit').click();await page.waitForFunction(()=>!busy);
    assert.equal(await page.locator('#version-count').innerText(),'1');
    assert.deepEqual(errors,[]);
    console.log('PASS: real SAM2 prompts at 50% zoom, include/exclude, point undo, disconnected-region union, remove/replace, error/retry, brush/undo, save/reopen and versions; no browser errors');
  }finally{await browser.close();await fs.rm(temporary,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exit(1);});
