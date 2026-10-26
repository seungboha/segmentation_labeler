// Optional end-to-end CVAT check. Requires Playwright and a disposable test server.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');

(async()=>{
  const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'cvat-ui-'));
  const browser=await chromium.launch({headless:true,channel:process.env.LABELER_TEST_BROWSER || 'chrome'});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    page.setDefaultTimeout(15000);
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto(process.env.LABELER_TEST_URL || 'http://127.0.0.1:8012');
    const png=await page.evaluate(nonce=>{
      const c=document.createElement('canvas');c.width=c.height=128;
      const x=c.getContext('2d');x.fillStyle='#444';x.fillRect(0,0,128,128);
      x.fillStyle='white';x.fillText(nonce,5,20);return c.toDataURL().split(',')[1];
    },crypto.randomUUID());
    const frames=path.join(temporary,'frames');await fs.mkdir(frames);
    for(const name of ['frame2.png','frame10.png'])await fs.writeFile(path.join(frames,name),Buffer.from(png,'base64'));
    await page.locator('#folderfiles').setInputFiles(frames);
    await page.locator('#canvas').waitFor({state:'visible'});await page.waitForFunction(()=>!busy);
    const xml=`<annotations><version>1.1</version><meta><task><labels><label><name>instrument</name><color>#82b6ff</color></label></labels></task></meta>
      <image id="99" name="frame2.png" width="128" height="128"><mask label="instrument" rle="0,400" left="50" top="50" width="20" height="20"/><box label="ignored"/></image>
      <image id="0" name="frame10.png" width="128" height="128"><polygon label="instrument" points="30,30;50,30;50,50;30,50"/></image></annotations>`;
    const upload=async content=>{
      await page.locator('#cvat-file').setInputFiles({name:'annotations.xml',mimeType:'application/xml',buffer:Buffer.from(content)});
      await page.waitForFunction(()=>!busy);
    };
    await upload('bad xml');assert.match(await page.locator('#status').innerText(),/Cannot read XML/);
    assert.equal(await page.locator('#count').innerText(),'0');
    await upload(xml);
    assert.equal(await page.locator('#cvat-dialog').isVisible(),true);
    assert.match(await page.locator('#cvat-summary').innerText(),/2 masks across 2 frames/);
    assert.match(await page.locator('#cvat-skipped').innerText(),/1 box/);
    assert.equal(await page.locator('#count').innerText(),'0');
    await page.locator('#cvat-cancel').click();await page.waitForFunction(()=>!busy);
    assert.equal(await page.locator('#cvat-dialog').isVisible(),false);
    assert.equal(await page.locator('#count').innerText(),'0');
    await upload(xml);await page.locator('#cvat-apply').click();await page.waitForFunction(()=>!busy);
    assert.equal(await page.locator('#count').innerText(),'1');
    assert.match(await page.locator('#objects').innerText(),/instrument/);
    assert.equal(await page.locator('#class option[value="instrument"]').count(),1);
    await page.getByRole('button',{name:'Edit mask',exact:true}).click();await page.waitForFunction(()=>!busy);
    const center=()=>page.evaluate(()=>draftMask.getContext('2d').getImageData(60,60,1,1).data[0]);
    assert.equal(await center(),255);
    const rect=await page.locator('#canvas').boundingBox();
    await page.locator('#negative').click();
    await page.mouse.click(rect.x+60/128*rect.width,rect.y+60/128*rect.height);
    assert.equal(await center(),0);
    await page.locator('#save').click();await page.waitForFunction(()=>!busy && !dirty);
    await page.getByRole('button',{name:'Edit mask',exact:true}).click();await page.waitForFunction(()=>!busy);
    assert.equal(await center(),0);
    await page.locator('#next').click();await page.waitForFunction(()=>!busy);
    assert.match(await page.locator('#filename').innerText(),/frame10/);
    assert.equal(await page.locator('#count').innerText(),'1');
    await page.locator('#save-version').click();await page.locator('#version-name').fill('From CVAT');
    await page.locator('#version-submit').click();await page.waitForFunction(()=>!busy);
    assert.equal(await page.locator('#version-count').innerText(),'1');
    const downloaded=page.waitForEvent('download');
    await page.getByRole('button',{name:'Export version 1: From CVAT',exact:true}).click();
    const download=await downloaded;assert.equal(await download.failure(),null);
    assert.deepEqual(errors,[]);
    console.log('PASS: invalid XML feedback, import preview/cancel, masks across matched frames, editable saved pixels, versions and downloads; no browser errors');
  } finally {
    await browser.close();await fs.rm(temporary,{recursive:true,force:true});
  }
})().catch(error=>{console.error(error);process.exit(1);});
