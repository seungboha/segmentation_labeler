// Optional browser check: requires Playwright and a running, disposable test server.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');

(async()=>{
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'mask-versions-ui-'));
  const browser = await chromium.launch({headless:true, channel:process.env.LABELER_TEST_BROWSER || 'chrome'});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    page.setDefaultTimeout(15000);
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto(process.env.LABELER_TEST_URL || 'http://127.0.0.1:8012');
    const png = await page.evaluate(nonce=>{
      const surface=document.createElement('canvas');surface.width=surface.height=128;
      const x=surface.getContext('2d');x.fillStyle='#303030';x.fillRect(0,0,128,128);
      x.fillStyle='white';x.fillText(nonce,5,20);return surface.toDataURL().split(',')[1];
    },crypto.randomUUID());
    await fs.writeFile(path.join(temporary,'frame.png'),Buffer.from(png,'base64'));
    await page.locator('#folderfiles').setInputFiles(temporary);
    await page.locator('#canvas').waitFor({state:'visible'});
    await page.waitForFunction(()=>!busy);
    await page.locator('#newlabel').fill('instrument');await page.locator('#addlabel').click();
    await page.waitForFunction(()=>!busy);
    const paint = async()=>{
      const box=await page.locator('#canvas').boundingBox();
      await page.mouse.click(box.x+box.width/2,box.y+box.height/2);
    };
    await paint();await page.locator('#save-version').click();
    assert.equal(await page.locator('#version-unsaved').isVisible(),true);
    await page.locator('#version-cancel').click();
    assert.equal(await page.evaluate(()=>dirty),true);
    await page.locator('#save-version').click();await page.locator('#version-discard').click();
    assert.equal(await page.evaluate(()=>dirty),false);
    await page.locator('#version-cancel').click();

    await paint();await page.locator('#save-version').click();
    // Validation failure stays in the dialog and preserves the edits.
    await page.evaluate(()=>$('class').value='');
    await page.locator('#version-save-object').click();
    await page.waitForFunction(()=>!busy);
    assert.match(await page.locator('#version-error').innerText(),/class name/);
    assert.equal(await page.evaluate(()=>dirty),true);
    await page.evaluate(()=>$('class').value='instrument');
    await page.locator('#version-save-object').click();
    await page.waitForFunction(()=>!busy && !dirty);
    await page.locator('#version-name').fill('Wide');await page.locator('#version-note').fill('Initial mask set');
    await page.locator('#version-submit').click();
    await page.waitForFunction(()=>!busy && !$('version-dialog').open);
    assert.equal(await page.locator('#version-count').innerText(),'1');
    assert.match(await page.locator('#version-list').innerText(),/Wide/);

    await page.getByRole('button',{name:'Edit mask',exact:true}).click();await page.waitForFunction(()=>!busy);
    await page.locator('#brushsize').fill('6');await page.locator('#negative').click();await paint();
    await page.locator('#save-version').click();await page.locator('#version-save-object').click();
    await page.waitForFunction(()=>!busy && !dirty);
    await page.locator('#version-name').fill('Narrow');await page.locator('#version-submit').click();
    await page.waitForFunction(()=>!busy && !$('version-dialog').open);
    assert.equal(await page.locator('#version-count').innerText(),'2');

    const downloadPromise=page.waitForEvent('download');
    await page.getByRole('button',{name:'Export version 1: Wide',exact:true}).click();
    const download=await downloadPromise;
    const exportFile=path.join(temporary,'wide.zip');await download.saveAs(exportFile);
    assert.equal((await fs.readFile(exportFile)).subarray(0,2).toString(),'PK');
    assert.equal(await page.evaluate(()=>dirty),false);
    // Reopen through the ordinary frame folder workflow to verify persisted version listing.
    const folder=await page.evaluate(()=>project.source_folder);
    await page.getByText('Or open by folder path',{exact:true}).click();
    await page.locator('#folder').fill(folder);await page.locator('#open').click();
    await page.waitForFunction(()=>!busy);
    assert.equal(await page.locator('#version-count').innerText(),'2');
    assert.match(await page.locator('#version-list').innerText(),/Narrow/);
    assert.deepEqual(errors,[]);
    console.log('PASS: version dialog, cancel/discard/save choices, validation errors, named whole-set versions, download, persisted listing; no browser errors');
  } finally {
    await browser.close();await fs.rm(temporary,{recursive:true,force:true});
  }
})().catch(error=>{console.error(error);process.exit(1);});
