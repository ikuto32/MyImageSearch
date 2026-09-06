// Browser interaction fixtures for confirmation and image details. These tests
// exercise the shipped frontend, without production data or ZIP generation.
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const {startFixture,ready,setLocale,fixtureState,frames} = require('./review_responsive.cjs')

const ROOT = path.resolve(__dirname,'..')
const LONG_NAME = 'ExtremelyLongImageFilenameWithoutAnyPathSeparators'.repeat(7) + '画像名の全文を省略せず確認する'.repeat(6) + '.jpg'
const VIEWPORTS = [
  {width:320,height:812},{width:375,height:812},{width:600,height:900},{width:768,height:1024},{width:1440,height:900},
  {width:812,height:320},{width:812,height:375},{width:900,height:600},{width:1024,height:768},{width:1440,height:600},
]
const prepareCount = fixture => fixture.calls.filter(call=>call.path==='/downloads/prepare').length

async function buttonPositions(page) {
  return page.evaluate(()=>{
    const rect=selector=>{const r=document.querySelector(selector).getBoundingClientRect();return{x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom}}
    return {toggle:rect('#selectionModeToggle'),download:rect('.downloadAction'),width:innerWidth,height:innerHeight,documentWidth:document.documentElement.scrollWidth}
  })
}

function sameRect(actual, expected, label) {
  for(const key of ['x','y','width','height'])assert(Math.abs(actual[key]-expected[key])<=1,`${label}: ${key} shifted ${expected[key]} -> ${actual[key]}`)
}

async function reviewPositions(page,fixture) {
  await fixtureState(page,0,false)
  const baseline=await buttonPositions(page), states=[]
  assert.equal(baseline.documentWidth,baseline.width)
  assert(baseline.toggle.right<=baseline.download.x+1,'Selection toggle must stay to the left of Save')
  for(const count of [0,1,1024]){
    await fixtureState(page,count,true)
    const current=await buttonPositions(page)
    sameRect(current.download,baseline.download,`Save with ${count} selected`)
    sameRect(current.toggle,baseline.toggle,`Selection toggle with ${count} selected`)
    assert.equal(current.documentWidth,current.width)
    states.push({count,...current})
  }
  const before=prepareCount(fixture), point={x:baseline.toggle.x+baseline.toggle.width/2,y:baseline.toggle.y+baseline.toggle.height/2}
  await page.mouse.click(point.x,point.y)
  await frames(page)
  assert.equal(await page.locator('#selectionModeToggle').getAttribute('aria-pressed'),'false')
  sameRect((await buttonPositions(page)).download,baseline.download,'Save after ending selection')
  // Repeat at the same physical location: it must re-enter selection, and must
  // never open Save confirmation or prepare a ZIP because a button moved.
  await page.mouse.click(point.x,point.y)
  await frames(page)
  assert.equal(await page.locator('#selectionModeToggle').getAttribute('aria-pressed'),'true')
  assert.equal(await page.locator('#downloadConfirmDialog').isVisible(),false)
  assert.equal(prepareCount(fixture),before)
  await page.locator('#selectionModeToggle').click()
  return {baseline,states,repeatAtExitDoesNotDownload:true}
}

async function reviewDownloadConfirmation(page,fixture) {
  await fixtureState(page,0,false)
  const before=prepareCount(fixture)
  const open=async()=>{
    await page.locator('.downloadAction').click()
    await page.locator('#downloadConfirmDialog').waitFor({state:'visible'})
    await page.waitForFunction(()=>document.activeElement?.id==='downloadCancel')
    assert.equal(prepareCount(fixture),before,'Opening confirmation must not prepare a ZIP')
    const card=await page.locator('#downloadConfirmDialog').boundingBox()
    const viewport=page.viewportSize()
    assert(card.x>=-1&&card.x+card.width<=viewport.width+1,'Confirmation overflows horizontally')
    const cancel=await page.locator('#downloadCancel').boundingBox(),confirm=await page.locator('#downloadConfirm').boundingBox()
    assert(cancel.y>=0&&cancel.y+cancel.height<=viewport.height+1,'Cancel must be reachable')
    assert(confirm.y>=0&&confirm.y+confirm.height<=viewport.height+1,'Confirm must be reachable')
  }
  for(const action of ['cancel','escape','outside']){
    await open()
    if(action==='cancel')await page.locator('#downloadCancel').click()
    else if(action==='escape')await page.keyboard.press('Escape')
    else await page.mouse.click(1,1)
    await page.locator('#downloadConfirmDialog').waitFor({state:'hidden'})
    await frames(page)
    assert.equal(prepareCount(fixture),before,`${action} must not prepare a ZIP`)
    await page.waitForFunction(()=>document.activeElement?.classList.contains('downloadAction'))
  }
  await open()
  const nativeDownload=page.waitForEvent('download')
  await page.locator('#downloadConfirm').dblclick({delay:20})
  const downloaded=await nativeDownload
  assert.equal(await downloaded.failure(),null)
  await page.waitForFunction(()=>!document.querySelector('.downloadAction')?.classList.contains('v-btn--loading'))
  assert.equal(prepareCount(fixture),before+1,'A repeated confirmation must prepare exactly one ZIP')
  const request=fixture.calls.filter(call=>call.path==='/downloads/prepare').at(-1)
  assert.equal(request.params.first,1024,'No selection prepares the first 1024 matching catalog items')
  assert.equal(request.params.ids,undefined)
  await fixtureState(page,1,true)
  await page.locator('.downloadAction').click()
  await page.locator('#downloadConfirmDialog').waitFor({state:'visible'})
  assert.equal(prepareCount(fixture),before+1,'Selected-image Save also requires confirmation')
  const selectedDownload=page.waitForEvent('download')
  await page.locator('#downloadConfirm').click()
  assert.equal(await (await selectedDownload).failure(),null)
  assert.equal(prepareCount(fixture),before+2)
  assert.deepEqual(fixture.calls.filter(call=>call.path==='/downloads/prepare').at(-1).params.ids,['fixture-0'])
  await page.waitForFunction(()=>!document.querySelector('.downloadAction')?.classList.contains('v-btn--loading'))
  await fixtureState(page,0,false)
  return {openRequests:0,cancelRequests:0,escapeRequests:0,outsideRequests:0,repeatedConfirmRequests:1,firstCount:1024,selectedConfirmRequests:1,selectedCount:1}
}

async function detailGeometry(page) {
  return page.evaluate(()=>{
    const full=document.querySelector('#detailFileName'),close=document.querySelector('#closeImageDetail'),card=document.querySelector('.imageDialog')
    const b=full.getBoundingClientRect(),c=close.getBoundingClientRect(),d=card.getBoundingClientRect(),style=getComputedStyle(full)
    const range=document.createRange();range.selectNodeContents(full)
    return {width:innerWidth,height:innerHeight,documentWidth:document.documentElement.scrollWidth,card:{left:d.left,right:d.right,top:d.top,bottom:d.bottom,scrollWidth:card.scrollWidth,clientWidth:card.clientWidth},
      filename:{text:full.textContent,width:b.width,height:b.height,scrollWidth:full.scrollWidth,clientWidth:full.clientWidth,scrollHeight:full.scrollHeight,clientHeight:full.clientHeight,whiteSpace:style.whiteSpace,textOverflow:style.textOverflow,lineRects:range.getClientRects().length},
      close:{left:c.left,right:c.right,top:c.top,bottom:c.bottom}}
  })
}

async function reviewImageDetails(page,fixture,outputDir,label,closeWithEscape=false) {
  await fixtureState(page,0,false)
  // Give each layout scenario a fresh source, so decoded-image reuse cannot
  // bypass the deliberately injected network fault. Retry keeps this URL.
  await page.evaluate(label=>{
    const root=document.querySelector('#app'),vm=root.__vue_app__?._instance?.proxy||root._vnode?.component?.proxy
    const target=vm.displayItems.find(item=>!item.placeholder)
    target.img_original=target.img_original.split('?')[0]+`?fixtureCase=${encodeURIComponent(label)}`
  },label)
  await frames(page)
  // Keep the upstream unavailable until Retry. Browsers may issue more than
  // one request for a newly mounted image; a single one-shot 503 is flaky.
  fixture.original.delayMs=700;fixture.original.failures=Infinity
  const before=fixture.original.requests.length
  await page.locator('.imageSelectButton').first().click()
  await page.locator('.imageDialog').waitFor({state:'visible'})
  await page.locator('.detailImageLoading').waitFor({state:'visible'})
  assert.equal(await page.locator('.detailBody').getAttribute('data-image-state'),'loading')
  const loading=await detailGeometry(page)
  assert.equal(loading.filename.text,LONG_NAME,'The full filename must be available in the detail body')
  assert.notEqual(loading.filename.whiteSpace,'nowrap')
  assert.notEqual(loading.filename.textOverflow,'ellipsis')
  assert(loading.filename.lineRects>1,'A long separator-free filename must wrap')
  assert(loading.filename.scrollWidth<=loading.filename.clientWidth+1,'Filename needs horizontal scrolling')
  assert(loading.filename.scrollHeight<=loading.filename.clientHeight+1,'Filename text is vertically clipped')
  assert(loading.card.left>=-1&&loading.card.right<=loading.width+1,'Image dialog exceeds viewport width')
  assert(loading.close.top>=0&&loading.close.bottom<=loading.height+1,'Close is not reachable while loading')
  await page.locator('#retryDetailImage').waitFor({state:'visible'})
  assert.equal(await page.locator('.detailBody').getAttribute('data-image-state'),'error')
  await page.locator('.detailImageLoading').waitFor({state:'hidden'})
  const error=await detailGeometry(page)
  assert(fixture.original.requests.length>before)
  await page.screenshot({path:path.join(outputDir,`detail-error-${label}.png`)})
  const beforeRetry=fixture.original.requests.length
  fixture.original.failures=0
  await page.locator('#retryDetailImage').click()
  await page.locator('.detailImageLoading').waitFor({state:'visible'})
  await page.waitForFunction(()=>document.querySelector('.detailBody')?.dataset.imageState==='loaded'&&[...document.querySelectorAll('.imageDialog .v-img img')].some(img=>img.complete&&img.naturalWidth>0))
  await page.locator('.detailImageLoading').waitFor({state:'hidden'})
  assert.equal(await page.locator('#retryDetailImage').isVisible(),false)
  assert(fixture.original.requests.length>beforeRetry,'Retry must issue a fresh original-image request')
  const loaded=await detailGeometry(page)
  assert(loaded.card.scrollWidth<=loaded.card.clientWidth+1)
  // Users must be able to reach all filename lines with vertical scrolling,
  // while the dedicated close control remains available in the viewport.
  await page.locator('#detailFileName').scrollIntoViewIfNeeded()
  await page.locator('.imageDialog').evaluate(card=>{for(const el of [card,...card.querySelectorAll('*')])if(el.scrollHeight>el.clientHeight+1&&/auto|scroll/.test(getComputedStyle(el).overflowY))el.scrollTop=el.scrollHeight})
  const scrolled=await detailGeometry(page)
  assert(scrolled.close.top>=0&&scrolled.close.bottom<=scrolled.height+1,'Close left the viewport while reading the full filename')
  await page.screenshot({path:path.join(outputDir,`detail-loaded-${label}.png`)})
  if(closeWithEscape)await page.keyboard.press('Escape')
  else await page.locator('#closeImageDetail').click()
  await page.locator('.imageDialog').waitFor({state:'hidden'})
  await page.waitForFunction(()=>document.activeElement?.classList.contains('imageSelectButton'))
  return {loading,error,loaded,scrolled,originalRequests:fixture.original.requests.slice(before),failedThenRetried:true,closeAndFocus:true}
}

async function reviewEscapeRegression(page,fixture,outputDir){
  await page.setViewportSize({width:667,height:375});await ready(page)
  await setLocale(page,'en')
  const first=await reviewImageDetails(page,fixture,outputDir,'escape-667x375-en',true)
  const reopened=[]
  for(let attempt=0;attempt<3;attempt++){
    await page.waitForFunction(()=>[...document.querySelectorAll('.v-overlay')].every(el=>{
      const r=el.getBoundingClientRect(),s=getComputedStyle(el)
      return r.height===0||s.visibility==='hidden'||s.display==='none'
    }))
    await page.evaluate(attempt=>{
      const root=document.querySelector('#app'),vm=root.__vue_app__?._instance?.proxy||root._vnode?.component?.proxy
      const item=vm.displayItems.find(item=>!item.placeholder)
      item.img_original=item.img_original.split('?')[0]+`?escapeRegression=${attempt}`
    },attempt)
    fixture.original.delayMs=1800;fixture.original.failures=0
    await frames(page)
    const point=await page.locator('.imageSelectButton').first().evaluate(el=>{
      const r=el.getBoundingClientRect(),area=document.querySelector('#scroll-target').getBoundingClientRect()
      return{x:(r.left+r.right)/2,y:(Math.max(r.top,area.top)+Math.min(r.bottom,area.bottom))/2}
    })
    // A direct pointer click avoids scrolling the complete card into the short
    // landscape viewport and preserves the real dialog-opening timing.
    await page.mouse.click(point.x,point.y)
    await page.locator('.detailImageLoading').waitFor({state:'visible'})
    const focusBefore=await page.evaluate(()=>({id:document.activeElement?.id,className:document.activeElement?.className}))
    await page.keyboard.press('Escape')
    await page.locator('.imageDialog').waitFor({state:'hidden'})
    await page.waitForFunction(()=>document.activeElement?.classList.contains('imageSelectButton'))
    assert.equal(await page.locator('#selectionModeToggle').getAttribute('aria-pressed'),'false','Esc must not reach background selection handling')
    reopened.push({attempt,focusBefore,closed:true,selectionMode:false})
  }
  assert(reopened.some(item=>String(item.focusBefore.className).includes('imageSelectButton')),'Exercise Esc before focus moves into the newly opened dialog')
  const before=prepareCount(fixture)
  await page.locator('.downloadAction').click()
  await page.locator('#downloadConfirmDialog').waitFor({state:'visible'})
  await page.keyboard.press('Escape')
  await page.locator('#downloadConfirmDialog').waitFor({state:'hidden'})
  assert.equal(prepareCount(fixture),before)
  await page.locator('#settingsToggle').click()
  await page.locator('#settingsPanel').waitFor({state:'visible'})
  await page.getByRole('combobox',{name:'Image categories',exact:true}).locator('input').press('ArrowDown')
  await page.getByRole('listbox').waitFor({state:'visible'})
  await page.keyboard.press('Escape')
  await page.getByRole('listbox').waitFor({state:'hidden'})
  assert.equal(await page.locator('#settingsPanel').isVisible(),true,'The first Esc closes only the settings popup')
  await page.keyboard.press('Escape')
  await page.locator('#settingsPanel').waitFor({state:'hidden'})
  await page.waitForFunction(()=>document.activeElement?.id==='settingsToggle')
  return {viewport:{width:667,height:375},locale:'en',first,reopened,confirmationEscRequests:0,settingsPopupThenMenuClosed:true}
}

async function runDialogs(browser,outputDir){
  fs.mkdirSync(outputDir,{recursive:true})
  const fixture=await startFixture({imageName:LONG_NAME,downloadDelayMs:350})
  const page=await browser.newPage({viewport:{width:1440,height:900},acceptDownloads:true})
  const report={kind:'deterministic frontend fixture',productionPerformance:false,viewports:VIEWPORTS,locales:['ja','en'],cases:[],failures:[],errors:[]}
  page.on('pageerror',error=>report.errors.push(error.message))
  const write=()=>fs.writeFileSync(path.join(outputDir,'dialog-review.json'),JSON.stringify(report,null,2))
  try{
    await page.goto(fixture.url,{waitUntil:'networkidle'});await ready(page)
    for(const viewport of VIEWPORTS){
      await page.setViewportSize(viewport);await ready(page)
      for(const locale of ['ja','en']){
        const label=`${viewport.width}x${viewport.height}-${locale}`
        await setLocale(page,locale)
        const result={...viewport,locale,passed:true}
        try{
          result.positions=await reviewPositions(page,fixture)
          result.confirmation=await reviewDownloadConfirmation(page,fixture)
          result.details=await reviewImageDetails(page,fixture,outputDir,label)
        }catch(error){
          result.passed=false;result.error=error.message;report.failures.push({label,error:error.message})
          await page.screenshot({path:path.join(outputDir,`failure-${label}.png`)})
          await page.keyboard.press('Escape');await page.keyboard.press('Escape')
        }
        report.cases.push(result);write()
        console.log(JSON.stringify({label,passed:result.passed,error:result.error}))
      }
    }
    try{report.escapeRegression=await reviewEscapeRegression(page,fixture,outputDir)}
    catch(error){report.failures.push({label:'escape-regression',error:error.message});await page.screenshot({path:path.join(outputDir,'failure-escape-regression.png')})}
    report.generatedAt=new Date().toISOString();report.prepareRequests=prepareCount(fixture);write()
    assert.deepEqual(report.errors,[])
    assert.equal(report.failures.length,0,`${report.failures.length} dialog/action checks failed`)
    return report
  }finally{write();await page.close();await fixture.close()}
}

module.exports={runDialogs,LONG_NAME,VIEWPORTS}
if(require.main===module){
  ;(async()=>{
    const{chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright')
    const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})})
    try{const report=await runDialogs(browser,path.join(ROOT,'.cache/dialog-review'));console.log(JSON.stringify({cases:report.cases.length,failures:report.failures.length,prepareRequests:report.prepareRequests}))}
    finally{await browser.close()}
  })().catch(error=>{console.error(error);process.exitCode=1})
}
