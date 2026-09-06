// Real browser input checks. --url runs the same read-only checks on a live catalog.
const assert = require('node:assert/strict')
const {startFixture,ready,frames} = require('./review_responsive.cjs')

const VIEWPORTS = [{width:390,height:844},{width:768,height:1024},{width:1440,height:900}]
const sleep = ms => new Promise(resolve => setTimeout(resolve,ms))

// Observe rendered rows, including loading placeholders; no Vue instance access.
async function position(page) {
  return page.evaluate(() => {
    const area=document.querySelector('#scroll-target'), grid=document.querySelector('#itemArea')
    const cards=[...grid.querySelectorAll('[data-logical-index]')], first=cards[0]
    const columns=getComputedStyle(grid).gridTemplateColumns.split(' ').length
    const style=getComputedStyle(first), rect=first.getBoundingClientRect()
    const rowHeight=rect.height+parseFloat(style.marginTop)+parseFloat(style.marginBottom)
    const logical=Math.floor(Number(first.dataset.logicalIndex)/columns)*rowHeight-rect.top+area.getBoundingClientRect().top+parseFloat(style.marginTop)
    return {logical,scrollTop:area.scrollTop,scrollHeight:area.scrollHeight,clientHeight:area.clientHeight,
      documentHeight:document.documentElement.scrollHeight,viewportHeight:innerHeight,
      documentWidth:document.documentElement.scrollWidth,viewportWidth:innerWidth,
      first:Number(first.dataset.logicalIndex),last:Number(cards.at(-1).dataset.logicalIndex),
      endVisible:(()=>{const r=document.querySelector('.resultsEnd')?.getBoundingClientRect();return !!r&&r.top>=0&&r.bottom<=innerHeight})()}
  })
}

async function resetPosition(page) {
  await page.locator('#scroll-target').focus()
  await page.keyboard.press('Home')
  // A native scrollbar seek avoids the initial toolbar crossing affecting deltas.
  await page.locator('#scroll-target').evaluate(element => {element.scrollTop=1000})
  await ready(page)
  return position(page)
}

async function touch(cdp,type,points) {
  await cdp.send('Input.dispatchTouchEvent',{type,touchPoints:points.map((point,id)=>({id,radiusX:4,radiusY:4,force:1,...point}))})
}

async function swipe(page,cdp,fingers,holdMs=0) {
  const before=await resetPosition(page), {width,height}=page.viewportSize()
  const points=Array.from({length:fingers},(_,i)=>({x:width/2+(i-(fingers-1)/2)*80,y:height-130}))
  await page.evaluate(()=>{
    const sample=window.__catalogScrollReview={active:true,last:0,gaps:[],maxCards:0}
    const tick=time=>{if(!sample.active)return;if(sample.last)sample.gaps.push(time-sample.last);sample.last=time
      sample.maxCards=Math.max(sample.maxCards,document.querySelectorAll('#itemArea .itemBlock').length);requestAnimationFrame(tick)}
    requestAnimationFrame(tick)
  })
  await touch(cdp,'touchStart',points)
  if(holdMs)await sleep(holdMs)
  for(let step=1;step<=12;step++) {
    await sleep(12)
    await touch(cdp,'touchMove',points.map(point=>({...point,y:point.y-step*20})))
  }
  await frames(page)
  const held=await position(page)
  await touch(cdp,'touchEnd',[])
  const coast=[]
  for(const delay of [80,160,260,500,500]) {
    await sleep(delay);coast.push(await position(page))
  }
  const performance=await page.evaluate(()=>{
    const sample=window.__catalogScrollReview;sample.active=false;delete window.__catalogScrollReview
    const gaps=sample.gaps.sort((a,b)=>a-b)
    return {maxCards:sample.maxCards,frameMedianMs:gaps[Math.floor(gaps.length/2)],frameP95Ms:gaps[Math.floor(gaps.length*.95)],frameMaxMs:gaps.at(-1)}
  })
  const drag=held.logical-before.logical
  assert(drag>180&&drag<300,`${fingers} fingers: 240px drag moved ${drag.toFixed(1)}px`)
  assert(coast[1].logical-held.logical>40,`${fingers} fingers: releasing a flick must keep scrolling (${coast.map(value=>(value.logical-held.logical).toFixed(1)).join(', ')}px)`)
  assert(coast.every((value,i)=>value.logical>=(i?coast[i-1]:held).logical-1),`${fingers} fingers: coast reversed direction`)
  const earlySpeed=(coast[1].logical-coast[0].logical)/160
  const lateSpeed=(coast[3].logical-coast[2].logical)/500
  assert(earlySpeed>lateSpeed+0.02,`${fingers} fingers: coast should slow down`)
  assert(Math.abs(coast[4].logical-coast[3].logical)<3,`${fingers} fingers: coast should settle`)
  assert.equal(await page.locator('.imageDialog').isVisible(),false,'Dragging must not open an image')
  assert(coast.every(value=>value.scrollHeight===before.scrollHeight),'Appending rows changed the scrollbar extent')
  return {fingers,holdMs,drag,coast:coast.map(value=>value.logical-held.logical),earlySpeed,lateSpeed,scrollHeight:before.scrollHeight,performance}
}

async function wheelChecks(page) {
  const results=[]
  // Trackpads already emit decaying momentum deltas: consume them exactly once.
  for(const horizontalRatio of [0,1,3]) {
    const before=await resetPosition(page)
    await page.mouse.move(page.viewportSize().width/2,page.viewportSize().height/2)
    const deltas=[120,80,40,20,8]
    for(const delta of deltas) {await page.mouse.wheel(horizontalRatio*delta,delta);await sleep(25)}
    await frames(page)
    const after=await position(page),distance=after.logical-before.logical
    assert(Math.abs(distance-deltas.reduce((a,b)=>a+b,0))<4,`Wheel dx/dy=${horizontalRatio}: unexpected multiplier (${distance}px)`)
    await sleep(350)
    assert(Math.abs((await position(page)).logical-after.logical)<2,'Wheel input must not add a second momentum animation')
    results.push({horizontalRatio,distance})
  }
  return results
}

async function gestureStability(page,cdp) {
  const cases=[]
  for(const name of ['spacing-wobble','diagonal-wobble','one-two-one','late-second-finger','repeated-pans']) {
    const result={name,passed:true,movements:[]}
    let active=false
    try {
      await cdp.send('Emulation.setPageScaleFactor',{pageScaleFactor:1})
      await resetPosition(page)
      await page.evaluate(()=>{
        window.__catalogGestureCancels=0
        window.__catalogGestureCancelListener=()=>window.__catalogGestureCancels++
        document.querySelector('#scroll-target').addEventListener('pointercancel',window.__catalogGestureCancelListener)
      })
      const {width,height}=page.viewportSize(),changing=name.includes('finger')||name==='one-two-one'
      for(let cycle=0;cycle<(name==='repeated-pans'?3:1);cycle++) {
        const direction=cycle===1?-1:1, startY=direction>0?height-130:height-450
        let second=!changing,previousPoints=[{id:0,x:width/2-80,y:startY},...(second?[{id:1,x:width/2+80,y:startY}]:[])]
        await touch(cdp,'touchStart',previousPoints)
        active=true
        const before=await position(page),steps=[],physicalSteps=[]
        let previous=before.logical,previousPhysical=before.scrollTop
        for(let step=1;step<=16;step++) {
          await sleep(15)
          const addAt=name==='one-two-one'?4:10
          if(changing&&step===addAt) {
            second=true
            const first=previousPoints[0]
            previousPoints=[first,{id:1,x:first.x+160,y:first.y+40}]
            await touch(cdp,'touchStart',previousPoints)
          }
          if(name==='one-two-one'&&step===11) {
            await touch(cdp,'touchEnd',[previousPoints[1]])
            second=false
          }
          const wobble=name.includes('wobble')?[0,0,5,10,6,-5,-10,-6][step%8]:0
          const drift=name==='diagonal-wobble'?96*Math.sin(step*Math.PI/8):0,y=startY-direction*step*15
          previousPoints=[{id:0,x:width/2-80+drift-wobble,y},...(second?[{id:1,x:width/2+80+drift+wobble,y:y+(changing?40:0)}]:[])]
          await touch(cdp,'touchMove',previousPoints);await frames(page)
          const current=await position(page)
          steps.push(current.logical-previous);previous=current.logical
          physicalSteps.push(current.scrollTop-previousPhysical);previousPhysical=current.scrollTop
        }
        const distance=previous-before.logical,scale=await page.evaluate(()=>visualViewport.scale)
        result.movements.push({distance,maxStep:Math.max(...steps.map(Math.abs)),maxPhysicalStep:Math.max(...physicalSteps.map(Math.abs)),scale})
        await touch(cdp,'touchEnd',[])
        active=false
        assert(Math.abs(distance-direction*240)<35,`${name}: 240px pan moved ${distance.toFixed(1)}px`)
        assert(Math.max(...steps.map(Math.abs))<45,`${name}: a small move jumped through catalog rows`)
        assert(Math.abs(scale-1)<0.02,`${name}: ordinary finger drift became zoom (${scale})`)
        await sleep(80)
      }
      result.pointerCancels=await page.evaluate(()=>window.__catalogGestureCancels)
      assert.equal(result.pointerCancels,0,`${name}: the browser unexpectedly took over a pan`)
    } catch(error) {result.passed=false;result.error=error.message;if(active)await touch(cdp,'touchCancel',[])}
    finally {await page.evaluate(()=>{
      document.querySelector('#scroll-target').removeEventListener('pointercancel',window.__catalogGestureCancelListener)
      delete window.__catalogGestureCancels;delete window.__catalogGestureCancelListener
    })}
    cases.push(result)
  }
  return cases
}

async function visibleImagePoint(page) {
  return page.locator('#scroll-target').evaluate(area=>{
    const box=area.getBoundingClientRect()
    const button=[...area.querySelectorAll('.imageSelectButton')].find(element=>{const r=element.getBoundingClientRect();return r.top>box.top+15&&r.bottom<box.bottom-15})
    if(!button)throw new Error('No fully visible image for tap')
    const r=button.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}
  })
}

async function selectionChecks(page,cdp) {
  await page.locator('#scroll-target').focus();await page.keyboard.press('Home');await ready(page)
  await page.locator('#selectionModeToggle').click()
  const drags=[]
  const selectedCount=async()=>Number((await page.locator('#selectionCount').textContent()).replace(/[^0-9]/g,''))
  for(const holdMs of [0,600]) {
    drags.push(await swipe(page,cdp,1,holdMs))
    assert.equal(await selectedCount(),0,'A pan in selection mode must not check an image')
  }
  const point=await visibleImagePoint(page)
  await touch(cdp,'touchStart',[point]);await touch(cdp,'touchEnd',[]);await frames(page)
  assert.equal(await selectedCount(),1,'A normal tap must still check exactly one image')
  await page.locator('#settingsToggle').click();await page.locator('#scrollToTop').click()
  await page.locator('#settingsPanel').waitFor({state:'hidden'});await ready(page)
  assert.equal(await page.locator('#scroll-target').evaluate(element=>element.scrollTop),0,'The menu action must return to the top')
  assert.equal(await selectedCount(),1,'Returning to the top must preserve selection')
  await page.locator('#selectionModeToggle').click()
  return {drags,tapSelectedCount:1}
}

async function tapAndPinch(page,cdp) {
  await resetPosition(page)
  const point=await visibleImagePoint(page)
  await touch(cdp,'touchStart',[point]);await touch(cdp,'touchEnd',[])
  await page.locator('.imageDialog').waitFor({state:'visible'})
  await page.keyboard.press('Escape');await page.locator('.imageDialog').waitFor({state:'hidden'})
  const before=await page.evaluate(()=>visualViewport.scale)
  const center=page.viewportSize().width/2
  let y=point.y
  await touch(cdp,'touchStart',[{x:center-40,y},{x:center+40,y}])
  for(let step=1;step<=4;step++) {
    y-=12;await sleep(20)
    await touch(cdp,'touchMove',[{x:center-40,y},{x:center+40,y}])
  }
  // Changing a two-finger pan into a pinch must still yield to browser zoom.
  for(let step=1;step<=8;step++) {
    await sleep(20)
    await touch(cdp,'touchMove',[{x:center-40-step*8,y},{x:center+40+step*8,y}])
  }
  await touch(cdp,'touchEnd',[]);await sleep(150)
  const after=await page.evaluate(()=>visualViewport.scale)
  assert(after>before+0.1,`Browser pinch zoom was prevented (${before} -> ${after})`)
  assert.equal(await page.locator('.imageDialog').isVisible(),false,'Pinch must not open an image')
  return {tapOpensImage:true,pinchAfterPan:true,pinchScaleBefore:before,pinchScaleAfter:after}
}

async function endChecks(page,total) {
  const before=await resetPosition(page)
  let scrollbarDrag
  if(page.viewportSize().width>=1000) {
    const bar=await page.locator('#scroll-target').evaluate(element=>{
      const rect=element.getBoundingClientRect(),gutter=element.offsetWidth-element.clientWidth
      return {x:rect.right-gutter/2,y:rect.top,height:element.clientHeight,gutter}
    })
    // Native themes differ in whether they put an arrow above the thumb.
    for(const offset of [0.5,1.5]) {
      await page.locator('#scroll-target').focus();await page.keyboard.press('Home');await frames(page)
      await page.mouse.move(bar.x,bar.y+bar.gutter*offset);await page.mouse.down()
      await page.mouse.move(bar.x,bar.y+bar.height*0.6,{steps:12});await page.mouse.up();await ready(page)
      scrollbarDrag=await position(page)
      if(scrollbarDrag.scrollTop>before.scrollHeight*0.2)break
    }
    assert(scrollbarDrag.scrollTop>before.scrollHeight*0.2,'Dragging the native scrollbar must seek through the catalog')
    assert.equal(scrollbarDrag.scrollHeight,before.scrollHeight,'Scrollbar dragging changed scroll extent')
  }
  await page.locator('#scroll-target').focus();await page.keyboard.press('End');await ready(page)
  const end=await position(page)
  assert.equal(end.scrollHeight,before.scrollHeight,'Jumping to the last page changed scroll extent')
  assert(end.endVisible,'End must reveal the end-of-results marker')
  if(total)assert.equal(end.last,total-1,'End must render the final catalog item')
  assert.equal(end.documentHeight,end.viewportHeight,'A second document scrollbar appeared')
  assert.equal(end.documentWidth,end.viewportWidth,'Horizontal document overflow appeared')
  return {...end,...(scrollbarDrag?{scrollbarDragTop:scrollbarDrag.scrollTop}:{})}
}

async function positionIndicatorChecks(page) {
  const read=()=>page.evaluate(()=>{
    const rect=selector=>{const element=document.querySelector(selector);if(!element)return null;const r=element.getBoundingClientRect();return {top:r.top,bottom:r.bottom,left:r.left,right:r.right,width:r.width,height:r.height}}
    const area=document.querySelector('#scroll-target'),bar=document.querySelector('#scrollProgress')
    return {percent:Number(bar.getAttribute('aria-valuenow')),label:document.querySelector('#scrollProgressValue').textContent.trim(),
      bar:rect('#scrollProgress'),text:rect('#scrollProgressValue'),track:rect('.scrollPositionTrack'),fill:rect('.scrollPositionFill'),image:rect('.imageSelectButton .v-img'),
      scrollTop:area.scrollTop,scrollHeight:area.scrollHeight,clientHeight:area.clientHeight,
      width:innerWidth,height:innerHeight,documentWidth:document.documentElement.scrollWidth,documentHeight:document.documentElement.scrollHeight}
  })
  const check=value=>{
    assert.equal(value.documentWidth,value.width);assert.equal(value.documentHeight,value.height)
    for(const rect of [value.bar,value.text,value.track])assert(rect.left>=0&&rect.right<=value.width&&rect.top>=0&&rect.bottom<=value.height,'Position indicator leaves the viewport')
    assert(value.text.left>=value.track.right,'Percentage overlaps the progress track')
    assert.equal(value.text.width,baseline.text.width,'Changing the percentage changed its reserved width')
    assert(Math.abs(value.fill.width/value.track.width*100-value.percent)<0.05,'Fill and percentage disagree')
  }
  const baseline=await read();check(baseline);assert.equal(baseline.label,'0%')
  const selections=[]
  await page.locator('#selectionModeToggle').click()
  for(const count of [0,1,9,10,1024]) {
    if(count===1)await page.locator('.imageSelectButton').first().click()
    if(count===9)await page.locator('.imageSelectButton').nth(8).dispatchEvent('click',{shiftKey:true})
    if(count===10)await page.locator('.imageSelectButton').nth(9).dispatchEvent('click')
    if(count===1024)await page.locator('#selectBatch').click()
    await page.waitForFunction(count=>Number(document.querySelector('#selectionCount').textContent.replace(/[^0-9]/g,''))===count,count)
    await frames(page);const current=await read();check(current)
    for(const element of ['bar','image'])assert(Math.abs(current[element].top-baseline[element].top)<1,`${count} selected shifted ${element}`)
    assert.equal(current.scrollHeight,baseline.scrollHeight)
    selections.push({count,barTop:current.bar.top,imageTop:current.image.top})
  }
  await page.locator('#selectionModeToggle').click();await frames(page)
  const cleared=await read();check(cleared)
  assert.equal(cleared.bar.top,baseline.bar.top);assert.equal(cleared.image.top,baseline.image.top)
  await page.locator('#scroll-target').evaluate(element=>{element.scrollTop=1000});await ready(page)
  const beginning=await read();check(beginning);assert.equal(beginning.label,'<1%')
  let heldRequests=0
  const hold=async route=>{heldRequests++;await sleep(180);await route.continue()}
  await page.route('**/image_item?*',hold)
  await page.locator('#scroll-target').evaluate(element=>{element.scrollTop=(element.scrollHeight-element.clientHeight)/2})
  await page.waitForFunction(()=>document.querySelector('#itemArea').getAttribute('aria-busy')==='true')
  await frames(page);const loading=await read();check(loading)
  await ready(page);const middle=await read();check(middle)
  await page.unroute('**/image_item?*',hold)
  assert(heldRequests>0,'The midpoint must exercise an unloaded catalog page')
  assert(Math.abs(middle.percent-50)<0.01);assert.equal(loading.percent,middle.percent)
  assert.equal(loading.scrollHeight,middle.scrollHeight);assert.equal(middle.scrollHeight,baseline.scrollHeight)
  await page.locator('#scroll-target').evaluate(element=>{element.scrollTop=(element.scrollHeight-element.clientHeight)*.995});await ready(page)
  const nearEnd=await read();check(nearEnd);assert.equal(nearEnd.label,'99%')
  await page.locator('#scroll-target').focus();await page.keyboard.press('End');await ready(page)
  const end=await read();check(end);assert.equal(end.percent,100);assert.equal(end.label,'100%')
  assert.equal(end.scrollHeight,baseline.scrollHeight)
  return {start:baseline.label,beginning:beginning.label,middle:middle.label,nearEnd:nearEnd.label,end:end.label,percentWidth:end.text.width,heldRequests,scrollHeight:end.scrollHeight,selections}
}

async function runScroll(browser,options={}) {
  const fixture=options.url?null:await startFixture()
  const report={kind:fixture?'deterministic 7,099,334-row catalog':'live catalog',url:options.url||fixture.url,cases:[],errors:[],failures:[]}
  try {
    for(const viewport of options.viewports||(options.positionOnly?[{width:320,height:812},...VIEWPORTS]:VIEWPORTS)) {
      const context=await browser.newContext({viewport,hasTouch:true,isMobile:viewport.width<1000})
      const page=await context.newPage(), cdp=await context.newCDPSession(page)
      const result={viewport,passed:true}
      let catalogResponse
      page.on('response',response=>{if(!catalogResponse&&new URL(response.url()).pathname==='/image_item')catalogResponse=response})
      page.on('pageerror',error=>report.errors.push(error.message))
      try {
        const started=Date.now()
        await page.goto(report.url,{waitUntil:'domcontentloaded',timeout:120000});await ready(page)
        result.readyMs=Date.now()-started
        result.catalogCount=Number((await catalogResponse?.allHeaders())?.['x-matching-total'])||undefined
        assert.equal(await page.locator('#itemArea.catalogGrid').count(),1,'This review requires a counted, compressed catalog')
        if(options.positionOnly)result.position=await positionIndicatorChecks(page)
        else {
        result.swipes=[]
        for(const fingers of [1,2])result.swipes.push(await swipe(page,cdp,fingers))
        assert(Math.abs(result.swipes[1].drag-result.swipes[0].drag)<30,'Two fingers must move at the same speed as one')
        result.selection=await selectionChecks(page,cdp)
        result.gestures=await gestureStability(page,cdp)
        assert(result.gestures.every(item=>item.passed),result.gestures.filter(item=>!item.passed).map(item=>item.error).join('; '))
        result.wheel=await wheelChecks(page)
        result.end=await endChecks(page,result.catalogCount)
        if(viewport.width===390)result.touch=await tapAndPinch(page,cdp)
        }
      } catch(error) {result.passed=false;result.error=error.message;report.failures.push({viewport,error:error.message})}
      finally {await context.close()}
      report.cases.push(result)
      console.log(JSON.stringify(result))
    }
  } finally {if(fixture)await fixture.close()}
  assert.deepEqual(report.errors,[],'Browser runtime errors')
  assert.equal(report.failures.length,0,`${report.failures.length} scrolling checks failed`)
  return report
}

module.exports={runScroll,position}
if(require.main===module) {
  ;(async()=>{
    const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright')
    const browser=await chromium.launch({headless:true,ignoreDefaultArgs:['--hide-scrollbars'],...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})})
    const urlIndex=process.argv.indexOf('--url')
    try {const report=await runScroll(browser,{url:urlIndex<0?undefined:process.argv[urlIndex+1],positionOnly:process.argv.includes('--position')});console.log(JSON.stringify({cases:report.cases.length,failures:report.failures.length,url:report.url}))}
    finally {await browser.close()}
  })().catch(error=>{console.error(error);process.exitCode=1})
}
