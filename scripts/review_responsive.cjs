// Deterministic responsive browser fixtures; this is layout/interaction QA,
// not a measurement of the production database or embedding backend.
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const http = require('node:http')

const ROOT = path.resolve(__dirname, '..')
const VIEW = path.join(ROOT, 'app/presentation/view')
const TOTAL = 7_099_334
const WIDTHS = [320, 375, 390, 599, 600, 601, 767, 768, 769, 959, 960, 961, 1279, 1280, 1281, 1440]
const COUNTS = [0, 1, 9, 10, 64, 65, 1024]
const item = index => ({id:`fixture-${index}`, name:`image-${String(index).padStart(7,'0')}.jpg`, tags:'sample', rating:'general'})
const frames = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))

async function startFixture(options = {}) {
  const image = fs.readFileSync(path.join(ROOT, 'tests/fixtures/image.png'))
  const calls = []
  const original = {delayMs:0, failures:0, requests:[]}
  const downloads = {delayMs:options.downloadDelayMs || 0}
  const fixtureItem = index => ({...item(index), ...(options.imageName ? {name:options.imageName} : {})})
  const server = http.createServer(async (request, response) => {
    const url = new URL(request.url, 'http://fixture')
    const json = (body, headers = {}) => { response.writeHead(200, {'Content-Type':'application/json', ...headers}); response.end(JSON.stringify(body)) }
    try {
      if (url.pathname === '/model_item') return json([{model_name:'ViT-L-14', pretrained:'openai'}])
      if (url.pathname === '/image_item') {
        const page = Number(url.searchParams.get('page') || 0), size = Number(url.searchParams.get('size') || 60)
        const ratings = url.searchParams.has('ratings') ? JSON.parse(url.searchParams.get('ratings')) : null
        const matching = ratings === null || ratings.includes('general') ? TOTAL : 0
        const list = Array.from({length:Math.max(0,Math.min(size,matching-page*size))}, (_,i)=>fixtureItem(page*size+i))
        calls.push({method:'GET', path:url.pathname, page, size, matching})
        return json(list, {'X-Catalog-Total':TOTAL, 'X-Matching-Total':matching})
      }
      if (/^\/image\/[^/]+\/(small|original)$/.test(url.pathname)) {
        const full = url.pathname.endsWith('/original')
        if (full) {
          const entry={url:request.url, startedAt:Date.now()}
          original.requests.push(entry)
          const shouldFail=original.failures>0, delayMs=original.delayMs
          if(shouldFail)original.failures--
          if(delayMs) await new Promise(resolve=>setTimeout(resolve,delayMs))
          entry.status=shouldFail?503:200;entry.finishedAt=Date.now()
          if(shouldFail){response.writeHead(503,{'Cache-Control':'no-store'});return response.end('Fixture original temporarily unavailable')}
        }
        response.writeHead(200, {'Content-Type':'image/png', 'Cache-Control':full?'no-store':'public,max-age=3600'}); return response.end(image)
      }
      if(url.pathname.startsWith('/downloads/')&&request.method==='GET'){
        calls.push({method:'GET',path:url.pathname})
        response.writeHead(200,{'Content-Type':'application/zip','Content-Disposition':'attachment; filename="images.zip"'})
        return response.end(Buffer.from('504b0506000000000000000000000000000000000000','hex'))
      }
      if (url.pathname.startsWith('/image_meta/')) return json({tags:'sample',rating:'general',style_cluster:'',aesthetic_quality:5})
      if (request.method === 'POST') {
        let body = ''; for await (const chunk of request) body += chunk
        const params = JSON.parse(body).params
        calls.push({method:'POST', path:url.pathname, params})
        if(url.pathname==='/downloads/prepare'){
          if(downloads.delayMs)await new Promise(resolve=>setTimeout(resolve,downloads.delayMs))
          const count=params.ids?.length || params.first || 0
          return json({download_url:`/downloads/fixture-${calls.filter(call=>call.path==='/downloads/prepare').length}`,requested_count:count,image_count:count,skipped_count:0,bytes:22})
        }
        if (url.pathname === '/image_ratings') return json(Object.fromEntries((params.ids || []).map(id=>[id,'general'])))
        if (url.pathname.startsWith('/search/')) {
          const length = Math.min(2048,params.result_size || 2048)
          return json({list:Array.from({length},(_,i)=>({item:fixtureItem(i),score:1-i/length})),search_query:'fixture-query'})
        }
      }
      const file = url.pathname === '/' ? path.join(VIEW,'index.html') : path.resolve(VIEW, `.${decodeURIComponent(url.pathname)}`)
      if (!file.startsWith(VIEW + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
        response.writeHead(404); return response.end('Missing fixture resource')
      }
      const types = {'.js':'application/javascript','.css':'text/css','.json':'application/json','.html':'text/html','.woff2':'font/woff2','.woff':'font/woff','.ttf':'font/ttf','.svg':'image/svg+xml'}
      response.writeHead(200, {'Content-Type':types[path.extname(file)] || 'application/octet-stream'})
      fs.createReadStream(file).pipe(response)
    } catch (error) { response.writeHead(500, {'Content-Type':'application/json'}); response.end(JSON.stringify({error:error.message})) }
  })
  await new Promise(resolve => server.listen(0,'127.0.0.1',resolve))
  return {url:`http://127.0.0.1:${server.address().port}`,calls,original,downloads,close:()=>new Promise(resolve=>server.close(resolve))}
}

async function ready(page) {
  await page.waitForFunction(() => document.querySelector('#itemArea')?.getAttribute('aria-busy') !== 'true' && document.querySelector('.itemBlock[data-image-id]'), null, {timeout:120000})
  await page.evaluate(() => document.fonts.ready)
  await frames(page)
}

async function setLocale(page, locale) {
  await page.locator('#settingsToggle').click()
  await page.locator('#settingsPanel').waitFor({state:'visible'})
  assert.equal(await page.locator('#headerLocale').count(),0,'Language controls belong only in the settings menu')
  await page.locator('#drawerLocale').selectOption(locale)
  await page.keyboard.press('Escape')
  await page.locator('#settingsPanel').waitFor({state:'hidden'})
  await frames(page)
}

async function fixtureState(page, count, selectMode) {
  await page.evaluate(({count,selectMode}) => {
    const root = document.querySelector('#app')
    const vm = root.__vue_app__?._instance?.proxy || root._vnode?.component?.proxy
    if (!vm) throw new Error('Vue root proxy is unavailable for deterministic layout fixture setup')
    vm.selectMode = selectMode
    vm.selectedItemId = Object.fromEntries(Array.from({length:count},(_,i)=>[`fixture-${i}`,true]))
    vm.selectionMessage = ''; vm.errorMessage = ''; vm.downloadMessage = ''
  }, {count,selectMode})
  await frames(page)
}

async function geometry(page) {
  return page.evaluate(() => {
    const visible = element => {
      if (!element) return false
      const style = getComputedStyle(element), r = element.getBoundingClientRect()
      return r.width > 0 && r.height > 0 && style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity) !== 0
    }
    const rect = element => {
      if (!element) return null
      const b = element.getBoundingClientRect()
      return {x:b.x,y:b.y,width:b.width,height:b.height,top:b.top,bottom:b.bottom,left:b.left,right:b.right}
    }
    const scroll = document.querySelector('#scroll-target'), grid = document.querySelector('#itemArea')
    const image = document.querySelector('.itemBlock[data-image-id] .v-img')
    const controls = [...document.querySelectorAll('#controlArea button,#selectionModeToggle,#selectionCount,#selectBatch,.downloadAction')].filter(visible)
    const overflowingControls = controls.filter(element => {const r=element.getBoundingClientRect();return r.left < -1 || r.right > innerWidth + 1}).map(element=>element.id||element.className)
    const helperParagraphs = [...document.querySelectorAll('.resultsContainer > p,.galleryChrome > p')].filter(visible).map(element=>element.textContent.trim()).filter(Boolean)
    return {
      width:innerWidth,height:innerHeight,documentWidth:document.documentElement.scrollWidth,documentHeight:document.documentElement.scrollHeight,
      scrollHeight:scroll.scrollHeight,scrollWidth:scroll.scrollWidth,clientWidth:scroll.clientWidth,clientHeight:scroll.clientHeight,scrollTop:scroll.scrollTop,
      grid:rect(grid),image:rect(image),header:rect(document.querySelector('#controlArea')),imageSearch:rect(document.querySelector('#imageSearchAction')),
      countText:document.querySelector('#selectionCount')?.textContent.trim()||'',
      visibleCheckboxes:[...document.querySelectorAll('.imageSelectButton[role="checkbox"]')].filter(visible).length,
      visibleSelectionMarks:[...document.querySelectorAll('.selectionMark')].filter(visible).length,
      visibleSelectionCounts:visible(document.querySelector('#selectionCount')) ? 1 : 0,
      expansionButtons:document.querySelectorAll('.imageDetailButton').length,
      overflowingControls,helperParagraphs,visibleControlText:controls.map(element=>element.textContent.trim()).join(' '),
    }
  })
}

function checkGeometry(current, baseline, label) {
  assert.equal(current.documentWidth,current.width,`${label}: document has horizontal overflow`)
  assert.equal(current.documentHeight,current.height,`${label}: a second document scrollbar appeared`)
  assert(current.scrollWidth <= current.clientWidth + 1,`${label}: grid has horizontal overflow`)
  assert.deepEqual(current.overflowingControls,[],`${label}: controls leave the viewport`)
  assert(current.image && current.image.top < Math.min(300,current.height*0.4),`${label}: text/chrome pushes the first image too far down`)
  assert(current.helperParagraphs.length <= 1,`${label}: explanatory paragraphs should stay in the menu/help`)
  assert(current.visibleControlText.length < 300,`${label}: excessive visible control copy`)
  if (baseline) {
    assert(Math.abs(current.image.top-baseline.image.top)<=1,`${label}: first image top shifted (${baseline.image.top} -> ${current.image.top})`)
    assert(Math.abs(current.scrollHeight-baseline.scrollHeight)<=1,`${label}: selection changed total scroll extent`)
  }
}

async function interactionChecks(page, fixture, locale) {
  await fixtureState(page,0,false)
  const normal = await geometry(page)
  assert.equal(normal.visibleCheckboxes,0)
  assert.equal(normal.visibleSelectionMarks,0)
  assert.equal(normal.visibleSelectionCounts,0)
  assert.equal(normal.expansionButtons,0)
  await page.locator('.imageSelectButton').first().click()
  await page.locator('.imageDialog').waitFor({state:'visible'})
  await page.keyboard.press('Escape')
  await page.locator('.imageDialog').waitFor({state:'hidden'})
  await page.waitForFunction(()=>document.activeElement?.classList.contains('imageSelectButton'))
  await page.locator('#selectionModeToggle').click()
  await page.locator('.imageSelectButton[role="checkbox"]').first().waitFor()
  await page.locator('#selectBatch').click()
  await page.waitForFunction(() => Number(document.querySelector('#selectionCount')?.textContent.replace(/[^0-9]/g,'')) === 1024)
  const batch = await geometry(page)
  assert.equal(batch.scrollHeight,normal.scrollHeight,'Selecting the first 1024 must preserve gallery height')
  assert(Math.abs(batch.image.top-normal.image.top)<=1,'Bulk selection must preserve the first image position')
  await page.locator('#selectionModeToggle').click()
  assert.equal(await page.locator('#selectionCount').count(),0,'Exiting selection mode removes the selection count')
  assert.equal(await page.locator('.itemBlock[data-selected="true"]').count(),0,'Exiting selection mode clears the selection')
  assert.equal(await page.locator('.imageSelectButton[role="checkbox"]').count(),0)
  await page.locator('#selectionModeToggle').click()
  await page.locator('.imageSelectButton').first().click()
  assert.equal(await page.locator('.imageDialog').isVisible(),false)
  assert.equal(await page.locator('.imageSelectButton').first().getAttribute('aria-checked'),'true')
  await page.locator('#scroll-target').hover()
  await page.mouse.wheel(0,900)
  await ready(page)
  const scrolled = await geometry(page)
  assert(scrolled.imageSearch.top >= 0 && scrolled.imageSearch.bottom <= scrolled.height,'Image search must remain available after scrolling')
  const before = fixture.calls.length
  await page.locator('#imageSearchAction').click()
  await ready(page)
  const imageSearch = fixture.calls.slice(before).find(call=>call.path==='/search/image')
  assert(imageSearch,'The fixed image-search action must execute a request')
  assert.equal(imageSearch.params.id.length,1)
  await fixtureState(page,65,true)
  const beforeLimit = await geometry(page)
  await page.locator('#imageSearchAction').click()
  await frames(page)
  const afterLimit = await geometry(page)
  assert.equal(fixture.calls.slice(before).filter(call=>call.path==='/search/image').length,1,'65 selected images must not submit oversized inference')
  assert.equal(afterLimit.scrollHeight,beforeLimit.scrollHeight,'Image-search limit notice must not push the grid')
  assert(Math.abs(afterLimit.image.top-beforeLimit.image.top)<=1)
  await fixtureState(page,0,false)
  const chooserPromise = page.waitForEvent('filechooser')
  await page.locator('#imageSearchAction').click()
  const chooser = await chooserPromise
  assert.equal(chooser.isMultiple(),false)
  await chooser.setFiles(path.join(ROOT,'tests/fixtures/image.png'))
  await ready(page)
  assert(fixture.calls.some(call=>call.path==='/search/uploadimage'),'Image search without selection opens a file picker and supports upload search')
  return {locale,normal,scrolled,bulk1024Selection:true,exitClearsSelection:true,selectedImageSearch:true,zeroSelectionFilePicker:true,limitPreservedLayout:true}
}

async function runResponsive(browser, outputDir) {
  fs.mkdirSync(outputDir,{recursive:true})
  const fixture = await startFixture()
  const page = await browser.newPage({viewport:{width:1440,height:1000}})
  const errors=[], report={kind:'deterministic frontend fixture',productionPerformance:false,totalFixtureRows:TOTAL,widths:WIDTHS,counts:COUNTS,locales:['ja','en'],surfaces:['catalog','search'],cases:[],interactions:[],failures:[],errors}
  page.on('pageerror',error=>errors.push(error.message))
  const write=()=>fs.writeFileSync(path.join(outputDir,'responsive-review.json'),JSON.stringify(report,null,2))
  try {
    await page.goto(fixture.url,{waitUntil:'networkidle'});await ready(page)
    for (const width of WIDTHS) {
      await page.setViewportSize({width,height:width<=600?812:1000});await ready(page)
      for (const locale of ['ja','en']) {
        await setLocale(page,locale)
        // Action checks use actual DOM interactions; selected-count matrices
        // then use explicit Vue state fixtures to avoid 1024 repetitive clicks.
        if ([320,375,600,768,960,1280,1440].includes(width)) {
          try {report.interactions.push({width,...await interactionChecks(page,fixture,locale)})}
          catch(error){
            report.failures.push({width,locale,stage:'interaction',error:error.message})
            await page.screenshot({path:path.join(outputDir,`failure-${width}-${locale}-interaction.png`)})
            await page.keyboard.press('Escape');await page.keyboard.press('Escape')
            await page.evaluate(()=>{const root=document.querySelector('#app'),vm=root.__vue_app__?._instance?.proxy||root._vnode?.component?.proxy;vm.activeDialogItem=null;vm.isShowSetting=false})
          }
        }
        for (const surface of ['catalog','search']) {
          await page.evaluate(async surface=>{
            const root=document.querySelector('#app'),vm=root.__vue_app__?._instance?.proxy||root._vnode?.component?.proxy
            vm.selectMode=false;vm.clearSelection(false)
            if(surface==='catalog')await vm.browseImages()
          },surface)
          if(surface==='search'){await page.locator('#textSearchBox').fill('sample');await page.locator('#textSearchBox').press('Enter')}
          await ready(page)
          await page.locator('#scroll-target').focus();await page.keyboard.press('Home');await ready(page)
          await fixtureState(page,0,false)
          const baseline=await geometry(page)
          try {checkGeometry(baseline,null,`${width}/${locale}/${surface}/normal`);assert.equal(baseline.visibleCheckboxes,0);assert.equal(baseline.visibleSelectionCounts,0)}
          catch(error){report.failures.push({width,locale,surface,stage:'normal',error:error.message})}
          for(const count of COUNTS){
            await fixtureState(page,count,true)
            const current=await geometry(page)
            const result={width,locale,surface,count,geometry:current,passed:true}
            try{
              checkGeometry(current,baseline,`${width}/${locale}/${surface}/${count}`)
              assert.equal(current.visibleSelectionCounts,1)
              if(count)assert.equal(Number(current.countText.replace(/[^0-9]/g,'')),count,'Selection count must reflect the actual number')
              else assert(current.countText.length>0&&!/[0-9]/.test(current.countText),'Zero selection uses a mode label without a redundant zero count')
            }
            catch(error){result.passed=false;result.error=error.message;report.failures.push({width,locale,surface,count,error:error.message})}
            report.cases.push(result)
            if([320,375,600,960,1440].includes(width)&&[0,1024].includes(count)&&surface==='catalog')await page.screenshot({path:path.join(outputDir,`layout-${width}-${locale}-${count}.png`)})
          }
        }
        write()
      }
      console.log(JSON.stringify({width,cases:report.cases.length,failures:report.failures.length}))
    }
    report.generatedAt=new Date().toISOString();write()
    assert.deepEqual(errors,[],'Browser runtime errors')
    assert.equal(report.failures.length,0,`${report.failures.length} responsive/interaction checks failed; see responsive-review.json`)
    return report
  } finally {write();await page.close();await fixture.close()}
}

module.exports={runResponsive,startFixture,geometry,ready,setLocale,fixtureState,frames}
if(require.main===module){
  ;(async()=>{
    const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright')
    const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})})
    try{const report=await runResponsive(browser,path.join(ROOT,'.cache/responsive-review'));console.log(JSON.stringify({cases:report.cases.length,interactions:report.interactions.length,failures:report.failures.length}))}
    finally{await browser.close()}
  })().catch(error=>{console.error(error);process.exitCode=1})
}
