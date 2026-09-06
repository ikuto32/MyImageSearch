// Real Chromium UX checks. Set PLAYWRIGHT_MODULE and CHROME_PATH when the
// browser/runtime is supplied outside node_modules. UX_BASE_URL defaults to 5001.
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

async function setLocale(page, locale) {
  assert.equal(await page.locator('#headerLocale').count(), 0)
  await page.locator('#settingsToggle').click()
  await page.locator('#settingsPanel').waitFor({state:'visible'})
  await page.locator('#drawerLocale').selectOption(locale)
  await page.keyboard.press('Escape')
  await page.locator('#settingsPanel').waitFor({state:'hidden'})
  assert.equal(await page.locator('html').getAttribute('lang'), locale)
}

async function setSelectionMode(page, enabled) {
  const toggle = page.locator('#selectionModeToggle')
  if ((await toggle.getAttribute('aria-pressed') === 'true') !== enabled) await toggle.click()
  await page.waitForFunction(expected => (document.querySelector('#selectionModeToggle')?.getAttribute('aria-pressed') === 'true') === expected, enabled)
}

async function backToCollection(page, locale = 'en') {
  await page.locator('#settingsToggle').click()
  await page.locator('#settingsPanel').waitFor({state:'visible'})
  await page.getByRole('button', {name:locale === 'en' ? 'Back to collection' : '画像一覧に戻る', exact:true}).click()
  if (await page.locator('#settingsPanel').isVisible()) await page.keyboard.press('Escape')
  await page.locator('#settingsPanel').waitFor({state:'hidden'})
  await waitForCatalogWindow(page)
}

async function metrics(page) {
  return page.evaluate(() => {
    const scroll = document.querySelector('#scroll-target')
    const viewport = scroll.getBoundingClientRect()
    const cards = [...document.querySelectorAll('.itemBlock')]
    const visible = cards.filter(card => { const bounds = card.getBoundingClientRect(); return bounds.bottom > viewport.top && bounds.top < viewport.bottom })
    const logicalIndices = visible.map(card => Number(card.dataset.logicalIndex)).filter(Number.isFinite)
    const first = visible.find(card => Number(card.dataset.logicalIndex) === Math.min(...logicalIndices))
    const columns = getComputedStyle(document.querySelector('#itemArea')).gridTemplateColumns.split(' ').length
    return {
      viewportWidth: innerWidth, viewportHeight: innerHeight,
      documentHeight: document.documentElement.scrollHeight,
      documentWidth: document.documentElement.scrollWidth,
      clientHeight: scroll.clientHeight, scrollHeight: scroll.scrollHeight,
      scrollTop: scroll.scrollTop, cards: document.querySelectorAll('.itemBlock').length,
      scrollBottom: viewport.bottom, scrollViewportTop: viewport.top,
      gridHeight: document.querySelector('#itemArea').getBoundingClientRect().height,
      candidateLabel: document.querySelector('.candidateCount')?.textContent.trim(),
      matchLabel: document.querySelector('.resultCount')?.textContent.trim(),
      logicalFirst: logicalIndices.length ? Math.min(...logicalIndices) : null,
      logicalLast: logicalIndices.length ? Math.max(...logicalIndices) : null,
      logicalScrollPixels: first ? Math.floor(Number(first.dataset.logicalIndex) / columns) * 282 + viewport.top - first.getBoundingClientRect().top : null,
      renderedIndices: cards.map(card => Number(card.dataset.logicalIndex)).filter(Number.isFinite),
      imageIds: visible.map(card => card.dataset.imageId).filter(Boolean),
      decodedImages: visible.filter(card => [...card.querySelectorAll('img')].some(image => image.complete && image.naturalWidth > 0)).length,
      unavailableImages: visible.filter(card => card.querySelector('.imageError')).length,
    }
  })
}

async function waitForScrollRest(page) {
  await page.evaluate(() => new Promise(resolve => {
    const area = document.querySelector('#scroll-target')
    let previous = area.scrollTop, steadyFrames = 0, frames = 0
    function sample() {
      steadyFrames = Math.abs(area.scrollTop - previous) < 0.01 ? steadyFrames + 1 : 0
      previous = area.scrollTop
      if (steadyFrames >= 4 || frames++ > 120) return resolve()
      requestAnimationFrame(sample)
    }
    requestAnimationFrame(sample)
  }))
}

async function waitForCatalogWindow(page) {
  await page.waitForFunction(() => {
    const grid = document.querySelector('#itemArea')
    const cards = [...document.querySelectorAll('.itemBlock[data-image-id]')]
    return grid && grid.getAttribute('aria-busy') !== 'true' && cards.length > 0
  }, null, {timeout:120000})
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
}

async function waitForVisibleImage(page) {
  await page.waitForFunction(() => {
    const bounds = document.querySelector('#scroll-target').getBoundingClientRect()
    return [...document.querySelectorAll('.itemBlock[data-image-id]')].some(card => {
      const rect = card.getBoundingClientRect()
      return rect.bottom > bounds.top && rect.top < bounds.bottom && (card.querySelector('.imageError') || [...card.querySelectorAll('img')].some(image => image.complete && image.naturalWidth > 0))
    })
  }, null, {timeout:120000})
}

async function waitForVisibleImagesSettled(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector('#scroll-target').getBoundingClientRect()
    const cards = [...document.querySelectorAll('.itemBlock[data-image-id]')].filter(card => {
      const image = card.querySelector('.v-img')
      if (!image) return false
      const rect = image.getBoundingClientRect()
      return rect.bottom > viewport.top && rect.top < viewport.bottom
    })
    return cards.length > 0 && cards.every(card => card.querySelector('.imageError') || [...card.querySelectorAll('img')].some(image => image.complete && image.naturalWidth > 0))
  }, null, {timeout:120000})
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
}

const ratingLabels = ['一般向け', '刺激の強い内容', '成人向けの可能性', '成人向け', '未分類・評価なし']

async function setCategories(page, requestedLabels) {
  await page.locator('#settingsToggle').click()
  await page.locator('#settingsPanel').waitFor({state:'visible'})
  await page.getByRole('combobox', {name:'対象にする画像の区分', exact:true}).locator('input').press('ArrowDown')
  for (const label of ratingLabels) {
    const option = page.getByRole('option', {name:label, exact:true})
    const selected = await option.locator('input[type="checkbox"]').isChecked()
    if (selected !== requestedLabels.includes(label)) await option.click()
  }
  // The first Escape dismisses the listbox, and the second dismisses settings.
  await page.keyboard.press('Escape')
  await page.getByRole('listbox').waitFor({state:'hidden'})
  await page.keyboard.press('Escape')
  await page.locator('#settingsPanel').waitFor({state:'hidden'})
}

async function reviewCatalogFilters(page, catalogResponses, fullTotal) {
  await setCategories(page, ['一般向け'])
  await waitForCatalogWindow(page)
  await page.waitForFunction(() => document.querySelector('.resultCount')?.textContent.trim() && document.querySelector('#itemArea').getAttribute('aria-busy') !== 'true')
  const filtered = await metrics(page)
  const matchingResponse = catalogResponses.findLast(response => {
    const ratings = new URL(response.url).searchParams.get('ratings')
    return ratings && JSON.parse(ratings).join(',') === 'general' && Number.isFinite(response.matchingCount)
  })
  assert(matchingResponse, 'Filtered requests must return a matching count')
  assert(matchingResponse.matchingCount > 0 && matchingResponse.matchingCount <= fullTotal)
  assert.equal(Number(filtered.matchLabel.match(/[\d,]+/)[0].replaceAll(',','')), matchingResponse.matchingCount)
  assert.equal(Number(filtered.candidateLabel.match(/[\d,]+/)[0].replaceAll(',','')), fullTotal)
  await page.locator('#scroll-target').hover()
  await page.mouse.wheel(0, 1300)
  await page.waitForTimeout(100)
  const during = await metrics(page)
  await waitForCatalogWindow(page)
  const after = await metrics(page)
  assert(Math.abs(during.scrollHeight - filtered.scrollHeight) <= 1)
  assert(Math.abs(after.scrollHeight - filtered.scrollHeight) <= 1)

  await setCategories(page, [])
  await page.waitForFunction(() => document.querySelector('#itemArea').getAttribute('aria-busy') !== 'true' && document.querySelectorAll('.itemBlock[data-image-id]').length === 0)
  const empty = await metrics(page)
  assert.equal(Number(empty.matchLabel.match(/[\d,]+/)[0].replaceAll(',','')), 0)
  assert.equal(await page.locator('.downloadAction').isDisabled(), true)
  assert(empty.gridHeight <= empty.clientHeight, 'No matching items should leave no enormous empty rail')
  assert.match(await page.locator('.emptyState').innerText(), /区分/)
  if (await page.locator('.loadMoreArea button').count()) assert.equal(await page.locator('.loadMoreArea button').isDisabled(), true)
  await setCategories(page, ratingLabels)
  await waitForCatalogWindow(page)
  await page.waitForFunction(total => {
    const text = document.querySelector('.resultCount')?.textContent || ''
    return Number(text.match(/[\d,]+/)?.[0].replaceAll(',','')) === total
  }, fullTotal)
  return {filtered, matchingCount:matchingResponse.matchingCount, during, after, empty}
}

async function reviewNarrowCatalog(page, total) {
  await backToCollection(page)
  await waitForCatalogWindow(page)
  await waitForVisibleImage(page)
  const initial = await metrics(page)
  const columns = await page.locator('#itemArea').evaluate(grid => getComputedStyle(grid).gridTemplateColumns.split(' ').length)
  assert.equal(initial.logicalFirst, 0)
  assert(initial.scrollBottom <= initial.viewportHeight + 1)
  assert.equal(initial.documentWidth, initial.viewportWidth)
  const oneScreenLimit = initial.clientHeight + 282
  await page.locator('#scroll-target').focus()
  await page.keyboard.press('PageDown')
  await waitForScrollRest(page)
  await waitForCatalogWindow(page)
  const afterKey = await metrics(page)
  assert(afterKey.logicalScrollPixels > initial.logicalScrollPixels)
  assert(afterKey.logicalScrollPixels - initial.logicalScrollPixels <= oneScreenLimit, 'PageDown jumped many logical screens on the compressed rail')
  assert(Math.abs(afterKey.scrollHeight - initial.scrollHeight) <= 1)
  await page.locator('#scroll-target').hover()
  await page.mouse.wheel(0, 700)
  await waitForScrollRest(page)
  await waitForCatalogWindow(page)
  const afterWheel = await metrics(page)
  assert(afterWheel.logicalScrollPixels > afterKey.logicalScrollPixels)
  assert(afterWheel.logicalScrollPixels - afterKey.logicalScrollPixels <= oneScreenLimit, 'Wheel movement jumped many logical screens on the compressed rail')
  assert(Math.abs(afterWheel.scrollHeight - initial.scrollHeight) <= 1)
  await page.locator('#scroll-target').focus()
  for (let i = 0; i < 5; i++) await page.keyboard.press('ArrowDown')
  await waitForScrollRest(page)
  const afterArrows = await metrics(page)
  assert(afterArrows.logicalScrollPixels - afterWheel.logicalScrollPixels >= 100, 'Small key movements were lost to scrollTop pixel rounding')
  assert(afterArrows.logicalScrollPixels - afterWheel.logicalScrollPixels <= 300, 'Small key movements accumulated an excessive logical distance')
  await page.locator('#scroll-target').focus()
  await page.keyboard.press('End')
  await page.waitForFunction(last => [...document.querySelectorAll('.itemBlock[data-logical-index]')].some(card => Number(card.dataset.logicalIndex) === last), total - 1, {timeout:120000})
  await waitForCatalogWindow(page)
  await page.waitForFunction(() => {
    const end = document.querySelector('.resultsEnd')?.getBoundingClientRect()
    return end && end.top >= document.querySelector('#scroll-target').getBoundingClientRect().top && end.bottom <= innerHeight + 1
  }, null, {timeout:120000})
  await waitForVisibleImage(page)
  const end = await metrics(page)
  assert(Math.abs(end.scrollHeight - initial.scrollHeight) <= 1)
  await page.keyboard.press('Home')
  await page.waitForFunction(() => document.querySelector('.itemBlock[data-logical-index]')?.dataset.logicalIndex === '0')
  await waitForCatalogWindow(page)
  return {columns, initial, afterKey, afterWheel, afterArrows, end}
}

async function reviewKeyboardContinuity(page) {
  await page.locator('#scroll-target').focus()
  await page.keyboard.press('Home')
  await page.waitForFunction(() => document.querySelector('.itemBlock[data-logical-index]')?.dataset.logicalIndex === '0')
  await waitForCatalogWindow(page)
  await page.locator('.imageSelectButton').first().focus()
  const steps = []
  for (let index = 0; index < 5; index++) {
    await page.keyboard.press('PageDown')
    await waitForScrollRest(page)
    await waitForCatalogWindow(page)
    const geometry = await metrics(page)
    const focus = await page.evaluate(() => ({tag:document.activeElement.tagName,id:document.activeElement.id,className:document.activeElement.className}))
    assert.notEqual(focus.tag, 'BODY', 'Virtualization discarded keyboard focus')
    assert(focus.id === 'scroll-target' || String(focus.className).includes('imageSelectButton'))
    if (steps.length) assert(geometry.logicalScrollPixels > steps.at(-1).logicalScrollPixels, 'Repeated PageDown stopped after a focused card was recycled')
    steps.push({...geometry,focus})
  }
  assert(steps.some(step => step.focus.id === 'scroll-target'), 'The test must recycle the initially focused card')
  await page.keyboard.press('Home')
  await page.waitForFunction(() => document.querySelector('.itemBlock[data-logical-index]')?.dataset.logicalIndex === '0')
  await waitForCatalogWindow(page)
  return steps
}

async function reviewHeaderImageSearch(page) {
  await page.locator('#scroll-target').focus()
  await page.keyboard.press('Home')
  await waitForCatalogWindow(page)
  await setSelectionMode(page, true)
  const selectedId = await page.locator('.itemBlock[data-image-id]').first().getAttribute('data-image-id')
  await page.locator('.imageSelectButton').first().click()
  await page.locator('#scroll-target').hover()
  await page.mouse.wheel(0, 1300)
  await waitForScrollRest(page)
  await waitForCatalogWindow(page)
  const before = await metrics(page)
  const action = await page.locator('#imageSearchAction').boundingBox()
  assert(action.y >= 0 && action.y + action.height <= before.viewportHeight, 'The header image-search action left the viewport')
  assert(before.logicalScrollPixels > 800, 'This action must be exercised while the gallery is scrolled')
  const requestTime = performance.now()
  const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/search/image', {timeout:120000})
  await page.locator('#imageSearchAction').click()
  const response = await responsePromise
  assert.equal(response.status(), 200)
  assert.deepEqual(response.request().postDataJSON().params.id, [selectedId])
  await waitForCatalogWindow(page)
  const after = await metrics(page)
  assert(after.imageIds.length > 0, 'Selected-image search must display real results')
  await backToCollection(page, 'ja')
  return {before, action, selectedId, requestStatus:response.status(), elapsedMs:Math.round(performance.now()-requestTime), elapsedScope:'Search request, result rendering, and restoration of the collection; not isolated inference latency', after}
}

async function runFinalChecks(page, url, outputDir) {
  const reportPath = path.join(outputDir, 'ux-browser-review.json')
  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'))
  await page.setViewportSize({width:1440,height:1000})
  await page.goto(url, {waitUntil:'networkidle'})
  await waitForCatalogWindow(page)
  report.keyboardContinuity = await reviewKeyboardContinuity(page)
  await waitForVisibleImagesSettled(page)
  report.desktopArtifact = {mode:'catalog', locale:'ja', ...await metrics(page)}
  await page.screenshot({path:path.join(outputDir, 'ux-desktop.png')})
  await setLocale(page, 'en')
  await page.setViewportSize({width:375,height:812})
  await page.waitForFunction(() => Math.abs(document.querySelector('#scroll-target').clientHeight - (innerHeight - document.querySelector('#scroll-target').getBoundingClientRect().top)) <= 1)
  await waitForCatalogWindow(page)
  const mobileFinalInitial = await metrics(page)
  await waitForVisibleImagesSettled(page)
  report.mobileArtifact = {mode:'catalog', locale:'en', ...await metrics(page)}
  await page.screenshot({path:path.join(outputDir, 'ux-mobile.png')})
  await page.locator('#scroll-target').focus()
  await page.keyboard.press('End')
  await page.waitForFunction(last => [...document.querySelectorAll('.itemBlock[data-logical-index]')].some(card => Number(card.dataset.logicalIndex) === last), report.catalog.total - 1, {timeout:120000})
  await waitForCatalogWindow(page)
  await waitForVisibleImage(page)
  await page.waitForFunction(() => {
    const end = document.querySelector('.resultsEnd')?.getBoundingClientRect()
    return end && end.top >= document.querySelector('#scroll-target').getBoundingClientRect().top && end.bottom <= innerHeight + 1
  }, null, {timeout:120000})
  await waitForVisibleImagesSettled(page)
  const mobileFinalEnd = await metrics(page)
  assert.equal(mobileFinalEnd.scrollHeight, mobileFinalInitial.scrollHeight)
  report.mobileFinal = {locale:'en', initial:mobileFinalInitial, end:mobileFinalEnd}
  delete report.mobileFinalInitial
  assert.equal(mobileFinalEnd.logicalLast, report.catalog.total - 1)
  assert(mobileFinalEnd.decodedImages > 0, 'At least one final-page image should finish decoding')
  await page.screenshot({path:path.join(outputDir, 'ux-mobile-catalog-end.png')})
  report.checks = [...new Set([...report.checks, 'virtualization-focus-retained', 'mobile-final-image-decoded'])]
  report.finalChecksAt = new Date().toISOString()
  fs.writeFileSync(reportPath, JSON.stringify(report,null,2))
  return {keyboardSteps:report.keyboardContinuity.map(({logicalFirst,logicalScrollPixels,focus})=>({logicalFirst,logicalScrollPixels,focus})), mobileFinal:report.mobileFinal, checks:report.checks}
}

async function probeBrowserHeightLimit(page) {
  const probeContext = await page.context().browser().newContext()
  const probe = await probeContext.newPage()
  try {
    await probe.setContent('<div id="rail" style="height:100px;overflow:auto"><div id="space" style="height:1000000000px"></div></div>')
    return await probe.evaluate(() => ({
      requestedHeight: 1000000000,
      renderedHeight: document.querySelector('#space').getBoundingClientRect().height,
      scrollHeight: document.querySelector('#rail').scrollHeight,
    }))
  } finally { await probeContext.close() }
}

async function reviewCatalogScroll(page, catalogResponses, browserHeightLimit) {
  await waitForCatalogWindow(page)
  await waitForVisibleImage(page)
  const initial = await metrics(page)
  assert.equal(initial.documentHeight, initial.viewportHeight)
  assert.equal(initial.documentWidth, initial.viewportWidth)
  assert(initial.scrollBottom <= initial.viewportHeight + 1)
  assert(initial.gridHeight <= 8000001, 'Physical grid must stay below the agreed 8M px budget')
  assert(initial.scrollHeight < browserHeightLimit.scrollHeight, 'Do not rely on browser-clamped CSS heights')
  assert.equal(initial.logicalFirst, 0)
  const firstIds = initial.imageIds
  assert.equal(await page.locator('.imageSelectButton[role="checkbox"]').count(), 0)
  assert.equal(await page.locator('#selectionCount').count(), 0)
  assert.equal(await page.locator('.imageDetailButton').count(), 0)
  await setSelectionMode(page, true)
  await page.locator('.imageSelectButton').first().click()
  const start = await metrics(page)
  const observations = [start]
  const initialRequests = catalogResponses.length
  await page.locator('#scroll-target').hover()
  for (let attempt = 0; attempt < 6 && catalogResponses.length === initialRequests; attempt++) {
    await page.mouse.wheel(0, 700)
    await page.waitForTimeout(100)
    observations.push(await metrics(page))
    await waitForCatalogWindow(page)
  }
  assert(catalogResponses.length > initialRequests, 'Wheel movement should load a new logical window')
  await waitForVisibleImage(page)
  observations.push(await metrics(page))
  assert(observations.every(item => Math.abs(item.scrollHeight - start.scrollHeight) <= 1), 'Catalog height changed while fetching more images')
  assert.match(await page.locator('#selectionCount').innerText(), /1/)

  for (const fraction of [0.25, 0.5, 0.9]) {
    const old = await metrics(page)
    await page.locator('#scroll-target').evaluate((scroll, targetFraction) => { scroll.scrollTop = (scroll.scrollHeight - scroll.clientHeight) * targetFraction }, fraction)
    await page.waitForFunction(previous => {
      const card = document.querySelector('.itemBlock[data-logical-index]')
      return card && Number(card.dataset.logicalIndex) !== previous
    }, old.renderedIndices[0])
    await waitForCatalogWindow(page)
    await waitForVisibleImage(page)
    const jumped = await metrics(page)
    assert(Math.abs(jumped.scrollHeight - start.scrollHeight) <= 1)
    assert(jumped.logicalFirst > old.logicalFirst, `Jump to ${fraction} did not advance through the catalog`)
    assert(jumped.imageIds.length > 0)
    const delivered = new Set(catalogResponses.flatMap(response => response.ids))
    assert(jumped.imageIds.every(id => delivered.has(id)), 'A displayed image does not belong to a real API response')
    observations.push(jumped)
  }

  await page.locator('#scroll-target').focus()
  await page.keyboard.press('End')
  await waitForCatalogWindow(page)
  await page.waitForFunction(() => {
    const end = document.querySelector('.resultsEnd')?.getBoundingClientRect()
    return end && end.top >= document.querySelector('#scroll-target').getBoundingClientRect().top && end.bottom <= innerHeight + 1
  }, null, {timeout:120000})
  await waitForVisibleImage(page)
  const end = await metrics(page)
  const totals = catalogResponses.filter(response => Number.isFinite(response.matchingCount || response.totalCount))
  const total = totals.at(-1)?.matchingCount ?? totals.at(-1)?.totalCount
  assert(Number.isFinite(total) && total > 0, 'The server must expose the catalog count')
  assert(end.renderedIndices.includes(total - 1), `End must reach logical image ${total - 1}`)
  assert(Math.abs(end.scrollHeight - start.scrollHeight) <= 1)
  observations.push(end)
  await page.keyboard.press('Home')
  await page.waitForFunction(() => document.querySelector('#scroll-target').scrollTop <= 1)
  await page.waitForFunction(() => document.querySelector('.itemBlock[data-logical-index]')?.dataset.logicalIndex === '0')
  await waitForCatalogWindow(page)
  const home = await metrics(page)
  assert.equal(home.logicalFirst, 0)
  assert(home.imageIds.some(id => firstIds.includes(id)))
  assert.match(await page.locator('#selectionCount').innerText(), /1/)
  await setSelectionMode(page, false)
  assert.equal(await page.locator('[data-selected="true"]').count(), 0)
  assert.equal(await page.locator('#selectionCount').count(), 0)
  return {initial, start, observations, total}
}

async function runReview(page, url, outputDir) {
  fs.mkdirSync(outputDir, {recursive:true})
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const external = []
  const catalogResponses = []
  const pendingResponses = []
  page.on('request', request => { if (!request.url().startsWith(new URL(url).origin)) external.push(request.url()) })
  page.on('response', response => {
    const parsed = new URL(response.url())
    if (parsed.pathname !== '/image_item' || response.status() !== 200) return
    const pending = response.json().then(body => {
      const items = Array.isArray(body) ? body : (body.items || body.list || [])
      const headers = response.headers()
      catalogResponses.push({ url:response.url(), page:Number(parsed.searchParams.get('page')), size:Number(parsed.searchParams.get('size')), totalCount:Number(headers['x-catalog-total']), matchingCount:headers['x-matching-total'] === undefined ? undefined : Number(headers['x-matching-total']), ids:items.map(item => (item.item || item).id) })
    })
    pendingResponses.push(pending)
  })
  const browserHeightLimit = await probeBrowserHeightLimit(page)
  await page.setViewportSize({width:1440,height:1000})
  await page.goto(url, {waitUntil:'networkidle'})
  await page.locator('.itemBlock').first().waitFor()
  assert.equal(await page.locator('.browsePagination').count(), 0)
  const catalog = await reviewCatalogScroll(page, catalogResponses, browserHeightLimit)
  const start = catalog.start
  const keyboardContinuity = await reviewKeyboardContinuity(page)
  const catalogFilters = await reviewCatalogFilters(page, catalogResponses, catalog.total)
  const headerImageSearch = await reviewHeaderImageSearch(page)
  await page.locator('#settingsToggle').focus()
  await page.keyboard.press('Enter')
  await page.locator('#settingsPanel').waitFor({state:'visible'})
  const panel = await page.locator('#settingsPanel').boundingBox()
  assert(panel.x <= 1)
  assert.equal(await page.locator('#settingsToggle').getAttribute('aria-expanded'), 'true')
  await page.keyboard.press('Escape')
  await page.locator('#settingsPanel').waitFor({state:'hidden'})
  await page.waitForFunction(() => document.activeElement?.id === 'settingsToggle')

  await setLocale(page, 'en')
  assert.equal(await page.locator('html').getAttribute('lang'), 'en')
  await page.locator('.resultCount').waitFor()
  await page.reload({waitUntil:'networkidle'})
  assert.equal(await page.locator('html').getAttribute('lang'), 'en')
  await setLocale(page, 'ja')
  await page.locator('#textSearchBox').fill('a cat')
  const searched = page.waitForResponse(response => response.url().endsWith('/search/text'))
  await page.getByRole('button',{name:'検索',exact:true}).click()
  assert.equal((await searched).status(), 200)
  await waitForCatalogWindow(page)
  await page.waitForFunction(expected => {
    const label = document.querySelector('.resultCount').textContent
    return Number(label.match(/[\d,]+/)?.[0].replaceAll(',','')) === expected
  }, Math.min(2048, catalog.total))
  const heights = []
  for (const key of ['Home','PageDown','PageDown','End','Home']) {
    await page.locator('#scroll-target').focus()
    await page.keyboard.press(key)
    if (key === 'Home') await page.waitForFunction(() => document.querySelector('#scroll-target').scrollTop <= 1)
    if (key === 'End') await page.waitForFunction(() => { const s=document.querySelector('#scroll-target'); return Math.abs(s.scrollTop + s.clientHeight - s.scrollHeight) <= 1 })
    await page.waitForTimeout(300)
    heights.push(await metrics(page))
  }
  assert.equal(new Set(heights.map(item => item.scrollHeight)).size, 1)
  assert(heights.every(item => item.cards <= 60))
  await page.locator('#scroll-target').focus()
  await page.keyboard.press('End')
  await page.locator('.resultsEnd').waitFor({state:'visible'})
  await page.waitForFunction(() => { const end=document.querySelector('.resultsEnd').getBoundingClientRect(); return end.top >= document.querySelector('#scroll-target').getBoundingClientRect().top && end.bottom <= innerHeight })
  assert.match(await page.locator('.resultsEnd').innerText(), /ここまで/)
  await page.keyboard.press('Home')
  await page.waitForFunction(() => document.querySelector('#scroll-target').scrollTop === 0)
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  await setSelectionMode(page, true)
  await page.locator('.imageSelectButton').nth(0).click()
  await page.locator('.imageSelectButton').nth(4).click({modifiers:['Shift']})
  assert.match(await page.locator('#selectionCount').innerText(), /5/)
  await setSelectionMode(page, false)
  assert.equal(await page.locator('#selectionCount').count(), 0)
  assert.equal(await page.locator('[data-selected="true"]').count(), 0)
  const detail = page.locator('.imageSelectButton').first()
  await detail.focus()
  await page.keyboard.press('Enter')
  await page.locator('.imageDialog').waitFor({state:'visible'})
  await page.keyboard.press('Escape')
  await page.locator('.imageDialog').waitFor({state:'hidden'})
  await page.waitForFunction(() => document.activeElement?.classList.contains('imageSelectButton'))
  await waitForVisibleImagesSettled(page)
  await page.screenshot({path:path.join(outputDir, 'ux-desktop.png')})

  await page.setViewportSize({width:375,height:812})
  await page.waitForFunction(() => { const s=document.querySelector('#scroll-target'); return Math.abs(s.clientHeight - (innerHeight - document.querySelector('#scroll-target').getBoundingClientRect().top)) <= 1 })
  const narrow = await metrics(page)
  assert.equal(narrow.documentHeight, narrow.viewportHeight)
  assert.equal(narrow.documentWidth, narrow.viewportWidth)
  assert(narrow.scrollBottom <= narrow.viewportHeight)
  await page.locator('#settingsToggle').click()
  await page.locator('#settingsPanel').waitFor({state:'visible'})
  await page.locator('#drawerLocale').selectOption('en')
  await page.keyboard.press('Escape')
  await page.locator('#settingsPanel').waitFor({state:'hidden'})
  assert.equal(await page.locator('html').getAttribute('lang'), 'en')
  await waitForVisibleImagesSettled(page)
  await page.screenshot({path:path.join(outputDir, 'ux-mobile.png')})
  await page.locator('#scroll-target').focus()
  await page.keyboard.press('End')
  await page.waitForFunction(() => { const end=document.querySelector('.resultsEnd').getBoundingClientRect(); return end.top >= document.querySelector('#scroll-target').getBoundingClientRect().top && end.bottom <= innerHeight })
  const narrowCatalog = await reviewNarrowCatalog(page, catalog.total)
  assert.deepEqual(errors, [])
  assert.deepEqual(external, [])
  await Promise.all(pendingResponses)
  const report = {generatedAt:new Date().toISOString(), start, catalog, catalogFilters, keyboardContinuity, headerImageSearch, browserHeightLimit, catalogRequests:catalogResponses.map(({ids,...request})=>({...request,delivered:ids.length})), searchScroll:heights, narrow, narrowCatalog, errors, externalRequests:external.length,
    checks:['single-scroll','fixed-catalog-height','wheel-fetch','arbitrary-catalog-jump','large-catalog-last-image','filtered-catalog-height','filtered-counts','empty-categories','selection-preserved','fixed-search-height','end-marker','selection-mode','shift-selection','exit-clears-selection','normal-click-enlarges','dialog-focus','header-image-search-while-scrolled','language-in-menu','locale-reload','narrow-layout','local-assets']}
  fs.writeFileSync(path.join(outputDir, 'ux-browser-review.json'), JSON.stringify(report,null,2))
  return report
}

module.exports = {runReview, runFinalChecks, metrics}
if (require.main === module) {
  ;(async () => {
    const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
    const browser = await chromium.launch({headless:true, ...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})})
    let page
    try {
      page = await browser.newPage()
      const review = process.env.UX_FINAL_CHECKS_ONLY === '1' ? runFinalChecks : runReview
      const report = await review(page, process.env.UX_BASE_URL || 'http://127.0.0.1:5001', path.join(__dirname,'../.cache/ux-review'))
      console.log(JSON.stringify({reportPath:path.join(__dirname,'../.cache/ux-review/ux-browser-review.json'),total:report.catalog?.total,checks:report.checks,errors:report.errors},null,2))
    } catch (error) {
      if (page) {
        try {
          console.error('Failure metrics:', JSON.stringify(await metrics(page)))
          console.error('Visible UI:', (await page.locator('body').innerText()).slice(0,5000))
          await page.screenshot({path:path.join(__dirname,'../.cache/ux-review/ux-failure.png')})
        } catch (_) {}
      }
      throw error
    } finally { await browser.close() }
  })().catch(error=>{console.error(error);process.exitCode=1})
}
