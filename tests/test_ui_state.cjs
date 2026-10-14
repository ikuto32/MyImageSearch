// Run with: node --test tests/test_ui_state.cjs
// Exercise the real application methods with controlled network completion order.
// These tests cover state transitions; browser checks remain necessary for Vue
// watcher scheduling, component rendering, and actual CSS geometry.
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const vm = require('node:vm')

const mainPath = path.join(__dirname, '../app/presentation/view/js/main.js')
const source = fs.readFileSync(mainPath, 'utf8').replace(
    /^import .*\r?\n/gm,
    '',
)

function deferred() {
    let resolve, reject
    const promise = new Promise((onResolve, onReject) => {
        resolve = onResolve
        reject = onReject
    })
    return { promise, resolve, reject }
}

function searchResult(ids, query = 'query', ratings = {}) {
    return {
        list: ids.map(id => ({ item: { id, name: `${id}.png`, rating: ratings[id] }, score: 1 })),
        search_query: query,
    }
}

// The first catalog response carries the totals that fix the scroll range.
function catalogItems(ids, total = ids.length, matching = total) {
    return Object.assign(ids.map(id => ({ id, name: `${id}.png`, rating: 'general' })),
        { totalCount: total, matchingCount: matching })
}

function createApp(overrides = {}, browser = {}) {
    let options
    let now = 0
    const scrollArea = { scrollTop: 0 }
    const repository = {
        MAX_UPLOAD_BYTES: 64 * 1024 * 1024,
        getModelItems: async () => [{ model_name: 'ViT-L-14', pretrained: 'openai' }],
        getImageItemsByPage: async () => catalogItems([]),
        getImageSmallUrl: id => `/image/${id}/small`,
        getImageOriginalUrl: id => `/image/${id}/original`,
        ...overrides,
    }
    vm.runInNewContext(source, {
        repository,
        i18n: {
            ready: Promise.resolve(),
            getInitialLocale: () => 'ja',
            setLocale() {},
            vuetifyMessages: {},
            t: (_locale, key, params = {}) => `translated:${key}${Object.keys(params).length ? JSON.stringify(params) : ''}`,
        },
        Vue: {
            markRaw: value => value,
            nextTick: async () => {},
            createApp: value => {
                options = value
                return { use() {}, mount() {} }
            },
        },
        Vuetify: { createVuetify: () => ({ locale: { current: { value: 'ja' } } }) },
        document: { getElementById: id => id === 'scroll-target' ? scrollArea : null },
        performance: { now: () => now },
        requestAnimationFrame: callback => callback(),
        console,
        ...browser,
    }, { filename: mainPath })

    const state = options.data()
    for (const [name, method] of Object.entries(options.methods)) {
        state[name] = method.bind(state)
    }
    for (const [name, getter] of Object.entries(options.computed || {})) {
        Object.defineProperty(state, name, { get: () => getter.call(state) })
    }
    state.isInitializing = false
    return {
        state,
        repository,
        scrollArea,
        watch: (name, value) => options.watch[name].call(state, value),
        setTime: value => { now = value },
    }
}

const displayedIds = state => Array.from(state.displayItems, item => item.id)
const nextTurn = () => new Promise(resolve => setImmediate(resolve))

test('ordinary image clicks enlarge and only explicit selection mode selects images', async () => {
    const { state } = createApp({ getImageMetadata: async () => ({ tags: '' }) })
    await state.runSearch(async () => searchResult(['a', 'b']))
    const image = state.displayItems[0]
    state.onImageClick(image)
    assert.equal(state.activeDialogItem.id, 'a')
    assert.equal(state.selectedCount, 0)
    assert.equal(state.selectMode, false)
    state.activeDialogItem = null
    await state.enterSelectionMode()
    await state.onImageClick(image)
    assert.equal(state.activeDialogItem, null)
    assert.equal(state.selectedItemId.a, true)
    await state.exitSelectionMode()
    assert.equal(state.selectedCount, 0)
    assert.equal(state.selectMode, false)
    assert.equal(state.selectionMessage, '')
    state.onImageClick(image)
    assert.equal(state.activeDialogItem.id, 'a')
})

test('header image search directly uses selected images and returns to browsing on success', async () => {
    const requests = []
    const { state, scrollArea } = createApp({ search: async (kind, params) => {
        requests.push([kind, Array.from(params.id)])
        return searchResult(['similar'])
    } })
    await state.runSearch(async () => searchResult(['a']))
    await state.enterSelectionMode()
    await state.onImageClick(state.displayItems[0])
    scrollArea.scrollTop = 50_000
    await state.imageSearchAction()
    assert.deepEqual(requests, [['image', ['a']]])
    assert.equal(state.selectedCount, 0)
    assert.equal(state.selectMode, false)
    assert.equal(state.displayItems[0].id, 'similar')
})

test('unselected header image search opens the native picker and cancelling starts no search', async () => {
    let clicks = 0, uploads = 0
    const picker = { value: 'previous', click() { clicks++ } }
    const { state } = createApp({ searchUploadImage: async () => { uploads++; return searchResult(['upload']) } }, {
        document: { getElementById: id => id === 'imageSearchFileInput' ? picker : null },
    })
    state.imageSearchAction()
    assert.equal(clicks, 1)
    assert.equal(picker.value, '')
    state.onImageSearchFileChange({ target: { files: [], value: '' } })
    assert.equal(uploads, 0)
    const file = { name: 'reference.png', size: 100 }
    await state.onImageSearchFileChange({ target: { files: [file], value: 'reference.png' } })
    assert.equal(uploads, 1)
    assert.equal(state.uploadFile, null)
    assert.equal(state.displayItems[0].id, 'upload')
})

test('header image search rejects more than 64 selections without changing selection', async () => {
    let requests = 0
    const { state } = createApp({ search: async () => { requests++; return searchResult([]) } })
    state.selectMode = true
    state.selectedItemId = Object.fromEntries(Array.from({ length: 65 }, (_, i) => [`${i}`, true]))
    await state.imageSearchAction()
    assert.equal(requests, 0)
    assert.equal(state.selectedCount, 65)
    assert.equal(state.selectMode, true)
    assert.equal(state.errorMessageKey, 'errors.imageSearchLimit')
})

test('Escape from the action toolbar exits selection mode without moving the image viewport', async () => {
    const { state, scrollArea } = createApp()
    state.selectMode = true
    state.selectedItemId = { a: true }
    scrollArea.scrollTop = 1000
    let prevented = false
    state.onCatalogKeydown({ key: 'Escape', target: { tagName: 'BUTTON' }, preventDefault() { prevented = true } })
    assert.equal(prevented, true)
    assert.equal(state.selectMode, false)
    assert.equal(state.selectedCount, 0)
    assert.equal(scrollArea.scrollTop, 1000)
})

function createKnownCatalog(total = 7_099_334, overrides = {}) {
    const requested = []
    const catalogPage = (page, size, _model, _pretrained, includeTotal) => {
        requested.push(page)
        const items = Array.from({ length: Math.max(0, Math.min(size, total - page * size)) },
            (_, i) => ({ id: `${page * size + i}`, name: `${page * size + i}.png`, rating: 'general' }))
        if (includeTotal) { items.totalCount = total; items.matchingCount = total }
        return items
    }
    const app = createApp({ getImageItemsByPage: async (...args) => catalogPage(...args), ...overrides })
    Object.assign(app.scrollArea, { clientHeight: 800, scrollHeight: 8_000_000 })
    return { ...app, requested, catalogPage }
}

test('known catalog reserves its final height before additional pages arrive', async () => {
    const { state, scrollArea, requested } = createKnownCatalog()
    await state.browseImages()
    const height = state.catalogPhysicalHeight
    assert.equal(state.searchCount, 7_099_334)
    assert.equal(height, 8_000_000)
    assert.equal(state.resultBuffer.length, 60)
    state.selectedItemId = { '0': true }
    scrollArea.scrollTop = height / 2
    state.updateImageFromScroll({ target: scrollArea })
    assert(state.displayItems.some(item => item.placeholder))
    const during = state.padding_top + Math.ceil(state.displayItems.length / 6) * 282 + state.padding_bottom
    await state.ensureCatalogWindow()
    const after = state.padding_top + Math.ceil(state.displayItems.length / 6) * 282 + state.padding_bottom
    assert(Math.abs(during - height) < 1)
    assert(Math.abs(after - height) < 1)
    assert(state.displayItems.every(item => !item.placeholder))
    assert.equal(state.selectedItemId['0'], true)
    assert(requested.length <= 3)
    assert(requested.some(page => page > 50_000))
})

test('known catalog geometry reaches every final row without exceeding native layout limits', async () => {
    for (const total of [0, 59, 60, 61, 2048, 7_099_334]) {
        const { state, scrollArea } = createKnownCatalog(total)
        await state.browseImages()
        state.isSearching = true // Isolate scroll geometry from subsequent network work.
        for (const columns of [1, 2, 6, 24]) {
            state.numCols = columns
            for (const fraction of [0, .25, .5, .9, 1]) {
                scrollArea.scrollTop = Math.max(0, state.catalogPhysicalHeight - 800) * fraction
                state.updateImageFromScroll({ target: scrollArea })
                const actual = state.padding_top + Math.ceil(state.displayItems.length / columns) * 282 + state.padding_bottom
                assert(Math.abs(actual - state.catalogPhysicalHeight) < 1, `${total}/${columns}/${fraction}: ${actual}`)
                assert(actual <= 8_000_001)
                assert(state.displayItems.length <= columns * state.numRows)
                if (fraction === 1 && total) assert.equal(state.displayItems.at(-1).position, total - 1)
            }
        }
    }
})

test('wheel and page keys retain normal pixel movement on a compressed rail', async () => {
    const { state, scrollArea } = createKnownCatalog()
    await state.browseImages()
    let prevented = 0
    state.onCatalogWheel({ deltaY: 1200, deltaX: 0, deltaMode: 0, preventDefault() { prevented++ } })
    assert.equal(prevented, 1)
    assert(Math.abs(state.catalogPhysicalOffset * state.catalogScrollScale - 1200) < 1)
    const before = state.catalogPhysicalOffset * state.catalogScrollScale
    state.onCatalogKeydown({ key: 'PageDown', target: { tagName: 'DIV' }, preventDefault() { prevented++ } })
    assert(Math.abs(state.catalogPhysicalOffset * state.catalogScrollScale - before - 518) < 1)
    state.onCatalogKeydown({ key: 'End', target: { tagName: 'DIV' }, preventDefault() {} })
    await state.ensureCatalogWindow()
    assert.equal(state.displayItems.at(-1).id, '7099333')
    state.onCatalogKeydown({ key: 'Home', target: { tagName: 'DIV' }, preventDefault() {} })
    assert.equal(scrollArea.scrollTop, 0)
})

test('known catalog retries a failed viewport without changing extent or offset', async () => {
    const fixture = createKnownCatalog(2048)
    const { state, scrollArea, repository } = fixture
    await state.browseImages()
    let fail = true
    repository.getImageItemsByPage = async (...args) => {
        if (fail) throw new Error('temporary page failure')
        return fixture.catalogPage(...args)
    }
    scrollArea.scrollTop = 4000
    state.updateImageFromScroll({ target: scrollArea })
    await state.ensureCatalogWindow()
    assert(state.catalogPageError)
    const height = state.catalogPhysicalHeight
    const position = state.showedItemIndex
    fail = false
    await state.retryCatalogPage()
    assert.equal(state.catalogPageError, '')
    assert.equal(state.catalogPhysicalHeight, height)
    assert.equal(state.showedItemIndex, position)
    assert.equal(scrollArea.scrollTop, 4000)
    assert(state.displayItems.every(item => !item.placeholder))
})

test('known catalog stale viewport responses cannot overwrite a new search', async () => {
    const page = deferred()
    const fixture = createKnownCatalog()
    await fixture.state.browseImages()
    fixture.repository.getImageItemsByPage = () => page.promise
    fixture.scrollArea.scrollTop = 4_000_000
    fixture.state.updateImageFromScroll({ target: fixture.scrollArea })
    const loading = fixture.state.ensureCatalogWindow()
    await fixture.state.runSearch(async () => searchResult(['new']))
    page.resolve([{ id: 'old', name: 'old.png' }])
    await loading
    assert.deepEqual(displayedIds(fixture.state), ['new'])
    assert.equal(fixture.state.isLoadingPage, false)
})

test('shift selection resolves unloaded intermediate pages in logical order', async () => {
    const { state, requested } = createKnownCatalog(2048)
    await state.browseImages()
    await state.toggleSelection('0')
    await state.loadCatalogPage(5)
    await state.toggleSelection('300', { shiftKey: true })
    assert.equal(state.selectedCount, 301)
    for (let i = 0; i <= 300; i++) assert.equal(state.selectedItemId[`${i}`], true)
    assert(requested.includes(1) && requested.includes(4))
    await state.loadCatalogPage(30)
    const before = requested.length
    await state.toggleSelection('1800', { shiftKey: true })
    assert.equal(requested.length, before)
    assert.equal(state.selectedCount, 301)
    assert.equal(state.selectionMessageKey, 'selection.limit')
})

test('bulk selection resolves first 1024 independent of the cached viewport', async () => {
    const { state, scrollArea } = createKnownCatalog()
    await state.browseImages()
    scrollArea.scrollTop = 7_999_200
    state.updateImageFromScroll({ target: scrollArea })
    await state.ensureCatalogWindow()
    await state.selectCurrentResults()
    assert.equal(state.selectedCount, 1024)
    assert.equal(state.selectedItemId['0'], true)
    assert.equal(state.selectedItemId['1023'], true)
    assert.equal(state.selectedItemId['7099333'], undefined)
})

test('page cache eviction keeps selection and the scrollbar fixed', async () => {
    const { state } = createKnownCatalog()
    await state.browseImages()
    await state.toggleSelection('0')
    for (let page = 1; page < 70; page++) {
        state.showedItemIndex = page * 60
        await state.loadCatalogPage(page)
    }
    assert.equal(Object.keys(state.catalogPages).length, 64)
    assert(state.resultBuffer.length <= 3840)
    assert.equal(state.catalogPages[0], undefined)
    assert.equal(state.selectedItemId['0'], true)
    assert.equal(state.selectionAnchorPosition, 0)
    assert.equal(state.catalogPhysicalHeight, 8_000_000)
})

test('compressed rendering preserves the negative offset near the first rows', async () => {
    const { state, scrollArea } = createKnownCatalog()
    await state.browseImages()
    for (const physical of [1, 10, 20, 100, 1000]) {
        scrollArea.scrollTop = physical
        state.updateImageFromScroll({ target: scrollArea })
        const logical = physical * state.catalogScrollScale
        const firstRow = Math.floor(state.showedItemIndex / state.visibleCols)
        const expectedScreenY = firstRow * state.item_height - logical
        const actualScreenY = state.padding_top + state.catalogRenderShift - physical
        assert(Math.abs(expectedScreenY - actualScreenY) < 1)
    }
})

test('superseded viewport failures do not stop the current viewport', async () => {
    const fixture = createKnownCatalog(2048)
    await fixture.state.browseImages()
    const old = deferred()
    fixture.repository.getImageItemsByPage = async (...args) => args[0] === 1 ? old.promise : fixture.catalogPage(...args)
    fixture.state.showedItemIndex = 60
    const loading = fixture.state.ensureCatalogWindow()
    fixture.state.showedItemIndex = 180
    old.reject(new Error('old viewport unavailable'))
    await loading
    assert.equal(fixture.state.catalogPageError, '')
    assert(fixture.requested.includes(3))
    assert.equal(fixture.state.displayItems[0].id, '180')
})

test('changing category again while its count is loading rejects the old category response', async () => {
    const fixture = createKnownCatalog(100)
    await fixture.state.browseImages()
    const old = deferred()
    const calls = []
    fixture.repository.getImageItemsByPage = async (...args) => {
        calls.push(Array.from(args[5]))
        if (args[5][0] === 'general') return old.promise
        const items = [{ id: 'explicit', name: 'explicit.png', rating: 'explicit' }]
        items.matchingCount = 1
        items.totalCount = 100
        return items
    }
    fixture.state.ratingFilter = ['general']
    const first = fixture.watch('ratingFilter')
    fixture.state.ratingFilter = ['explicit']
    await fixture.watch('ratingFilter')
    old.resolve(fixture.catalogPage(0, 60, '', '', true))
    await first
    assert.deepEqual(calls, [['general'], ['explicit']])
    assert.equal(fixture.state.searchCount, 1)
    assert.equal(fixture.state.displayItems[0].id, 'explicit')
})

test('Escape cancels pending bulk selection and responses do not restore it', async () => {
    const fixture = createKnownCatalog(100)
    await fixture.state.browseImages()
    const next = deferred()
    fixture.repository.getImageItemsByPage = () => next.promise
    const selecting = fixture.state.selectCurrentResults()
    await nextTurn()
    fixture.state.onGridKeydown({ key: 'Escape', preventDefault() {} })
    next.resolve(fixture.catalogPage(1, 60))
    await selecting
    assert.equal(fixture.state.selectedCount, 0)
    assert.equal(fixture.state.isSelecting, false)
})

test('an unrelated action error does not block catalog loading', async () => {
    const { state, scrollArea } = createKnownCatalog(2048)
    await state.browseImages()
    state.showError(new Error('ZIP disk unavailable'))
    scrollArea.scrollTop = 6000
    state.updateImageFromScroll({ target: scrollArea })
    await state.ensureCatalogWindow()
    assert(state.errorMessage)
    assert(state.displayItems.every(item => !item.placeholder))
    assert.equal(state.catalogPageError, '')
})

test('Space on the scroller moves a logical page and leaves checkbox activation native', async () => {
    const { state } = createKnownCatalog()
    await state.browseImages()
    let prevented = 0
    state.onCatalogKeydown({ key: ' ', target: { id: 'scroll-target', tagName: 'DIV' }, preventDefault() { prevented++ } })
    assert.equal(prevented, 1)
    assert(Math.abs(state.catalogPhysicalOffset * state.catalogScrollScale - 518) < 1)
    state.onCatalogKeydown({ key: ' ', target: { tagName: 'BUTTON' }, preventDefault() { prevented++ } })
    assert.equal(prevented, 1)
})

test('logical increments survive native integer scrollTop rounding at massive row counts', async () => {
    const { state, scrollArea } = createKnownCatalog()
    await state.browseImages()
    state.numCols = 1
    let nativeTop = 0
    Object.defineProperty(scrollArea, 'scrollTop', { get: () => nativeTop, set: value => { nativeTop = Math.round(value) } })
    for (let i = 1; i <= 5; i++) {
        state.onCatalogKeydown({ key: 'ArrowDown', target: { tagName: 'DIV' }, preventDefault() {} })
        assert.equal(state.catalogLogicalOffset, i * 40)
        const firstRowY = Math.floor(state.showedItemIndex / state.visibleCols) * state.item_height
        assert(Math.abs(state.padding_top + state.catalogRenderShift - nativeTop - firstRowY + i * 40) < 1)
    }
    scrollArea.scrollTop = 4000
    state.updateImageFromScroll({ target: scrollArea })
    assert.equal(state.catalogLogicalOffset, 4000 * state.catalogScrollScale)
})

test('search is blocked during initialization and works after initialization', async () => {
    const models = deferred()
    let searches = 0
    const { state } = createApp({
        getModelItems: () => models.promise,
        search: async () => { searches += 1; return searchResult(['cat']) },
    })
    const initialization = state.init()
    state.text = 'cat'
    await state.textSearchButton()
    assert.equal(searches, 0)
    assert.equal(state.isSearching, false)

    models.resolve([{ model_name: 'ViT-L-14', pretrained: 'openai' }])
    await initialization
    await state.textSearchButton()
    assert.equal(searches, 1)
    assert.equal(state.isInitializing, false)
    assert.equal(state.isSearching, false)
    assert.deepEqual(displayedIds(state), ['cat'])
})

test('an old browse response preserves newer search results, selection, and scroll', async () => {
    const page = deferred()
    const { state, scrollArea } = createApp({ getImageItemsByPage: () => page.promise })
    const browsing = state.browseImages()
    await state.runSearch(async () => searchResult(['new']))
    state.selectedItemId = { new: true }
    scrollArea.scrollTop = 500

    page.resolve([{ id: 'old', name: 'old.png' }])
    await browsing
    assert.deepEqual(displayedIds(state), ['new'])
    assert.equal(state.selectedItemId.new, true)
    assert.equal(scrollArea.scrollTop, 500)
    assert.equal(state.isPagedBrowseMode, false)
})

test('old search completion cannot change a newer search or its busy state', async () => {
    const oldResponse = deferred()
    const newResponse = deferred()
    const { state, setTime } = createApp()
    const oldSearch = state.runSearch(() => oldResponse.promise)
    await nextTurn()
    state.model_name = 'new-model'
    await state.onModelChange()
    setTime(100)
    const newSearch = state.runSearch(() => newResponse.promise)
    await nextTurn()

    setTime(200)
    oldResponse.resolve(searchResult(['old'], 'old-query'))
    await oldSearch
    assert.equal(state.isSearching, true)
    assert.equal(state.searchDurationMs, null)
    assert.deepEqual(displayedIds(state), [])

    setTime(250)
    newResponse.resolve(searchResult(['new'], 'new-query'))
    await newSearch
    assert.equal(state.isSearching, false)
    assert.equal(state.searchDurationMs, 150)
    assert.equal(state.search_query, 'new-query')
    assert.deepEqual(displayedIds(state), ['new'])
})

test('an old search failure cannot replace the current result with an error', async () => {
    const oldResponse = deferred()
    const { state } = createApp()
    const oldSearch = state.runSearch(() => oldResponse.promise)
    await nextTurn()
    state.model_name = 'new-model'
    await state.onModelChange()
    await state.runSearch(async () => searchResult(['new']))
    oldResponse.reject(new Error('obsolete failure'))
    await oldSearch

    assert.equal(state.errorMessage, '')
    assert.equal(state.isSearching, false)
    assert.deepEqual(displayedIds(state), ['new'])
})

test('a failed search shows the API error, preserves results, and permits retry', async () => {
    const { state } = createApp()
    await state.runSearch(async () => searchResult(['previous'], 'previous-query'))
    await state.runSearch(async () => {
        throw { response: { data: { error: 'text is not a valid regular expression' } } }
    })
    assert.equal(state.errorDetail, 'text is not a valid regular expression')
    assert.equal(state.errorMessage, 'translated:errors.generic')
    assert.equal(state.isSearching, false)
    assert.deepEqual(displayedIds(state), ['previous'])
    assert.equal(state.search_query, 'previous-query')

    await state.runSearch(async () => searchResult(['retry'], 'retry-query'))
    assert.equal(state.errorMessage, '')
    assert.equal(state.isSearching, false)
    assert.deepEqual(displayedIds(state), ['retry'])
})

test('a failed search preserves previous buffers, selection, and upload state', async () => {
    const { state } = createApp()
    await state.runSearch(async () => searchResult(['previous'], 'previous-query'))
    state.selectedItemId = { previous: true }
    state.uploadFile = { name: 'upload.png' }
    await state.runSearch(
        async () => { throw new Error('search request failed') },
        () => { state.uploadFile = null },
    )

    assert.equal(state.errorDetail, 'search request failed')
    assert.equal(state.isSearching, false)
    assert.deepEqual(displayedIds(state), ['previous'])
    assert.deepEqual(Array.from(state.rawResultBuffer, result => result.item.id), ['previous'])
    assert.equal(state.search_query, 'previous-query')
    assert.equal(state.selectedItemId.previous, true)
    assert.equal(state.uploadFile.name, 'upload.png')
})

test('a failed first page shows its error and browsing again retries it', async () => {
    const requestedPages = []
    const { state } = createApp({
        getImageItemsByPage: async page => {
            requestedPages.push(page)
            if (requestedPages.length === 1) throw new Error('offline')
            return catalogItems(['first'])
        },
    })
    await state.browseImages()
    assert.equal(state.isLoadingPage, false)
    assert.equal(state.errorDetail, 'offline')
    assert.equal(state.stableCatalog, false)
    assert.equal(state.scrollProgressPercent, null)
    assert.deepEqual(displayedIds(state), [])
    // A failed collection does not block searching.
    await state.runSearch(async () => searchResult(['found']))
    assert.deepEqual(displayedIds(state), ['found'])

    await state.browseImages()
    assert.deepEqual(requestedPages, [0, 0])
    assert.equal(state.errorMessage, '')
    assert.equal(state.stableCatalog, true)
    assert.deepEqual(displayedIds(state), ['first'])
})

test('a collection response without totals is reported instead of leaving an endless list', async () => {
    const { state } = createApp({ getImageItemsByPage: async () => [{ id: 'image', name: 'image.png' }] })
    await state.browseImages()
    assert.match(state.errorDetail, /totals/)
    assert.equal(state.stableCatalog, false)
    assert.equal(state.isLoadingPage, false)
    assert.deepEqual(displayedIds(state), [])
})

test('scrolling reaches a partial last row with no phantom bottom padding', async () => {
    const { state } = createApp()
    await state.runSearch(async () => searchResult(Array.from({ length: 65 }, (_, i) => `${i}`)))
    assert.equal(state.displayItems.length, 60)
    assert.equal(state.padding_top, 0)
    assert.equal(state.padding_bottom, 282)

    state.updateImageFromScroll({ target: { scrollTop: 1_000_000 } })
    assert.equal(state.showedItemIndex, 6)
    assert.equal(state.padding_top, 282)
    assert.equal(state.padding_bottom, 0)
    assert.equal(state.displayItems.length, 59)
    assert.equal(state.displayItems.at(-1).id, '64')
})

test('column changes reset scroll and preserve reachable results with valid padding', async () => {
    const { state, scrollArea, watch } = createApp()
    await state.runSearch(async () => searchResult(Array.from({ length: 65 }, (_, i) => `${i}`)))
    state.updateImageFromScroll({ target: { scrollTop: 1_000_000 } })
    scrollArea.scrollTop = 1_000_000
    state.selectedItemId = { '64': true }
    watch('numCols', '4')

    assert.equal(state.numCols, 4)
    assert.equal(scrollArea.scrollTop, 0)
    assert.equal(state.showedItemIndex, 0)
    assert.equal(state.displayItems.length, 40)
    assert.equal(state.padding_top, 0)
    assert.equal(state.padding_bottom, 7 * 282)
    assert.equal(state.selectedItemId['64'], true)
    state.updateImageFromScroll({ target: { scrollTop: 1_000_000 } })
    assert.equal(state.displayItems.at(-1).id, '64')
    assert.equal(state.padding_bottom, 0)

    watch('numCols', '0')
    assert.equal(state.numCols, 1)
    watch('numCols', '1000')
    assert.equal(state.numCols, 24)
    assert.ok(Number.isFinite(state.padding_bottom))
})

test('responsive columns fit narrow screens and restore the saved user preference', async () => {
    const { state } = createApp()
    await state.runSearch(async () => searchResult(Array.from({ length: 125 }, (_, i) => `${i}`)))
    state.selectedItemId = { '20': true }
    state.updateGridWidth(343)
    assert.equal(state.getColumnCount(), 2)
    assert.equal(state.numCols, 6)
    assert.equal(state.displayItems.length, 20)
    assert.equal(state.selectedItemId['20'], true)
    state.updateImageFromScroll({ target: { scrollTop: 1_000_000 } })
    assert.equal(state.displayItems.at(-1).id, '124')
    assert.equal(state.padding_bottom, 0)

    state.updateGridWidth(1440)
    assert.equal(state.getColumnCount(), 6)
    assert.equal(state.selectedItemId['20'], true)
    state.updateGridWidth(100)
    assert.equal(state.getColumnCount(), 1)
    assert.ok(Number.isFinite(state.padding_bottom))
})

test('zoom and user column changes do not change pagination offsets', async () => {
    const pages = []
    const { state, watch } = createApp({
        getImageItemsByPage: async (page, size) => {
            pages.push([page, size])
            return catalogItems(Array.from({ length: size }, (_, index) => `${page * size + index}`), 120)
        },
    })
    state.updateGridWidth(343)
    await state.browseImages()
    state.updateGridWidth(1400)
    watch('numCols', 2)
    await state.loadCatalogPage(1)
    assert.deepEqual(pages, [[0, 60], [1, 60]])
    assert.equal(state.rawResultBuffer.length, 120)
    assert.equal(new Set(state.rawResultBuffer.map(result => result.item.id)).size, 120)
    assert.equal(state.rawResultBuffer[0].item.id, '0')
    assert.equal(state.getColumnCount(), 2)
})

test('failed model initialization can be retried without refreshing the browser', async () => {
    let requests = 0
    const { state } = createApp({
        getModelItems: async () => {
            if (++requests === 1) throw new Error('offline')
            return [{ model_name: 'available-model', pretrained: 'weights' }]
        },
        getImageItemsByPage: async () => catalogItems(['image']),
    })
    await state.init()
    assert.equal(state.initializationFailed, true)
    assert.equal(state.isInitializing, false)
    let searches = 0
    await state.runSearch(async () => { searches += 1; return searchResult([]) })
    assert.equal(searches, 0)
    await state.browseImages()
    assert.equal(requests, 2)
    assert.equal(state.initializationFailed, false)
    assert.equal(state.errorMessage, '')
    assert.equal(state.model_name, 'available-model')
    assert.equal(state.pretrained, 'weights')
    assert.deepEqual(displayedIds(state), ['image'])
})

test('an empty model list offers recovery instead of searching with a nonexistent model', async () => {
    const { state } = createApp({ getModelItems: async () => [] })
    await state.init()
    assert.equal(state.initializationFailed, true)
    assert.equal(state.errorMessage, 'translated:errors.noModels')
    assert.equal(state.isInitializing, false)
})

test('native selection buttons toggle selection without opening the image dialog', () => {
    const { state } = createApp()
    state.toggleSelection('image')
    assert.equal(state.selectedItemId.image, true)
    assert.equal(state.activeDialogItem, null)
    state.toggleSelection('image')
    assert.equal(state.selectedItemId.image, undefined)
})

test('ZIP download uses selected images, rejects excess counts, and prevents duplicates', async () => {
    const response = deferred()
    const requested = []
    const revoked = []
    let clicked = 0
    let removed = 0
    const { state } = createApp({}, {
        fetch: async (_url, options) => {
            requested.push(JSON.parse(options.body).params.ids)
            return response.promise
        },
        document: {
            getElementById: () => null,
            body: { appendChild() {} },
            createElement: () => ({ click() { clicked += 1 }, remove() { removed += 1 } }),
        },
        URL: { createObjectURL: () => 'blob:archive', revokeObjectURL: url => revoked.push(url) },
        setTimeout: callback => callback(),
    })
    state.selectedItemId = Object.fromEntries(Array.from({ length: 1025 }, (_, i) => [`${i}`, true]))
    await state.allDownloadImagesButton()
    assert.equal(requested.length, 0)
    assert.equal(state.errorMessage, 'translated:errors.selectionLimit')

    state.selectedItemId = { '2': true, '15': true }
    state.allDownloadImagesButton()
    assert.equal(requested.length, 0)
    assert.equal(state.downloadConfirmation.count, 2)
    const download = state.confirmDownload()
    await state.confirmDownload()
    await state.allDownloadImagesButton()
    assert.equal(state.isDownloading, true)
    assert.deepEqual(requested, [['2', '15']])
    response.resolve({ ok: true, json: async () => ({ download_url: '/downloads/token', image_count: 2, skipped_count: 0 }) })
    await download
    assert.equal(state.isDownloading, false)
    assert.equal(state.errorMessage, '')
    assert.equal(clicked, 1)
    assert.equal(removed, 1)
    assert.deepEqual(revoked, [])
    assert.match(state.downloadMessage, /download.started/)
})

test('cancelling a ZIP confirmation sends nothing and preserves selection and browsing position', async () => {
    let requests = 0
    const { state, scrollArea } = createApp({}, { fetch: async () => { requests++ } })
    state.selectMode = true
    state.selectedItemId = { chosen: true }
    scrollArea.scrollTop = 1200
    state.allDownloadImagesButton()
    assert.equal(state.downloadConfirmation.count, 1)
    assert.equal(state.downloadConfirmation.selected, true)
    assert.equal(state.isDownloading, false)
    assert.equal(requests, 0)
    state.cancelDownloadConfirmation()
    await state.confirmDownload()
    assert.equal(requests, 0)
    assert.equal(state.downloadConfirmation, null)
    assert.equal(state.selectedItemId.chosen, true)
    assert.equal(state.selectMode, true)
    assert.equal(scrollArea.scrollTop, 1200)
})

test('ZIP confirmation freezes the selected IDs rather than saving a later selection', async () => {
    const requests = []
    const { state } = createApp({}, { fetch: async (_url, options) => {
        requests.push(JSON.parse(options.body).params)
        return { ok: false, json: async () => ({ error_code: 'downloadFailed' }) }
    } })
    state.selectedItemId = { first: true, second: true }
    state.allDownloadImagesButton()
    state.selectedItemId = { different: true }
    state.allDownloadImagesButton()
    assert.equal(state.downloadConfirmation.count, 2)
    await state.confirmDownload()
    assert.deepEqual(requests, [{ ids: ['first', 'second'] }])
})

test('Escape closes a just-reopened modal before focus enters it without clearing the background selection', () => {
    const { state, scrollArea } = createApp()
    state.selectMode = true
    state.selectedItemId = { chosen: true }
    scrollArea.scrollTop = 1200
    let stopped = 0, prevented = 0
    const escape = { key: 'Escape', preventDefault() { prevented++ }, stopImmediatePropagation() { stopped++ } }
    state.activeDialogItem = { id: 'just-reopened' }
    state.onModalKeydown(escape)
    assert.equal(state.activeDialogItem, null)
    assert.equal(state.selectedItemId.chosen, true)
    assert.equal(state.selectMode, true)
    assert.equal(scrollArea.scrollTop, 1200)
    state.allDownloadImagesButton()
    state.onModalKeydown(escape)
    assert.equal(state.downloadConfirmation, null)
    assert.equal(state.isDownloading, false)
    assert.equal(stopped, 2)
    assert.equal(prevented, 2)
    // Other menus retain their own Escape behavior.
    state.isShowSetting = true
    state.onModalKeydown(escape)
    assert.equal(state.isShowSetting, true)
    assert.equal(stopped, 2)
})

test('unselected ZIP confirmation freezes the model and filters visible when opened', async () => {
    const requests = []
    const { state } = createApp({}, { fetch: async (_url, options) => {
        requests.push(JSON.parse(options.body).params)
        return { ok: false, json: async () => ({ error_code: 'downloadFailed' }) }
    } })
    state.catalogMatchingCount = 88
    state.ratingFilter = ['general']
    state.allDownloadImagesButton()
    assert.equal(state.downloadConfirmation.count, 88)
    assert.equal(state.downloadConfirmation.selected, false)
    state.model_name = 'another-model'
    state.ratingFilter.push('sensitive')
    await state.confirmDownload()
    assert.deepEqual(requests, [{ first: 1024, model_name: 'ViT-L-14', pretrained: 'openai', ratings: ['general'] }])
})

test('detail loading and retry ignore an earlier image or an earlier attempt', async () => {
    const { state } = createApp({ getImageMetadata: async () => ({ tags: '' }) })
    state.openDialog({ id: 'a' })
    const first = state.detailImageAttempt
    assert.equal(state.detailImageState, 'loading')
    state.activeDialogItem = null
    state.openDialog({ id: 'b' })
    const second = state.detailImageAttempt
    state.onDetailImageError('a', first)
    assert.equal(state.detailImageState, 'loading')
    state.onDetailImageError('b', second)
    assert.equal(state.detailImageState, 'error')
    state.retryDetailImage()
    const retry = state.detailImageAttempt
    assert.equal(state.detailImageState, 'loading')
    state.onDetailImageLoad('b', second)
    state.onDetailImageError('b', second)
    assert.equal(state.detailImageState, 'loading')
    state.onDetailImageLoad('b', retry)
    assert.equal(state.detailImageState, 'loaded')
    state.activeDialogItem = null
    state.openDialog({ id: 'b' })
    state.onDetailImageLoad('b', retry)
    assert.equal(state.detailImageState, 'loading')
    state.onDetailImageLoad('b', state.detailImageAttempt)
    assert.equal(state.detailImageState, 'loaded')
})

test('ZIP errors display backend detail and allow the user to retry', async () => {
    let requests = 0
    const { state } = createApp({}, {
        fetch: async () => {
            requests += 1
            return { ok: false, json: async () => ({ error: '画像の合計が256MiBを超えています。件数を減らしてください。' }) }
        },
    })
    state.resultBuffer = searchResult(['first', 'second']).list
    state.isPagedBrowseMode = false
    assert.deepEqual(Array.from(state.getDownloadIds()), ['first', 'second'])
    await state.allDownloadImagesButton()
    await state.confirmDownload()
    assert.match(state.errorDetail, /合計が256MiB/)
    assert.equal(state.isDownloading, false)
    await state.allDownloadImagesButton()
    await state.confirmDownload()
    assert.equal(requests, 2)
    assert.equal(state.isDownloading, false)
})

test('unclassified images are explicit and clearing all rating filters hides every image', async () => {
    const requests = []
    const { state, watch } = createApp({ search: async kind => {
        requests.push(kind)
        return searchResult(['general', 'empty', 'unknown', 'legacy', 'missing'], 'query',
            { general: 'general', empty: '', unknown: 'unknown', legacy: 'safe' })
    } })
    await state.randomSearchButton()
    assert.deepEqual(displayedIds(state), ['general', 'empty', 'unknown', 'legacy', 'missing'])

    state.ratingFilter = ['general']
    watch('ratingFilter')
    assert.deepEqual(displayedIds(state), ['general'])

    state.ratingFilter = ['unclassified']
    watch('ratingFilter')
    assert.deepEqual(displayedIds(state), ['empty', 'unknown', 'legacy', 'missing'])

    state.ratingFilter = []
    watch('ratingFilter')
    assert.deepEqual(displayedIds(state), [])
    assert.equal(state.resultBuffer.length, 0)
    // Results carry their own category: filtering never asks the server again.
    assert.deepEqual(requests, ['random'])
})

test('changing categories while browsing asks the server for the filtered collection', async () => {
    const calls = []
    const { state, watch } = createApp({
        getImageItemsByPage: async (page, _size, _model, _pretrained, includeTotal, ratings) => {
            calls.push({ page, includeTotal, ratings: Array.from(ratings) })
            const matching = ratings.includes('explicit') ? ['explicit-image'] : []
            return catalogItems(matching, 100, matching.length)
        },
    })
    await state.browseImages()
    state.selectedItemId = { stale: true }
    state.ratingFilter = ['explicit']
    await watch('ratingFilter')
    assert.deepEqual(calls.at(-1), { page: 0, includeTotal: true, ratings: ['explicit'] })
    assert.deepEqual(displayedIds(state), ['explicit-image'])
    assert.equal(state.candidateCount, 100)
    assert.equal(state.searchCount, 1)
    assert.equal(state.selectedCount, 0)

    state.ratingFilter = []
    await watch('ratingFilter')
    assert.deepEqual(calls.at(-1).ratings, [])
    assert.equal(state.searchCount, 0)
    assert.equal(state.downloadCount, 0)
})

test('oversized uploads are rejected before upload conversion or search starts', async () => {
    let uploadCalls = 0
    const { state, repository } = createApp({
        searchUploadImage: async () => { uploadCalls += 1; return searchResult(['uploaded']) },
    })
    state.uploadFile = [{ name: 'too-large.png', size: repository.MAX_UPLOAD_BYTES + 1 }]
    await state.uploadImageSearchButton()
    assert.equal(uploadCalls, 0)
    assert.equal(state.isSearching, false)
    assert.equal(state.errorMessage, 'translated:errors.uploadTooLarge')
    assert.equal(state.uploadFile[0].name, 'too-large.png')

    state.uploadFile = { name: 'at-limit.png', size: repository.MAX_UPLOAD_BYTES }
    await state.uploadImageSearchButton()
    assert.equal(uploadCalls, 1)
    assert.equal(state.errorMessage, '')
    assert.equal(state.uploadFile, null)
})

test('file conversion rejects oversized input without constructing FileReader', async () => {
    const utilPath = path.join(__dirname, '../app/presentation/view/js/modules/util.js')
    const utilSource = fs.readFileSync(utilPath, 'utf8').replace(/^export /gm, '')
    let readers = 0
    let reads = 0
    const context = {
        FileReader: class {
            constructor() { readers += 1 }
            readAsDataURL() {
                reads += 1
                this.result = 'data:image/png;base64,bytes'
                this.onload()
            }
        },
    }
    vm.runInNewContext(`${utilSource}\nglobalThis.exports = { fileToBase64, MAX_UPLOAD_BYTES }`, context)
    await assert.rejects(context.exports.fileToBase64({ size: context.exports.MAX_UPLOAD_BYTES + 1 }), /64MiB/)
    assert.equal(readers, 0)
    assert.equal(reads, 0)
    assert.equal(await context.exports.fileToBase64({ size: context.exports.MAX_UPLOAD_BYTES }), 'data:image/png;base64,bytes')
    assert.equal(readers, 1)
    assert.equal(reads, 1)
})

test('virtual window changes preserve the result height without further requests', async () => {
    let pageRequests = 0
    const { state } = createApp({
        getImageItemsByPage: async (_page, size) => {
            pageRequests += 1
            return catalogItems(Array.from({ length: size }, (_, i) => `${i}`))
        },
    })
    await state.browseImages()
    for (const scrollTop of [0, 500, 5_000, 100_000, 0]) {
        state.updateImageFromScroll({ target: { scrollTop } })
        await state.ensureCatalogWindow()
    }
    assert.equal(pageRequests, 1)
    assert.equal(state.rawResultBuffer.length, 60)

    await state.runSearch(async () => searchResult(Array.from({ length: 2048 }, (_, i) => `${i}`)))
    const height = Math.ceil(2048 / state.visibleCols) * state.item_height
    for (const scrollTop of [0, 500, 5_000, 100_000, 200_000, 0]) {
        state.updateImageFromScroll({ target: { scrollTop } })
        const renderedHeight = Math.ceil(state.displayItems.length / state.visibleCols) * state.item_height
        assert.equal(state.padding_top + renderedHeight + state.padding_bottom, height)
    }
    assert.equal(pageRequests, 1)
})

test('candidate counts and filtered search counts remain distinct', async () => {
    const { state, watch } = createApp()
    await state.runSearch(async () => searchResult(['one', 'two', 'three'], 'query',
        { one: 'general', two: 'explicit', three: 'general' }))
    state.ratingFilter = ['general']
    watch('ratingFilter')
    assert.equal(state.candidateCount, 3)
    assert.equal(state.searchCount, 2)
})

test('shift selection covers offscreen results, and oversized ranges leave selection unchanged', async () => {
    const { state } = createApp()
    await state.runSearch(async () => searchResult(Array.from({ length: 1500 }, (_, i) => `${i}`)))
    state.toggleSelection('1')
    state.updateImageFromScroll({ target: { scrollTop: 10_000 } })
    state.toggleSelection('101', { shiftKey: true })
    assert.equal(state.selectedCount, 101)
    assert.equal(state.selectedItemId['50'], true)
    state.toggleSelection('1499', { shiftKey: true })
    assert.equal(state.selectedCount, 101)
    assert.match(state.selectionMessage, /selection.limit/)
    state.selectCurrentResults()
    assert.equal(state.selectedCount, 1024)
    assert.equal(state.selectedItemId['1023'], true)
    assert.equal(state.selectedItemId['1024'], undefined)
    state.clearSelection()
    assert.equal(state.selectedCount, 0)
    assert.match(state.selectionMessage, /selection.cleared/)
})

test('filter changes remove hidden selections and start new searches with no old selection', async () => {
    const { state, watch } = createApp()
    await state.runSearch(async () => searchResult(['one', 'two'], 'query', { one: 'general', two: 'explicit' }))
    state.selectCurrentResults()
    state.ratingFilter = ['general']
    watch('ratingFilter')
    assert.deepEqual(Object.keys(state.selectedItemId), ['one'])
    assert.match(state.selectionMessage, /"count":1/)
    await state.runSearch(async () => searchResult(['one']))
    assert.equal(state.selectedCount, 0)
})

test('grid keyboard shortcuts select a bounded result set and clear selection', async () => {
    const { state } = createApp()
    await state.runSearch(async () => searchResult(['one', 'two', 'three']))
    let prevented = 0
    state.onGridKeydown({ key: 'a', ctrlKey: true, preventDefault() { prevented += 1 } })
    assert.equal(state.selectedCount, 3)
    state.onGridKeydown({ key: 'Escape', preventDefault() { prevented += 1 } })
    assert.equal(state.selectedCount, 0)
    assert.equal(prevented, 2)
})

test('unselected download takes the first 1024 results regardless of scroll position', async () => {
    const requests = []
    const { state } = createApp({}, {
        fetch: async (_url, options) => {
            requests.push(JSON.parse(options.body).params)
            return { ok: false, json: async () => ({ error_code: 'downloadFailed' }) }
        },
    })
    await state.runSearch(async () => searchResult(Array.from({ length: 1500 }, (_, i) => `${i}`)))
    state.updateImageFromScroll({ target: { scrollTop: 1_000_000 } })
    await state.allDownloadImagesButton()
    await state.confirmDownload()
    assert.equal(requests[0].ids.length, 1024)
    assert.equal(requests[0].ids[0], '0')
    assert.equal(requests[0].ids.at(-1), '1023')
})

test('unselected catalog download asks the server for filtered first 1024 including unloaded pages', async () => {
    const requests = []
    const { state } = createApp({}, {
        fetch: async (_url, options) => {
            requests.push(JSON.parse(options.body).params)
            return { ok: false, json: async () => ({ error_code: 'downloadFailed' }) }
        },
    })
    state.ratingFilter = ['general', 'unclassified']
    await state.allDownloadImagesButton()
    await state.confirmDownload()
    assert.deepEqual(requests, [{
        first: 1024, model_name: 'ViT-L-14', pretrained: 'openai', ratings: ['general', 'unclassified'],
    }])
})

test('image similarity search explains its separate 64-image input limit', async () => {
    let requests = 0
    const { state } = createApp({ search: async () => { requests += 1; return searchResult([]) } })
    state.selectedItemId = Object.fromEntries(Array.from({ length: 65 }, (_, i) => [`${i}`, true]))
    await state.imagesSearchButton()
    assert.equal(requests, 0)
    assert.match(state.errorMessage, /imageSearchLimit/)
    delete state.selectedItemId['64']
    await state.imagesSearchButton()
    assert.equal(requests, 1)
})

test('language changes translate active errors and selection announcements', () => {
    const { state, watch } = createApp({}, {
        i18n: {
            ready: Promise.resolve(), getInitialLocale: () => 'ja', setLocale() {}, vuetifyMessages: {},
            t: (locale, key, params = {}) => `${locale}:${key}:${JSON.stringify(params)}`,
        },
    })
    state.showError({ code: 'uploadTooLarge' })
    state.toggleSelection('image')
    assert.match(state.errorMessage, /^ja:/)
    assert.match(state.selectionMessage, /^ja:/)
    state.locale = 'en'
    watch('locale', 'en')
    assert.match(state.errorMessage, /^en:errors.uploadTooLarge:/)
    assert.match(state.selectionMessage, /^en:selection.count:/)
    assert.match(state.selectionMessage, /"count":1/)
})

test('catalog totals and explicit model identity are preserved across page changes', async () => {
    const calls = []
    const { state } = createApp({
        getImageItemsByPage: async (page, size, model, pretrained, includeTotal) => {
            calls.push({ page, size, model, pretrained, includeTotal })
            const count = Math.max(0, Math.min(size, 100 - page * size))
            return catalogItems(Array.from({ length: count }, (_, i) => `${page * size + i}`), 120, 100)
        },
    })
    state.model_name = 'another-model'
    state.pretrained = 'another-weights'
    await state.browseImages()
    assert.equal(state.candidateCount, 120)
    assert.equal(state.searchCount, 100)
    assert.equal(state.downloadCount, 100)
    assert.equal(state.resultBuffer.length, 60)
    await state.loadCatalogPage(1)
    assert.equal(state.resultBuffer.length, 100)
    assert.equal(state.catalogTotal, 120)
    assert.deepEqual(calls, [
        { page: 0, size: 60, model: 'another-model', pretrained: 'another-weights', includeTotal: true },
        { page: 1, size: 60, model: 'another-model', pretrained: 'another-weights', includeTotal: false },
    ])
})

test('every search sends the shared settings with its own parameters', async () => {
    const requests = []
    const { state } = createApp({
        search: async (kind, params) => { requests.push([kind, params]); return searchResult([], 'saved-query') },
        searchUploadImage: async (file, params) => { requests.push(['uploadimage', { ...params, file: file.name }]); return searchResult([]) },
    })
    Object.assign(state, {
        model_name: 'model', pretrained: 'weights', text: 'cat', isRegexp: true, resultSize: '60',
        aesthetic_quality_beta: 0.25, aesthetic_quality_range: [2, 8], aesthetic_model_name: 'pony', features_strength: -0.5,
    })
    const shared = {
        model_name: 'model', pretrained: 'weights', aesthetic_quality_beta: 0.25,
        aesthetic_quality_range: [2, 8], aesthetic_model_name: 'pony', result_size: 60,
    }
    await state.textSearchButton()
    await state.nameSearchButton()
    await state.tagSearchButton()
    await state.styleClusterSearchButton()
    await state.randomSearchButton()
    await state.querySearchButton()
    await state.addTextFeaturesButton()
    state.selectedItemId = { a: true, b: true }
    await state.imagesSearchButton()
    state.uploadFile = { name: 'reference.png', size: 100 }
    await state.uploadImageSearchButton()
    assert.equal(JSON.stringify(requests), JSON.stringify([
        ['text', { ...shared, text: 'cat' }],
        ['name', { ...shared, text: 'cat', is_regexp: true }],
        ['tags', { ...shared, text: 'cat', is_regexp: true }],
        ['style_cluster', { ...shared, text: 'cat', is_regexp: true }],
        ['random', shared],
        ['query', { ...shared, search_query: 'saved-query' }],
        ['queryaddtext', { ...shared, text: 'cat', search_query: 'saved-query', features_strength: -0.5 }],
        ['image', { ...shared, id: ['a', 'b'] }],
        ['uploadimage', { ...shared, file: 'reference.png' }],
    ]))
    // The request holds a copy: moving the sliders afterwards cannot change it.
    assert.notEqual(requests[0][1].aesthetic_quality_range, state.aesthetic_quality_range)

    state.resultSize = 'invalid'
    state.text = '   '
    await state.textSearchButton()
    await state.nameSearchButton()
    await state.randomSearchButton()
    assert.equal(requests.length, 10)
    assert.equal(requests.at(-1)[1].result_size, 2048)
})

test('closing image details restores the trigger focus or the result heading after virtualization', () => {
    let triggerFocus = 0
    let fallbackFocus = 0
    const trigger = { isConnected: true, focus: options => { assert.equal(options.preventScroll, true); triggerFocus += 1 } }
    const { state } = createApp({ getImageMetadata: async () => ({}) }, {
        document: { getElementById: id => id === 'results-heading' ? { focus: () => { fallbackFocus += 1 } } : null },
    })
    state.openDialog({ id: 'image' }, { currentTarget: trigger })
    state.restoreDetailFocus()
    assert.equal(triggerFocus, 1)
    state.openDialog({ id: 'image' }, { currentTarget: trigger })
    trigger.isConnected = false
    state.restoreDetailFocus()
    assert.equal(fallbackFocus, 1)
})

test('large ZIP preparation starts a native download without buffering a Blob and reports skipped files', async () => {
    const links = []
    let requestedUrl = null
    const { state, watch } = createApp({}, {
        i18n: {
            ready: Promise.resolve(), getInitialLocale: () => 'ja', setLocale() {}, vuetifyMessages: {},
            t: (locale, key, params = {}) => `${locale}:${key}:${JSON.stringify(params)}`,
        },
        fetch: async url => {
            requestedUrl = url
            return {
                ok: true,
                json: async () => ({ download_url: '/downloads/one-time-token', image_count: 1020, skipped_count: 4, bytes: 12 * 1024 ** 3 }),
                blob: async () => { throw new Error('Large ZIPs must use the native browser download') },
            }
        },
        document: {
            getElementById: () => null,
            body: { appendChild() {} },
            createElement: () => ({ click() { links.push({ href: this.href, download: this.download }) }, remove() {} }),
        },
    })
    await state.allDownloadImagesButton()
    await state.confirmDownload()
    assert.equal(requestedUrl, '/downloads/prepare')
    assert.deepEqual(links, [{ href: '/downloads/one-time-token', download: 'images.zip' }])
    assert.match(state.downloadMessage, /^ja:download.startedSkipped:/)
    assert.match(state.downloadMessage, /"count":1020,"skipped":4/)
    state.locale = 'en'
    watch('locale', 'en')
    assert.match(state.downloadMessage, /^en:download.startedSkipped:/)
    assert.equal(state.isDownloading, false)
})

test('changing the categories while a search runs returns to the collection without staying busy', async () => {
    const response = deferred()
    const { state, watch } = createApp({ getImageItemsByPage: async () => catalogItems(['a', 'b']) })
    await state.browseImages()
    const search = state.runSearch(() => response.promise)
    await nextTurn()
    assert.equal(state.isSearching, true)

    state.ratingFilter = ['general']
    await watch('ratingFilter')
    assert.equal(state.isSearching, false)
    assert.deepEqual(displayedIds(state), ['a', 'b'])

    response.resolve(searchResult(['late']))
    await search
    assert.equal(state.isSearching, false)
    assert.equal(state.isPagedBrowseMode, true)
    assert.deepEqual(displayedIds(state), ['a', 'b'])
    // The next search is not blocked.
    await state.runSearch(async () => searchResult(['next'], 'query', { next: 'general' }))
    assert.deepEqual(displayedIds(state), ['next'])
})

// The conditions index.html puts on the reload button and on the empty state, evaluated against a state.
const template = fs.readFileSync(path.join(__dirname, '../app/presentation/view/index.html'), 'utf8')
function templateCondition(pattern) {
    const expression = template.match(pattern)[1]
    return state => Boolean(new Function('state', `with (state) { return ${expression} }`)(state))
}
const offersReload = templateCondition(/<v-btn v-else-if="([^"]+)" id="retryBrowse"/)
const showsEmptyState = templateCondition(/<div v-if="([^"]+)" class="emptyState"/)

test('a collection that failed to load keeps offering a reload after its error is dismissed', async () => {
    let fail = true
    const { state } = createApp({
        getImageItemsByPage: async () => {
            if (fail) throw new Error('server unreachable')
            return catalogItems(['a'])
        },
    })
    await state.browseImages()
    assert.equal(state.errorDetail, 'server unreachable')
    assert.equal(state.catalogUnavailable, true)
    state.errorMessage = ''
    assert.equal(state.catalogUnavailable, true)
    assert.equal(offersReload(state), true)
    assert.equal(showsEmptyState(state), false)

    // No reload is offered while a search runs or while the first page is being fetched.
    const response = deferred()
    const search = state.runSearch(() => response.promise)
    await nextTurn()
    assert.equal(state.catalogUnavailable, false)
    response.reject(new Error('search failed'))
    await search
    assert.equal(state.catalogUnavailable, true)

    fail = false
    const reload = state.browseImages()
    assert.equal(state.catalogUnavailable, false)
    await reload
    assert.equal(state.catalogUnavailable, false)
    assert.deepEqual(displayedIds(state), ['a'])
})

test('a loaded collection without matching images is empty rather than unavailable', async () => {
    const { state } = createApp({ getImageItemsByPage: async () => catalogItems([], 400, 0) })
    await state.browseImages()
    assert.equal(state.stableCatalog, true)
    assert.equal(state.searchCount, 0)
    assert.equal(state.catalogUnavailable, false)
    assert.equal(offersReload(state), false)
    assert.equal(showsEmptyState(state), true)
})

test('a search that fails over the collection resumes the page loads and selection it interrupted', async () => {
    const ids = Array.from({ length: 180 }, (_, i) => `image-${i}`)
    const pending = []
    const { state } = createApp({
        getImageItemsByPage: (page, size) => {
            if (page === 0) return Promise.resolve(catalogItems(ids.slice(0, size), ids.length))
            const response = deferred()
            pending.push({ page, response, items: catalogItems(ids.slice(page * size, (page + 1) * size)) })
            return response.promise
        },
    })
    state.numRows = 1
    state.numCols = 1
    await state.browseImages()

    // The viewport and a select-all both wait for page 1 when a search starts and then fails.
    const selecting = state.selectCurrentResults()
    state.showedItemIndex = 60
    state.refreshVisibleItems()
    state.ensureCatalogWindow()
    await nextTurn()
    assert.equal(state.isSelecting, true)
    assert.deepEqual(pending.map(request => request.page), [1])
    await state.runSearch(async () => { throw new Error('invalid expression') })
    assert.equal(state.isSearching, false)
    assert.equal(state.isSelecting, false)
    assert.equal(state.stableCatalog, true)

    // The failed search requests the page in view again instead of reusing the interrupted request.
    await nextTurn()
    assert.deepEqual(pending.map(request => request.page), [1, 1])
    assert.equal(state.isLoadingPage, true)
    assert.equal(state.displayItems[0].placeholder, true)
    pending[1].response.resolve(pending[1].items)
    await nextTurn()
    assert.equal(state.isLoadingPage, false)
    assert.equal(state.displayItems[0].id, 'image-60')

    // The interrupted request finishing late changes nothing.
    pending[0].response.resolve(pending[0].items)
    await selecting
    assert.equal(state.isLoadingPage, false)
    assert.equal(state.isSelecting, false)
    assert.equal(state.selectedCount, 0)
    await state.toggleSelection('image-60')
    assert.equal(state.selectedItemId['image-60'], true)
})

test('search result limit describes the completed request even if settings change while it runs', async () => {
    const response = deferred()
    const { state } = createApp()
    state.resultSize = 2048
    const search = state.runSearch(() => response.promise)
    await nextTurn()
    state.resultSize = 10
    response.resolve(searchResult(['image']))
    await search
    assert.equal(state.lastSearchLimit, 2048)
    await state.runSearch(async () => searchResult(['image']))
    assert.equal(state.lastSearchLimit, 10)
})

test('image details show the recorded file hash, or not available for images without one', async () => {
    const row = template.match(/<dt>\{\{ ([^}]+) \}\}<\/dt><dd id="detailFileHash"[^>]*>\{\{ ([^}]+) \}\}<\/dd>/)
    assert.ok(row, 'The image details must show the file hash')
    const [label, text] = row.slice(1).map(expression => new Function('state', `with (state) { return ${expression} }`))
    const hash = 'ab'.repeat(32)
    const { state } = createApp({ getImageMetadata: async (_model, _pretrained, id) => ({ tags: '', blake3: id === 'hashed' ? hash : null }) })
    assert.equal(label(state), 'translated:image.blake3')
    for (const [id, shown] of [['hashed', hash], ['unhashed', 'translated:common.notAvailable']]) {
        state.openDialog({ id })
        await nextTurn()
        assert.equal(text(state), shown)
    }
})
