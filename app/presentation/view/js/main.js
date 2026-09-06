import * as repository from "./modules/repository.js"
import { i18n } from "./i18n.js"
import { createCatalogScroll } from "./modules/catalog_scroll.js"

const { markRaw, nextTick } = Vue

/**
 * @typedef {{id: string, score: number, img_name: string, img_small: string, img_original: string, selected: boolean}} DisplayItem
 */

const vuetify = Vuetify.createVuetify({
    locale: { locale: i18n.getInitialLocale(), fallback: 'ja', messages: i18n.vuetifyMessages }
})

const app = Vue.createApp({
    el:"#app",
    data() {
        return {
            locale: i18n.getInitialLocale(),
            pageSize: 0,
            currentPage: 0,
            catalogTotal: null,
            catalogMatchingCount: null,
            catalogPages: {},
            catalogViewportHeight: 800,
            catalogPhysicalOffset: 0,
            catalogLogicalOffset: 0,
            catalogRenderShift: 0,
            scrollPositionPercent: 0,
            catalogPageError: '',
            catalogPageErrorKey: '',
            catalogErrorPage: null,
            serverCatalogAvailable: false,
            isSelecting: false,
            selectMode: false,
            selectionAnchorPosition: null,
            selectionRevision: 0,
            nextPage: 0,
            hasMorePages: true,
            isLoadingPage: false,
            isPagedBrowseMode: true,
            text: "",
            isShowSetting: false,
            isRegexp: false,
            numCols: 6,
            gridWidth: 0,
            numRows: 10,
            resultSize: 2048,
            lastSearchLimit: 2048,
            model_name: "ViT-L-14",
            pretrained: "openai",
            modelItems: [],
            search_query: "",
            showedItemIndex: 0,
            aesthetic_quality_beta: 0.00,
            aesthetic_quality_range: [0, 10],
            features_strength: 1.00,
            aesthetic_model_name: "original",
            uploadFile: null,
            padding_top: 0,
            padding_bottom: 500,
            item_height:282,
            searchDurationMs: null,
            clientDurationMs: null,
            isSearching: false,
            isDownloading: false,
            downloadConfirmation: null,
            downloadTriggerElement: null,
            downloadMessage: '',
            downloadMessageKey: '',
            downloadMessageParams: {},
            selectionAnchorId: null,
            selectionMessage: '',
            selectionMessageKey: '',
            selectionMessageParams: {},
            errorMessage: "",
            errorMessageKey: '',
            errorDetail: '',
            isInitializing: true,
            initializationFailed: false,
            isExploringPages: false,
            resultsRevision: 0,
            pageRequestId: 0,

            /**
             * @type {ResultItem[]}
             */
            rawResultBuffer:[],

            /**
             * @type {ResultItem[]}
             */
            resultBuffer:[],

            /**
             * @type {DisplayItem[]}
             */
            displayItems:[],

            /**
             * @type {{[itemId: string]: boolean}}
             */
            selectedItemId:{},

            /**
             * @type {{[itemId: string]: {tags: string, style_cluster: string, rating: string, aesthetic_quality: number}}}
             */
            imageMeta:{},

            /**
             * @type {{[modelKey: string]: {[itemId: string]: string}}}
             */
            ratingMaps:{},

            ratingFilter: ["general", "questionable", "sensitive", "explicit", "unclassified"],

            /**
             * @type {{[itemId: string]: boolean}}
             */
            activeDialogItem: null,
            detailTriggerElement: null,
            detailImageState: 'loading',
            detailImageAttempt: 0,
        }
    },
    mounted() {
        window.addEventListener('keydown', this.onModalKeydown, true)
        const grid = document.getElementById('itemArea')
        if (grid && typeof ResizeObserver !== 'undefined') {
            this._gridResizeObserver = new ResizeObserver(entries => {
                this.updateGridWidth(entries[0].contentRect.width)
                this.updateScrollProgress(document.getElementById('scroll-target'))
            })
            this._gridResizeObserver.observe(grid)
            this.updateGridWidth(grid.clientWidth)
            const area = document.getElementById('scroll-target')
            if (area) {
                this._areaResizeObserver = new ResizeObserver(() => {
                    if (this.stableCatalog) this.updateImageFromScroll({ target: area })
                    else this.updateScrollProgress(area)
                })
                this._areaResizeObserver.observe(area)
            }
        }
        this.init()
    },
    beforeUnmount() {
        this._catalogInput?.stop()
        window.removeEventListener('keydown', this.onModalKeydown, true)
        this._gridResizeObserver?.disconnect()
        this._areaResizeObserver?.disconnect()
    },
    watch:{
        isShowSetting(value) { if (value) this._catalogInput?.stop() },
        activeDialogItem(value) { if (value) this._catalogInput?.stop() },
        downloadConfirmation(value) { if (value) this._catalogInput?.stop() },
        isSearching(value) { if (value) this._catalogInput?.stop() },
        catalogScrollScale() { this._catalogInput?.stop() },
        locale(value) {
            i18n.setLocale(value)
            vuetify.locale.current.value = value
            if (this.errorMessage && this.errorMessageKey) this.errorMessage = this.t(this.errorMessageKey)
            if (this.catalogPageError) this.catalogPageError = this.t(this.catalogPageErrorKey)
            if (this.selectionMessage && this.selectionMessageKey) {
                this.selectionMessage = this.t(this.selectionMessageKey, this.selectionMessageParams)
            }
            if (this.downloadMessage && this.downloadMessageKey) {
                this.downloadMessage = this.t(this.downloadMessageKey, this.downloadMessageParams)
            }
        },
        ratingFilter(){
            if (this.isPagedBrowseMode && (this.serverCatalogAvailable || this.isLoadingPage)) return this.browseImages()
            this.applyRatingFilterToBuffer()
            this.pruneSelection()
            this.initImage(false)
            this.fillInitialViewport()
        },
        numCols(value) {
            this.numCols = Math.max(1, Math.min(24, Number.parseInt(value, 10) || 1))
            this.initImage(false)
        },
        model_name(){
            if (this.syncPretrainedForSelectedModel()) return
            this.onModelChange()
        },
        pretrained(){
            this.onModelChange()
        }
    },
    computed:{
        candidateCount() { return this.isPagedBrowseMode ? this.catalogTotal : this.rawResultBuffer.length },
        stableCatalog() { return this.isPagedBrowseMode && Number.isInteger(this.catalogMatchingCount) },
        searchCount() { return this.stableCatalog ? this.catalogMatchingCount : this.resultBuffer.length },
        scrollProgressPercent() {
            if (this.isInitializing || this.initializationFailed ||
                (this.isPagedBrowseMode && !this.stableCatalog && this.hasMorePages)) return null
            return this.searchCount ? this.scrollPositionPercent : 0
        },
        scrollProgressLabel() {
            const percent = this.scrollProgressPercent
            if (percent === null) return '—'
            return this.t('results.positionPercent', {
                percent: percent > 0 && percent < 1 ? '<1' : Math.floor(percent).toLocaleString(this.locale),
            })
        },
        catalogLogicalHeight() { return Math.ceil((this.catalogMatchingCount || 0) / this.visibleCols) * this.item_height },
        // Keep native layout within browser limits; cards and wheel steps keep their size.
        catalogPhysicalHeight() { return Math.min(8_000_000, this.catalogLogicalHeight) },
        catalogScrollScale() {
            return Math.max(1, (this.catalogLogicalHeight - this.catalogViewportHeight) /
                Math.max(1, this.catalogPhysicalHeight - this.catalogViewportHeight))
        },
        selectedCount() { return Object.keys(this.selectedItemId).length },
        selectedInCurrentCount() {
            return this.resultBuffer.filter(result => this.selectedItemId[result.item.id]).length
        },
        allCurrentSelected() {
            if (this.stableCatalog) {
                const count = Math.min(1024, this.catalogMatchingCount)
                return count > 0 && Array.from({ length: count }, (_, i) => this.catalogResultAt(i))
                    .every(result => result && this.selectedItemId[result.item.id])
            }
            return this.resultBuffer.length > 0 && this.resultBuffer.every(result => this.selectedItemId[result.item.id])
        },
        ratingOptions() {
            return ['general', 'sensitive', 'questionable', 'explicit', 'unclassified']
                .map(value => ({ title: this.t(`rating.${value}`), value }))
        },
        downloadCount() {
            if (this.selectedCount) return this.selectedCount
            return this.isPagedBrowseMode ? Math.min(1024, this.catalogMatchingCount ?? this.catalogTotal ?? 1024) : Math.min(1024, this.resultBuffer.length)
        },
        downloadLabel() {
            return this.t(this.selectedCount ? 'download.selected' : 'download.first', { count: this.downloadCount })
        },
        visibleCols() {
            return this.getColumnCount()
        },
        modelNameItems() {
            const names = this.modelItems.map(item => item.model_name)
            return [...new Set([this.model_name, ...names])]
        },
        pretrainedItems() {
            const values = this.modelItems
                .filter(item => item.model_name === this.model_name)
                .map(item => item.pretrained)
            return values.length > 0 ? [...new Set(values)] : [this.pretrained]
        },
    },
    methods:{
        t(key, params = {}) { return i18n.t(this.locale, key, params) },
        shortImageName(name) { return String(name || '').split(/[\\/]/).pop() },
        ratingTitle(value) {
            const category = ['general', 'sensitive', 'questionable', 'explicit'].includes(value) ? value : 'unclassified'
            return this.t(`rating.${category}`)
        },

        /**
         * 初期化する
         */
        init() {
            this.isInitializing = true
            this.initializationFailed = false
            this.errorMessage = ""
            return this.loadModelItems()
                .then(() => this.initBuffer())
                .catch(error => {
                    this.initializationFailed = true
                    this.showError(error)
                })
                .finally(() => { this.isInitializing = false })
        },

        getColumnCount() {
            const requested = Math.max(1, Math.min(24, Number.parseInt(this.numCols, 10) || 1))
            const fitting = this.gridWidth > 0 ? Math.max(1, Math.floor(this.gridWidth / 148)) : requested
            return Math.min(requested, fitting)
        },

        updateGridWidth(width) {
            if (!Number.isFinite(width) || width <= 0) return
            const previousCols = this.getColumnCount()
            const firstItem = this.showedItemIndex
            this.gridWidth = width
            const columns = this.getColumnCount()
            if (columns === previousCols) return
            if (this.stableCatalog) {
                this.scrollCatalogToItem(firstItem)
                return
            }
            const lastRow = Math.max(0, Math.ceil(this.resultBuffer.length / columns) - this.numRows)
            this.showedItemIndex = Math.min(Math.floor(firstItem / columns), lastRow) * columns
            this.refreshVisibleItems()
            // Keep the same neighborhood visible without clearing image selections.
            const scrollArea = document.getElementById('scroll-target')
            if (scrollArea) scrollArea.scrollTop = this.padding_top
        },

        loadModelItems() {
            return repository.getModelItems()
            .then(items => {
                this.modelItems = items
                if (!items.length) throw Object.assign(new Error(), { code: 'noModels' })
                if (items.length && !items.some(item => item.model_name === this.model_name)) {
                    this.model_name = items[0].model_name
                }
                this.syncPretrainedForSelectedModel()
            })
        },

        syncPretrainedForSelectedModel() {
            const candidates = this.modelItems
                .filter(item => item.model_name === this.model_name)
                .map(item => item.pretrained)

            if (candidates.length === 0) {
                return false
            }

            const preferred = candidates.find(value => value === this.pretrained)
            if (preferred !== undefined && !(this.model_name.includes("/") && preferred === "openai")) {
                return false
            }

            this.pretrained = candidates[0]
            return true
        },

        onModelChange(){
            if (this.isInitializing) return
            this.isSearching = false
            this.search_query = ""
            this.searchDurationMs = null
            this.clientDurationMs = null
            this.activeDialogItem = null
            return this.browseImages()
        },

        browseImages() {
            if (this.initializationFailed) return this.init()
            this.errorMessage = ""
            return this.initBuffer()
                .catch(error => this.showError(error))
        },

        describeError(error) {
            const detail = error?.response?.data?.error
            const code = error?.response?.data?.error_code || error?.code
            const key = `errors.${code || 'generic'}`
            const translated = this.t(key)
            const messageKey = translated === key ? 'errors.generic' : key
            return { key: messageKey, message: this.t(messageKey), detail: typeof detail === 'string' ? detail : error?.message || '' }
        },

        showError(error) {
            const description = this.describeError(error)
            this.errorMessageKey = description.key
            this.errorMessage = description.message
            this.errorDetail = description.detail
        },

        /**
         * 表示画像をリセットして、一部を表示する
         */
        initImage(clearSelection = true) {
            this._catalogInput?.stop()
            if (clearSelection) {
                this.clearSelection(false)
                this.selectMode = false
            }
            this.showedItemIndex = 0
            this.scrollPositionPercent = 0
            this.padding_top = 0
            const scrollArea = document.getElementById('scroll-target')
            if (scrollArea) scrollArea.scrollTop = 0
            this.catalogPhysicalOffset = 0
            this.catalogLogicalOffset = 0
            this._catalogScrollIntent = null
            if (this.stableCatalog) {
                this.refreshVisibleItems()
                this.ensureCatalogWindow()
                return
            }

            const end = this.numRows * this.getColumnCount()
            this.displayItems = this.resultBuffer.slice(0, end).map(result => ({
                id: result.item.id,
                score: result.score,
                tags: result.item.tags,
                img_name: result.item.name,
                img_small: repository.getImageSmallUrl(result.item.id),
                img_original: repository.getImageOriginalUrl(result.item.id),
                selected: false
            }))
            this.updatePadding()
        },

        /**
         * バッファを初期化する
         *
         * @return {Promise<void>}
         */
        initBuffer() {
            this._catalogInput?.stop()
            this.resultsRevision += 1
            this.pageRequestId += 1
            this.isLoadingPage = false
            this.isExploringPages = false
            // Keep this size stable across resizes so page offsets never skip images.
            this.pageSize = 60
            this.nextPage = 0
            this.currentPage = 0
            this.catalogTotal = null
            this.catalogMatchingCount = null
            this.catalogPages = {}
            this.catalogPageError = ''
            this.catalogErrorPage = null
            this._catalogPending = markRaw(new Map())
            this._catalogWindowRevision = null
            this.isSelecting = false
            this.hasMorePages = true
            this.isPagedBrowseMode = true
            this.rawResultBuffer = []
            this.resultBuffer = []
            this.initImage()

            return this.loadNextImagePage().then(() => this.fillInitialViewport())
        },

        async loadMoreImages() {
            if (this.stableCatalog) {
                this.catalogPageError = ''
                this.catalogErrorPage = null
                return this.ensureCatalogWindow()
            }
            if (this.isExploringPages || this.isLoadingPage || this.isSearching || this.initializationFailed || !this.ratingFilter.length) return
            this.isExploringPages = true
            this.errorMessage = ''
            const revision = this.resultsRevision
            const previousCount = this.resultBuffer.length
            const budget = 3
            try {
                for (let page = 0; page < budget && this.hasMorePages; page += 1) {
                    await this.loadNextImagePage()
                    if (revision !== this.resultsRevision || this.errorMessage || this.resultBuffer.length > previousCount) break
                }
            } finally {
                if (revision === this.resultsRevision) this.isExploringPages = false
            }
        },

        async fillInitialViewport() {
            if (this.stableCatalog) return this.ensureCatalogWindow()
            if (this.isExploringPages || this.isLoadingPage || this.isSearching || this.initializationFailed || this.errorMessage || !this.ratingFilter.length) return
            this.isExploringPages = true
            const revision = this.resultsRevision
            try {
                // A large viewport or selective rating filter may leave no scroll
                // gesture available. Bound automatic work; the button can continue.
                for (let page = 0; page < 3 && this.hasMorePages; page += 1) {
                    await nextTick()
                    if (revision !== this.resultsRevision || this.errorMessage) break
                    const area = document.getElementById('scroll-target')
                    const fitsViewport = area?.clientHeight > 0 && area.scrollHeight <= area.clientHeight + 1
                    if (this.resultBuffer.length && !fitsViewport) break
                    await this.loadNextImagePage()
                }
            } finally {
                if (revision === this.resultsRevision) this.isExploringPages = false
            }
        },

        loadNextImagePage() {
            if (this.stableCatalog) return this.loadCatalogPage(this.nextPage)
            if (!this.isPagedBrowseMode || !this.hasMorePages || this.isLoadingPage || this.isSearching || this.initializationFailed) return Promise.resolve()
            const requestedPage = this.nextPage
            this.isLoadingPage = true
            const revision = this.resultsRevision
            const requestId = ++this.pageRequestId
            return repository.getImageItemsByPage(requestedPage, this.pageSize, this.model_name, this.pretrained, this.catalogTotal === null, this.ratingFilter)
                .then(async objs => {
                    if (revision !== this.resultsRevision || requestId !== this.pageRequestId) return
                    const results = objs.map(obj => ({ item: obj, score: 0 }))
                    if (Number.isInteger(objs.matchingCount) && objs.matchingCount >= 0) {
                        this.catalogTotal = objs.totalCount
                        this.catalogMatchingCount = objs.matchingCount
                        this.serverCatalogAvailable = true
                        this.catalogPages = { [requestedPage]: markRaw(results) }
                        this.currentPage = requestedPage
                        this.nextPage = requestedPage + 1
                        this.hasMorePages = false
                        this.rebuildCatalogBuffer()
                        const area = document.getElementById('scroll-target')
                        if (area?.clientHeight) this.catalogViewportHeight = area.clientHeight
                        this.refreshVisibleItems()
                        return
                    }
                    await this.ensureRatingMapForResults(results)
                    if (revision !== this.resultsRevision || requestId !== this.pageRequestId) return
                    if (Number.isInteger(objs.totalCount) && objs.totalCount >= 0) this.catalogTotal = objs.totalCount
                    this.hasMorePages = this.catalogTotal === null ? objs.length === this.pageSize
                        : (requestedPage + 1) * this.pageSize < this.catalogTotal
                    this.currentPage = requestedPage
                    this.nextPage = requestedPage + 1
                    // Only newly fetched results extend the scrollbar. Moving the
                    // virtual window keeps the total loaded-row height unchanged.
                    this.rawResultBuffer = markRaw(this.rawResultBuffer.concat(results))
                    this.applyRatingFilterToBuffer()
                    this.refreshVisibleItems()
                })
                .catch(error => {
                    if (revision === this.resultsRevision && requestId === this.pageRequestId) this.showError(error)
                })
                .finally(() => {
                    if (requestId === this.pageRequestId) this.isLoadingPage = false
                })
        },

        getRatingMapKey() {
            return `${this.model_name}-${this.pretrained}`
        },

        catalogResultAt(position) {
            return this.catalogPages[Math.floor(position / this.pageSize)]?.[position % this.pageSize]
        },

        rebuildCatalogBuffer() {
            this.rawResultBuffer = markRaw(Object.keys(this.catalogPages).map(Number).sort((a, b) => a - b)
                .flatMap(page => this.catalogPages[page]))
            this.resultBuffer = this.rawResultBuffer
        },

        async loadCatalogPage(page) {
            if (!this.stableCatalog || page < 0 || page * this.pageSize >= this.catalogMatchingCount) return
            if (this.catalogPages[page]) return
            if (this._catalogPending.has(page)) return this._catalogPending.get(page)
            const revision = this.resultsRevision
            const pending = this._catalogPending
            this.isLoadingPage = true
            const request = repository.getImageItemsByPage(page, this.pageSize, this.model_name, this.pretrained, false, [...this.ratingFilter])
                .then(objs => {
                    if (revision !== this.resultsRevision || !this.stableCatalog) return
                    this.catalogPages = { ...this.catalogPages, [page]: markRaw(objs.map(item => ({ item, score: 0 }))) }
                    this.trimCatalogPages()
                    this.rebuildCatalogBuffer()
                    this.refreshVisibleItems()
                })
                .catch(error => {
                    if (revision === this.resultsRevision && this.catalogPageInView(page)) {
                        const description = this.describeError(error)
                        this.catalogPageError = description.message
                        this.catalogPageErrorKey = description.key
                        this.catalogErrorPage = page
                    }
                    throw error
                })
                .finally(() => {
                    pending.delete(page)
                    if (revision === this.resultsRevision) this.isLoadingPage = pending.size > 0
                })
            pending.set(page, markRaw(request))
            return request
        },

        trimCatalogPages() {
            const pages = Object.keys(this.catalogPages).map(Number)
            if (pages.length <= 64 || this.isSelecting) return
            const current = Math.floor(this.showedItemIndex / this.pageSize)
            // Selection IDs and its logical anchor survive page-cache eviction.
            const removable = pages.sort((a, b) => Math.abs(b - current) - Math.abs(a - current))
            const next = { ...this.catalogPages }
            while (Object.keys(next).length > 64 && removable.length) delete next[removable.shift()]
            this.catalogPages = next
        },

        async ensureCatalogRange(start, end, selectionRevision = this.selectionRevision) {
            const revision = this.resultsRevision
            for (let page = Math.floor(start / this.pageSize); page <= Math.floor((end - 1) / this.pageSize) && end > start; page++) {
                if (revision !== this.resultsRevision || selectionRevision !== this.selectionRevision) return
                await this.loadCatalogPage(page)
            }
        },

        async ensureCatalogWindow() {
            if (!this.stableCatalog || this.isSearching || this.catalogPageError) return
            const revision = this.resultsRevision
            if (this._catalogWindowRevision === revision) return this._catalogWindowPromise
            this._catalogWindowRevision = revision
            this._catalogWindowPromise = markRaw((async () => {
                try {
                    // Re-evaluate the viewport after each response. A thumb drag can
                    // jump directly to the last page without fetching intermediate pages.
                    while (revision === this.resultsRevision && this.stableCatalog) {
                        const start = this.showedItemIndex
                        const end = Math.min(start + this.numRows * this.visibleCols, this.catalogMatchingCount)
                        const page = Array.from({ length: Math.max(0, Math.ceil(end / this.pageSize) - Math.floor(start / this.pageSize)) },
                            (_, i) => Math.floor(start / this.pageSize) + i).find(value => !this.catalogPages[value])
                        if (page === undefined) break
                        try { await this.loadCatalogPage(page) }
                        catch (error) {
                            if (revision !== this.resultsRevision || this.catalogPageInView(page)) break
                            // A superseded viewport failed; still load the current one.
                        }
                    }
                } catch (_) { /* The fixed viewport notification offers a retry. */ }
                finally { if (this._catalogWindowRevision === revision) this._catalogWindowRevision = null }
            })())
            return this._catalogWindowPromise
        },

        catalogGridOffset(area) {
            const grid = document.getElementById('itemArea')
            return grid && area.getBoundingClientRect
                ? grid.getBoundingClientRect().top - area.getBoundingClientRect().top + area.scrollTop : 0
        },

        catalogPageInView(page) {
            const start = this.showedItemIndex
            const end = Math.min(start + this.numRows * this.visibleCols, this.catalogMatchingCount)
            return page * this.pageSize < end && (page + 1) * this.pageSize > start
        },

        scrollCatalogBy(delta, area = document.getElementById('scroll-target')) {
            if (!area) return 0
            const offset = this.catalogGridOffset(area)
            const physicalRange = Math.max(0, this.catalogPhysicalHeight - this.catalogViewportHeight)
            const logicalRange = Math.max(0, this.catalogLogicalHeight - this.catalogViewportHeight)
            const position = area.scrollTop - offset
            const fromPhysical = position => position < 0 ? position : position > physicalRange
                ? logicalRange + position - physicalRange : position * this.catalogScrollScale
            const logical = this._catalogScrollIntent?.top === area.scrollTop
                ? this._catalogScrollIntent.logical : fromPhysical(position)
            const maximum = fromPhysical(area.scrollHeight - area.clientHeight - offset)
            const next = Math.max(-offset, Math.min(maximum, logical + delta))
            area.scrollTop = offset + (next < 0 ? next : next > logicalRange
                ? physicalRange + next - logicalRange : next / this.catalogScrollScale)
            // Chrome rounds scrollTop to whole CSS pixels. Keep the intended
            // logical offset so short wheel/key/touch movements are not lost.
            this._catalogScrollIntent = markRaw({ top: area.scrollTop, logical: next })
            this.updateImageFromScroll({ target: area })
            return next - logical
        },

        scrollCatalogToItem(position) {
            this._catalogInput?.stop()
            const area = document.getElementById('scroll-target')
            if (!area) return
            nextTick().then(() => {
                const logical = Math.floor(position / this.visibleCols) * this.item_height
                area.scrollTop = this.catalogGridOffset(area) + logical / this.catalogScrollScale
                this._catalogScrollIntent = markRaw({ top: area.scrollTop, logical })
                this.updateImageFromScroll({ target: area })
            })
        },

        async scrollToTop() {
            this._catalogInput?.stop()
            this.isShowSetting = false
            await nextTick()
            const area = document.getElementById('scroll-target')
            if (!area) return
            this._catalogScrollIntent = null
            area.scrollTop = 0
            this.updateImageFromScroll({ target: area })
            area.focus?.({ preventScroll: true })
        },

        onCatalogWheel(event) {
            this._catalogInput?.stop()
            if (!this.stableCatalog || this.catalogScrollScale <= 1 || event.ctrlKey || !event.deltaY || event.cancelable === false) return
            event.preventDefault()
            this.scrollCatalogBy(event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.catalogViewportHeight : 1))
        },

        onCatalogPointerDown(event) {
            if (!['touch', 'pen'].includes(event.pointerType)) return this._catalogInput?.pointerDown(event)
            if (!this._catalogInput) this._catalogInput = markRaw(createCatalogScroll({
                enabled: () => this.stableCatalog && this.catalogScrollScale > 1 &&
                    !this.isShowSetting && !this.activeDialogItem && !this.downloadConfirmation && !this.isSearching && !this.isInitializing,
                scrollBy: delta => this.scrollCatalogBy(delta),
                reducedMotion: () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
            }))
            this._catalogInput.pointerDown(event)
        },
        onCatalogPointerMove(event) { this._catalogInput?.pointerMove(event) },
        onCatalogPointerUp(event) { this._catalogInput?.pointerUp(event) },
        onCatalogPointerCancel() { this._catalogInput?.pointerCancel() },
        onCatalogTouchStart(event) { this._catalogInput?.touchStart(event) },
        onCatalogTouchMove(event) { this._catalogInput?.touchMove(event) },
        onCatalogClick(event) { this._catalogInput?.click(event) },

        onCatalogKeydown(event) {
            this._catalogInput?.keyDown()
            if (event.key === 'Escape' && this.selectMode) {
                this.exitSelectionMode()
                event.preventDefault()
                return
            }
            if (!this.stableCatalog || this.catalogScrollScale <= 1 || event.ctrlKey || event.metaKey || event.altKey ||
                /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName) || event.target.isContentEditable) return
            const step = Math.max(this.item_height, this.catalogViewportHeight - this.item_height)
            const delta = event.key === ' ' && event.target.id === 'scroll-target' ? (event.shiftKey ? -step : step)
                : { ArrowDown: 40, ArrowUp: -40, PageDown: step, PageUp: -step }[event.key]
            if (delta !== undefined) {
                event.preventDefault()
                this.scrollCatalogBy(delta)
            } else if (event.key === 'Home' || event.key === 'End') {
                event.preventDefault()
                const area = document.getElementById('scroll-target')
                this._catalogScrollIntent = null
                area.scrollTop = event.key === 'Home' ? 0 : area.scrollHeight
                this.updateImageFromScroll({ target: area })
            }
        },

        ensureRatingMapForResults(results) {
            return this.ensureRatingMapForItemIds(results.map(result => result.item.id))
        },

        ensureRatingMapForItemIds(itemIds) {
            const ratingKey = this.getRatingMapKey()
            const ratingMap = this.ratingMaps[ratingKey] || {}
            const uniqueItemIds = [...new Set(itemIds)]
            const missingItemIds = uniqueItemIds.filter(itemId => ratingMap[itemId] === undefined)

            if(missingItemIds.length === 0) {
                this.ratingMaps[ratingKey] = ratingMap
                return Promise.resolve()
            }

            return repository.getImageRatings(this.model_name, this.pretrained, missingItemIds)
            .then((ratings) => {
                const merged = { ...this.ratingMaps[ratingKey], ...ratings }
                // Discard unrelated old-query metadata when the cache grows.
                // Preserve the visible query even when an older request finishes.
                const retainIds = new Set(uniqueItemIds)
                if (ratingKey === this.getRatingMapKey()) {
                    this.rawResultBuffer.forEach(result => retainIds.add(result.item.id))
                }
                this.ratingMaps[ratingKey] = Object.keys(merged).length > 16384
                    ? Object.fromEntries([...retainIds].filter(id => merged[id] !== undefined).map(id => [id, merged[id]]))
                    : merged
            })
        },

        getCurrentRatingMap() {
            const ratingKey = this.getRatingMapKey()
            return this.ratingMaps[ratingKey] || {}
        },

        applyRatingFilter(resultList) {
            const allowedRatings = new Set(this.ratingFilter)
            const ratingMap = this.getCurrentRatingMap()
            const knownRatings = new Set(['general', 'questionable', 'sensitive', 'explicit'])

            return resultList.filter(result => {
                const rating = ratingMap[result.item.id]
                const category = knownRatings.has(rating) ? rating : 'unclassified'
                return allowedRatings.has(category)
            })
        },

        applyRatingFilterToBuffer() {
            this.resultBuffer = this.applyRatingFilter(this.rawResultBuffer)
        },


        updateScrollProgress(area) {
            if (!area) return
            // Counted catalogs reserve their entire scroll range, including
            // unloaded rows. Use that range, not the changing rendered window.
            const range = Math.max(0, (area.scrollHeight || 0) - (area.clientHeight || 0))
            const top = Math.max(0, area.scrollTop || 0)
            this.scrollPositionPercent = range === 0 || top === 0 ? 0
                : range - top <= 1 ? 100 : Math.min(99.999, top / range * 100)
        },

        // Scroll only changes the rendered window inside the current result set.
        updateImageFromScroll(e) {
            this.updateScrollProgress(e.target)
            if (this.stableCatalog) {
                const area = e.target
                if (area.clientHeight) this.catalogViewportHeight = area.clientHeight
                const offset = this.catalogGridOffset(area)
                const range = Math.max(0, this.catalogPhysicalHeight - this.catalogViewportHeight)
                this.catalogPhysicalOffset = Math.max(0, Math.min(range, area.scrollTop - offset))
                const intended = this._catalogScrollIntent?.top === area.scrollTop ? this._catalogScrollIntent.logical : null
                if (intended === null) this._catalogScrollIntent = null
                const logical = Math.max(0, Math.min(Math.max(0, this.catalogLogicalHeight - this.catalogViewportHeight),
                    intended ?? this.catalogPhysicalOffset * this.catalogScrollScale))
                this.catalogLogicalOffset = logical
                const row = Math.max(0, Math.floor(logical / this.item_height) - 2)
                const last = Math.max(0, Math.ceil(this.catalogMatchingCount / this.visibleCols) - this.numRows)
                this.showedItemIndex = Math.min(row, last) * this.visibleCols
                if (this.catalogPageError && !this.catalogPageInView(this.catalogErrorPage)) {
                    this.catalogPageError = ''
                    this.catalogErrorPage = null
                }
                this.refreshVisibleItems()
                this.ensureCatalogWindow()
                return
            }
            const columns = this.getColumnCount()
            const grid = document.getElementById('itemArea')
            const gridOffset = grid && e.target.getBoundingClientRect
                ? grid.getBoundingClientRect().top - e.target.getBoundingClientRect().top + e.target.scrollTop : 0
            const row = Math.max(0, Math.floor((e.target.scrollTop - gridOffset) / this.item_height) - 2)
            const lastRow = Math.max(0, Math.ceil(this.resultBuffer.length / columns) - this.numRows)
            this.showedItemIndex = Math.min(row, lastRow) * columns
            this.refreshVisibleItems()
            this.loadNextImagePageIfNeeded(e.target)
        },

        loadNextImagePageIfNeeded(area = document.getElementById('scroll-target')) {
            if (!this.isPagedBrowseMode || this.isExploringPages || this.isLoadingPage || this.isSearching || this.isInitializing || this.initializationFailed || this.errorMessage || !this.resultBuffer.length) return
            if (area?.clientHeight > 0 && area.scrollTop + area.clientHeight >= area.scrollHeight - this.item_height * 2) {
                this.loadMoreImages()
            }
        },

        refreshVisibleItems(){
            if (this.stableCatalog) {
                const end = Math.min(this.showedItemIndex + this.numRows * this.visibleCols, this.catalogMatchingCount)
                const visible = Array.from({ length: end - this.showedItemIndex }, (_, offset) => {
                    const position = this.showedItemIndex + offset
                    const result = this.catalogResultAt(position)
                    return result ? {
                        id: result.item.id, score: result.score, tags: result.item.tags,
                        img_name: result.item.name, img_small: repository.getImageSmallUrl(result.item.id),
                        img_original: repository.getImageOriginalUrl(result.item.id), position,
                    } : { id: `pending-${position}`, position, placeholder: true }
                })
                const focusedCard = document.activeElement?.closest?.('#itemArea .itemBlock')
                if (focusedCard && !visible.some(item => item.id === focusedCard.dataset.imageId)) {
                    document.getElementById('scroll-target')?.focus?.({ preventScroll: true })
                }
                this.displayItems = visible
                this.updatePadding()
                return
            }
            const end = Math.min(this.showedItemIndex + this.numRows * this.getColumnCount(), this.resultBuffer.length)
            this.displayItems = []
            this.sliceShowImg(this.showedItemIndex, end)
            this.updatePadding()
        },

        updatePadding() {
            const columns = this.getColumnCount()
            if (this.stableCatalog) {
                const logical = this.catalogLogicalOffset
                const renderTop = this.catalogPhysicalOffset + Math.floor(this.showedItemIndex / columns) * this.item_height - logical
                this.padding_top = Math.max(0, renderTop)
                this.catalogRenderShift = Math.min(0, renderTop)
                this.padding_bottom = Math.max(0, this.catalogPhysicalHeight - this.padding_top -
                    Math.ceil(this.displayItems.length / columns) * this.item_height)
                return
            }
            this.padding_top = Math.floor(this.showedItemIndex / columns) * this.item_height
            const totalRows = Math.ceil(this.resultBuffer.length / columns)
            const visibleRows = Math.ceil(this.displayItems.length / columns)
            this.padding_bottom = Math.max(0, totalRows * this.item_height - this.padding_top - visibleRows * this.item_height)
        },

        sliceShowImg(start, end){
            const slice = this.resultBuffer.slice(start, end).map(result => ({
                id: result.item.id,
                score: result.score,
                tags: result.item.tags,
                img_name: result.item.name,
                img_small: repository.getImageSmallUrl(result.item.id),
                img_original: repository.getImageOriginalUrl(result.item.id),
                selected: false
            }))
            this.displayItems = this.displayItems.concat(slice) // 一度の代入
        },


        /**
         * 検索結果の反映後に表示を更新して状態を戻す
         * @param {() => Promise<repository.ResultItem[]>} request
         * @param {(() => void) | null} afterBuffer
         * @return {Promise<void>}
         */
        runSearch(request, afterBuffer = null) {
            if (this.isInitializing || this.initializationFailed || this.isSearching) return Promise.resolve()
            this.isSearching = true
            this.isShowSetting = false
            const searchStart = performance.now()
            const parsedLimit = Number.parseInt(this.resultSize, 10)
            const requestedLimit = Number.isFinite(parsedLimit) && parsedLimit > 0 ? parsedLimit : 2048
            const revision = ++this.resultsRevision
            this.pageRequestId += 1
            this.isLoadingPage = false
            this.isExploringPages = false
            this.errorMessage = ""
            this.searchDurationMs = null
            this.clientDurationMs = null
            return Promise.resolve().then(request).then(result => {
                if (revision !== this.resultsRevision) return
                this.searchDurationMs = performance.now() - searchStart
                // ── 計測開始：fetch は完了している
                const clientStart = performance.now()

                // サーバ側でスコア降順・名前昇順に整列済み
                const array = result.list

                return this.ensureRatingMapForResults(array)
                .then(() => {
                    if (revision !== this.resultsRevision) return
                    this.search_query = result.search_query
                    this.lastSearchLimit = requestedLimit
                    this.rawResultBuffer = markRaw(array)
                    this.isPagedBrowseMode = false
                    this.applyRatingFilterToBuffer()

                    if (afterBuffer) afterBuffer()

                    // displayItems を更新（ここで Vue が patch を予約する）
                    this.initImage()

                    // ── Vue の DOM patch 完了を待つ
                    return nextTick()
                    // ── さらにブラウザのレイアウト・ペイント完了まで待つ
                    .then(() => new Promise(resolve => {
                        requestAnimationFrame(() => requestAnimationFrame(resolve))
                    }))
                    .then(() => {
                        if (revision === this.resultsRevision) this.clientDurationMs = performance.now() - clientStart
                    })
                })
            })
            .catch(error => {
                if (revision === this.resultsRevision) this.showError(error)
            })
            .finally(() => {
                if (revision === this.resultsRevision) this.isSearching = false
            })
        },



        textSearchButton() {
            if (!this.text?.trim()) return
            return this.runSearch(() => repository.searchText(this.model_name, this.pretrained, this.text, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name, this.resultSize))
        },

        imageSearchAction() {
            if (this.isSearching || this.isInitializing || this.initializationFailed || this.isSelecting) return
            return this.selectedCount ? this.imagesSearchButton() : this.chooseSearchFromFile()
        },

        chooseSearchFromFile() {
            const input = document.getElementById('imageSearchFileInput')
            if (!input) return
            input.value = ''
            input.click()
        },

        onImageSearchFileChange(event) {
            const file = event.target.files?.[0]
            event.target.value = ''
            if (!file) return
            this.uploadFile = file
            return this.uploadImageSearchButton()
        },

        enterSelectionMode() {
            if (this.isSearching || this.isInitializing || this.initializationFailed || !this.searchCount) return
            this.selectMode = true
            this.announceSelection('')
            return nextTick().then(() => document.getElementById('selectionModeToggle')?.focus?.({ preventScroll: true }))
        },

        exitSelectionMode() {
            this.clearSelection(false)
            this.selectMode = false
            return nextTick().then(() => document.getElementById('selectionModeToggle')?.focus?.({ preventScroll: true }))
        },

        onImageClick(item, event = {}) {
            if (item.placeholder || this.isSearching || this.isInitializing) return
            return this.selectMode ? this.toggleSelection(item.id, event) : this.openDialog(item, event)
        },

        imagesSearchButton() {
            const ids = Object.keys(this.selectedItemId)
            if (!ids.length) return
            if (ids.length > 64) {
                this.showError(Object.assign(new Error(), { code: 'imageSearchLimit' }))
                return
            }
            return this.runSearch(() => repository.searchImage(this.model_name, this.pretrained, ids, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name, this.resultSize))
        },

        uploadImageSearchButton() {
            const file = Array.isArray(this.uploadFile) ? this.uploadFile[0] : this.uploadFile
            if (!file) return
            if (file.size > repository.MAX_UPLOAD_BYTES) {
                this.showError(Object.assign(new Error(), { code: 'uploadTooLarge' }))
                return
            }
            return this.runSearch(() => repository.searchUploadImage(this.model_name, this.pretrained, file, this.resultSize), () => { this.uploadFile = null })
        },

        nameSearchButton() {
            if (!this.text?.trim()) return
            return this.runSearch(() => repository.searchName(this.model_name, this.pretrained, this.text, this.isRegexp, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name, this.resultSize))
        },

        randomSearchButton() {
            return this.runSearch(() => repository.searchRandom(this.model_name, this.pretrained, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name, this.resultSize))
        },

        querySearchButton() {
            if (!this.search_query?.trim()) return
            return this.runSearch(() => repository.searchQuery(this.model_name, this.pretrained, this.search_query, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name, this.resultSize))
        },

        addTextFeaturesButton() {
            if (!this.text?.trim() || !this.search_query?.trim()) return
            return this.runSearch(() => repository.addTextFeatures(this.model_name, this.pretrained, this.text, this.search_query, this.features_strength, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name, this.resultSize))
        },

        tagSearchButton() {
            if (!this.text?.trim()) return
            return this.runSearch(() => repository.searchTags(this.model_name, this.pretrained, this.text, this.isRegexp, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name, this.resultSize))
        },

        styleClusterSearchButton() {
            if (!this.text?.trim()) return
            return this.runSearch(() => repository.searchStyleCluster(this.model_name, this.pretrained, this.text, this.isRegexp, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name, this.resultSize))
        },

        getDownloadIds() {
            const selected = Object.keys(this.selectedItemId)
            return selected.length ? selected : this.resultBuffer.slice(0, 1024).map(result => result.item.id)
        },

        allDownloadImagesButton(event = {}) {
            if (this.isDownloading || this.downloadConfirmation || this.isSearching || this.isInitializing || this.initializationFailed || this.isSelecting || !this.downloadCount) return
            const ids = this.getDownloadIds()
            const selected = this.selectedCount > 0
            const fromCatalog = this.isPagedBrowseMode && !selected
            if (!ids.length && !fromCatalog) return
            if (ids.length > 1024) {
                this.showError(Object.assign(new Error(), { code: 'selectionLimit' }))
                return
            }
            if (!selected && !this.ratingFilter.length) return
            const trigger = event.currentTarget || document.activeElement
            this.downloadTriggerElement = trigger ? markRaw(trigger) : null
            this.downloadMessage = ''
            // The confirmation describes this exact request even if an async
            // update changes the current result or selection before confirmation.
            this.downloadConfirmation = {
                count: fromCatalog ? this.downloadCount : ids.length,
                selected,
                params: fromCatalog ? {
                    first: 1024,
                    model_name: this.model_name,
                    pretrained: this.pretrained,
                    ratings: [...this.ratingFilter],
                } : { ids },
            }
        },

        focusDownloadCancel() {
            if (this.downloadConfirmation) document.getElementById('downloadCancel')?.focus({ preventScroll: true })
        },

        cancelDownloadConfirmation() {
            this.downloadConfirmation = null
        },

        onModalKeydown(event) {
            if (event.key !== 'Escape' || event.isComposing) return
            // A reopened Vuetify overlay can be visible before its deferred
            // top-of-stack flag updates. Close our simple modals immediately,
            // including while focus still sits on the image that opened them.
            if (this.downloadConfirmation) this.cancelDownloadConfirmation()
            else if (this.activeDialogItem) this.activeDialogItem = null
            else return
            event.preventDefault()
            event.stopImmediatePropagation()
        },

        restoreDownloadFocus() {
            if (this.downloadConfirmation) return
            const trigger = this.downloadTriggerElement
            this.downloadTriggerElement = null
            if (trigger?.isConnected && !trigger.disabled) trigger.focus({ preventScroll: true })
            else document.getElementById('results-heading')?.focus({ preventScroll: true })
        },

        async confirmDownload() {
            const confirmation = this.downloadConfirmation
            if (!confirmation || this.isDownloading) return
            // Consume the confirmation before the first await: rapid repeats
            // cannot create a second ZIP or adopt a different selection.
            this.downloadConfirmation = null
            this.isDownloading = true
            this.downloadMessage = ''
            this.errorMessage = ''
            try {
                const response = await fetch('/downloads/prepare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ params: confirmation.params })
                })
                if (!response.ok) {
                    let data = null
                    try { data = await response.json() } catch (_) {}
                    throw { response: { data: data || { error_code: 'downloadFailed' } } }
                }
                const download = await response.json()
                if (typeof download.download_url !== 'string' || !download.download_url.startsWith('/downloads/')) {
                    throw Object.assign(new Error(), { code: 'downloadFailed' })
                }
                let link = null
                try {
                    link = document.createElement('a')
                    link.href = download.download_url
                    link.download = 'images.zip'
                    document.body.appendChild(link)
                    link.click()
                } finally {
                    link?.remove()
                }
                this.downloadMessageKey = download.skipped_count > 0 ? 'download.startedSkipped' : 'download.started'
                this.downloadMessageParams = { count: download.image_count, skipped: download.skipped_count || 0 }
                this.downloadMessage = this.t(this.downloadMessageKey, this.downloadMessageParams)
            } catch (error) {
                this.showError(error)
            } finally {
                this.isDownloading = false
            }
        },
        openDialog(item, event = {}) {
            const trigger = event.currentTarget || document.activeElement
            this.detailTriggerElement = trigger ? markRaw(trigger) : null
            this.detailImageAttempt += 1
            this.detailImageState = 'loading'
            this.activeDialogItem = item
            this.fetchImageMetadata(item)
        },

        onDetailImageLoad(itemId, attempt) {
            if (this.activeDialogItem?.id === itemId && this.detailImageAttempt === attempt) this.detailImageState = 'loaded'
        },

        onDetailImageError(itemId, attempt) {
            if (this.activeDialogItem?.id === itemId && this.detailImageAttempt === attempt) this.detailImageState = 'error'
        },

        retryDetailImage() {
            if (!this.activeDialogItem || this.detailImageState !== 'error') return
            this.detailImageAttempt += 1
            this.detailImageState = 'loading'
        },

        restoreDetailFocus() {
            const trigger = this.detailTriggerElement
            this.detailTriggerElement = null
            if (trigger?.isConnected) trigger.focus({ preventScroll: true })
            else document.getElementById('results-heading')?.focus({ preventScroll: true })
        },

        fetchImageMetadata(item) {
            const key = `${this.getRatingMapKey()}:${item.id}`
            const revision = this.resultsRevision
            if (this.imageMeta[key]) {
                return
            }

            repository.getImageMetadata(this.model_name, this.pretrained, item.id)
            .then(meta => {
                this.imageMeta[key] = meta
            })
            .catch(error => {
                if (revision === this.resultsRevision && this.activeDialogItem?.id === item.id) this.showError(error)
            })
        },

        getImageMetadata(itemId) {
            return this.imageMeta[`${this.getRatingMapKey()}:${itemId}`] || { tags: "", style_cluster: "", rating: "", aesthetic_quality: 0 }
        },

        onSelectItem(event) {

            const index = event.currentTarget.dataset.index;
            const item = this.displayItems[index];
            if (item) this.onImageClick(item, event)
        },

        async toggleSelection(id, event = {}) {
            if (this.isSelecting) return
            if (this.stableCatalog) {
                const page = Object.keys(this.catalogPages).find(key => this.catalogPages[key].some(result => result.item.id === id))
                if (page === undefined) return
                const position = Number(page) * this.pageSize + this.catalogPages[page].findIndex(result => result.item.id === id)
                if (event.shiftKey && this.selectionAnchorPosition !== null) {
                    const start = Math.min(position, this.selectionAnchorPosition)
                    const end = Math.max(position, this.selectionAnchorPosition) + 1
                    if (end - start > 1024) { this.announceSelection('selection.limit'); return }
                    const revision = this.resultsRevision
                    const selectionRevision = ++this.selectionRevision
                    const selected = !this.selectedItemId[id]
                    this.isSelecting = true
                    try {
                        await this.ensureCatalogRange(start, end)
                        if (revision !== this.resultsRevision || selectionRevision !== this.selectionRevision) return
                        const next = { ...this.selectedItemId }
                        for (let i = start; i < end; i++) {
                            const itemId = this.catalogResultAt(i)?.item.id
                            if (!itemId) throw new Error('Selection range is unavailable')
                            if (selected) next[itemId] = true
                            else delete next[itemId]
                        }
                        if (Object.keys(next).length > 1024) this.announceSelection('selection.limit')
                        else {
                            this.selectedItemId = next
                            this.announceSelection('selection.count', { count: Object.keys(next).length })
                        }
                    } catch (error) { if (revision === this.resultsRevision && selectionRevision === this.selectionRevision) this.showError(error) }
                    finally { if (revision === this.resultsRevision && selectionRevision === this.selectionRevision) { this.isSelecting = false; this.trimCatalogPages(); this.rebuildCatalogBuffer() } }
                    return
                }
                this.selectionAnchorPosition = position
            }
            const targetSelected = !this.selectedItemId[id]
            const ids = this.resultBuffer.map(result => result.item.id)
            const anchor = ids.indexOf(this.selectionAnchorId)
            const target = ids.indexOf(id)
            const affected = event.shiftKey && anchor >= 0 && target >= 0
                ? ids.slice(Math.min(anchor, target), Math.max(anchor, target) + 1) : [id]
            const next = { ...this.selectedItemId }
            for (const affectedId of affected) {
                if (targetSelected) next[affectedId] = true
                else delete next[affectedId]
            }
            if (Object.keys(next).length > 1024) {
                this.announceSelection('selection.limit')
                return
            }
            this.selectedItemId = next
            if (!event.shiftKey || anchor < 0) this.selectionAnchorId = id
            this.announceSelection('selection.count', { count: Object.keys(next).length })
        },

        async selectCurrentResults() {
            if (this.isSelecting) return
            this.selectMode = true
            if (this.stableCatalog) {
                const revision = this.resultsRevision
                const selectionRevision = ++this.selectionRevision
                const count = Math.min(1024, this.catalogMatchingCount)
                this.isSelecting = true
                try {
                    await this.ensureCatalogRange(0, count)
                    if (revision !== this.resultsRevision || selectionRevision !== this.selectionRevision) return
                    const ids = Array.from({ length: count }, (_, i) => this.catalogResultAt(i)?.item.id)
                    if (ids.some(id => !id)) throw new Error('Selection range is unavailable')
                    this.selectedItemId = Object.fromEntries(ids.map(id => [id, true]))
                    this.selectionAnchorId = ids[0] || null
                    this.selectionAnchorPosition = count ? 0 : null
                    this.announceSelection('selection.firstSelected', { count })
                } catch (error) { if (revision === this.resultsRevision && selectionRevision === this.selectionRevision) this.showError(error) }
                finally { if (revision === this.resultsRevision && selectionRevision === this.selectionRevision) { this.isSelecting = false; this.trimCatalogPages(); this.rebuildCatalogBuffer() } }
                return
            }
            // This action explicitly replaces the selection with the current
            // page, or with the labelled first 1024 search results.
            const ids = this.resultBuffer.slice(0, 1024).map(result => result.item.id)
            this.selectedItemId = Object.fromEntries(ids.map(id => [id, true]))
            this.selectionAnchorId = ids[0] || null
            this.announceSelection(this.resultBuffer.length > 1024 ? 'selection.firstSelected' : 'selection.count', { count: ids.length })
        },

        clearSelection(announce = true) {
            this.selectionRevision += 1
            this.isSelecting = false
            this.selectedItemId = {}
            this.selectionAnchorId = null
            this.selectionAnchorPosition = null
            this.announceSelection(announce ? 'selection.cleared' : '')
        },

        pruneSelection() {
            // A filter change must never silently leave hidden images selected.
            const visible = new Set(this.resultBuffer.map(result => result.item.id))
            this.selectedItemId = Object.fromEntries(Object.keys(this.selectedItemId)
                .filter(id => visible.has(id)).map(id => [id, true]))
            this.selectionAnchorId = null
            this.announceSelection('selection.count', { count: Object.keys(this.selectedItemId).length })
        },

        announceSelection(key, params = {}) {
            this.selectionMessageKey = key
            this.selectionMessageParams = params
            this.selectionMessage = key ? this.t(key, params) : ''
        },

        onGridKeydown(event) {
            if (event.key === 'Escape') {
                this.exitSelectionMode()
                event.preventDefault()
            } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a') {
                this.selectCurrentResults()
                event.preventDefault()
            }
        },

        formatDuration(durationMs) {
            if (!Number.isFinite(durationMs)) {
                return "--"
            }

            return `${Math.round(durationMs)} ms`
        }
    }
})

app.use(vuetify)
app.mount('#app')
