import natsort from 'https://cdn.jsdelivr.net/npm/natsort@2.0.3/+esm'
import jszip from 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm'


import * as repository from "./modules/repository.js"
import * as util from "./util.js"


/**
 * @typedef {{id: string, score: number, img_name: string, img_small: string, img_original: string, selected: boolean}} DisplayItem
 */

const vuetify = Vuetify.createVuetify()

const app = Vue.createApp({
    el:"#app",
    data() {
        return {
            load_size: 50,
            message: "test",
            text: "",
            isShowSetting: false,
            isRegexp: false,
            numCols: 6,
            numRows: 10,
            model_name: "ViT-L-14",
            pretrained: "openai",
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

            ratingOptions: ["general", "questionable", "sensitive", "explicit"],
            ratingFilter: ["general", "questionable", "sensitive", "explicit"],

            /**
             * @type {{[itemId: string]: boolean}}
             */
            dialogStates:{},
        }
    },
    mounted() {
        // window.addEventListener("scroll", this.updateImageFromScroll)
        window.addEventListener("load", this.init)
    },
    watch:{
        ratingFilter(){
            this.applyRatingFilterToBuffer()
            this.initImage()
        },
        model_name(){
            this.onModelChange()
        },
        pretrained(){
            this.onModelChange()
        }
    },
    methods:{

        /**
         * 初期化する
         */
        init() {

            this.ensureRatingMap()
            .then(this.initBuffer)
            .then(this.initImage)
        },

        onModelChange(){
            this.ensureRatingMap()
            .then(() => {
                this.applyRatingFilterToBuffer()
                this.initImage()
            })
        },

        /**
         * 表示画像をリセットして、一部を表示する
         */
        initImage() {

            //現在表示されているものを削除
            this.displayItems = []
            this.selectedItemId = {}
            this.showedItemIndex = 0;
            //一部表示
            this.sliceShowImg(0, this.numRows * this.numCols)
            this.padding_bottom = Math.max(this.item_height * this.resultBuffer.length / this.numCols - this.padding_top, 0);
        },

        /**
         * バッファを初期化する
         *
         * @return {Promise<void>}
         */
        initBuffer() {

            //画像項目をページ指定で取得する（デフォルトは最初の1万件）
            return repository.getImageItemsByPage(0)
            .then(objs => {

                //バッファに登録
                /**
                 * @type {repository.ResultItem[]}
                 */
                this.rawResultBuffer = objs.map(obj => ({

                    item: obj,
                    score: 0
                }));
                this.applyRatingFilterToBuffer()
            })
        },

        getRatingMapKey() {
            return `${this.model_name}-${this.pretrained}`
        },

        ensureRatingMap() {
            const ratingKey = this.getRatingMapKey()

            if(this.ratingMaps[ratingKey]) {
                return Promise.resolve()
            }

            return repository.getImageRatings(this.model_name, this.pretrained)
            .then((ratings) => {
                this.ratingMaps = { ...this.ratingMaps, [ratingKey]: ratings }
            })
        },

        getCurrentRatingMap() {
            const ratingKey = this.getRatingMapKey()
            return this.ratingMaps[ratingKey] || {}
        },

        applyRatingFilter(resultList) {
            const allowedRatings = new Set(this.ratingFilter)
            const ratingMap = this.getCurrentRatingMap()

            return resultList.filter(result => {
                const rating = ratingMap[result.item.id] || ""

                if(rating === "") {
                    return true
                }

                return allowedRatings.has(rating)
            })
        },

        applyRatingFilterToBuffer() {
            this.resultBuffer = this.applyRatingFilter(this.rawResultBuffer)
        },


        //下までスクロールすると、次の画像を読み込む。
        updateImageFromScroll(e) {
            // 現在のスクロール値
            let scrollY = Math.max(e.target.scrollTop - 2 * this.item_height, 0);

            console.log("スクロール: " + scrollY + " / " + (this.item_height * this.resultBuffer.length / this.numCols));

            while (scrollY - this.padding_top > this.item_height){
                this.padding_top += this.item_height;
                this.padding_bottom = (this.item_height * this.resultBuffer.length / this.numCols) - this.padding_top;
                this.showNextImg();
            }
            while (scrollY - this.padding_top <= -this.item_height){
                this.padding_top -= this.item_height;
                this.padding_bottom = (this.item_height * this.resultBuffer.length / this.numCols) - this.padding_top;
                this.showPrevImg();
            }
        },



        /**
         * バッファ上の画像項目を逐次、描画する
         */
        showNextImg(){
            if(this.resultBuffer.length < this.showedItemIndex)
            {
                console.log("これ以上描画できません。")
                return
            }

            //今回、追加表示する範囲
            let start = Math.min(this.showedItemIndex + this.numCols, this.resultBuffer.length)
            let end = Math.min(start + this.numRows * this.numCols, this.resultBuffer.length)
            console.log("下に追加：先頭: " + JSON.stringify(start) + "後尾:" + JSON.stringify(end))

            //不足した情報を追加して、表示
            this.displayItems = []
            this.sliceShowImg(start, end)
            //最後のインデックス
            this.showedItemIndex = start
        },

        showPrevImg(){
            if(0 > this.showedItemIndex)
            {
                console.log("これ以上描画できません。")
                return
            }

            //今回、追加表示する範囲
            let start = Math.max(this.showedItemIndex - this.numCols, 0)
            let end = Math.min(start + this.numRows * this.numCols, this.resultBuffer.length)
            console.log("上に追加：先頭: " + JSON.stringify(start) + "後尾:" + JSON.stringify(end))

            //不足した情報を追加して、表示
            this.displayItems = []
            this.sliceShowImg(start, end)
            //最後のインデックス
            this.showedItemIndex = start
        },

        sliceShowImg(start, end){
            this.resultBuffer.slice(start, end).forEach(result => {

                console.log("表示: " + JSON.stringify(result))

                /**
                 * @type {DisplayItem[]}
                 */
                this.displayItems.push({
                    id: result.item.id,
                    score: result.score,
                    tags: result.item.tags,
                    img_name: result.item.name,
                    img_small: repository.getImageSmallUrl(result.item.id),
                    img_original: repository.getImageOriginalUrl(result.item.id),
                    selected: false
                });
            });
        },


        /**
         * 検索結果をバッファに登録する
         * @param {Promise<repository.ResultItem[]>} promise
         * @return {Promise<void>}
         */
        setBuffer(promise) {

            return this.ensureRatingMap()
            .then(() => promise)
            .then(result => {
                const clientStart = performance.now()

                let array = result.list
                console.log('結果', result)
                console.log('ソート前' + JSON.stringify(array))

                //並び替え
                array = array.sort(util.cancatComparator(
                    (a, b) => b.score - a.score,
                    (a, b) => natsort.default()(a.item.name, b.item.name)
                ))

                console.log('ソート後' + JSON.stringify(array))

                //バッファに登録
                this.search_query = result.search_query
                this.rawResultBuffer = array
                this.applyRatingFilterToBuffer()
                this.clientDurationMs = performance.now() - clientStart
            })
        },



        //テキストから検索するボタンの動作
        textSearchButton() {
            const searchStart = performance.now()
            this.isSearching = true
            const searchPromise = repository.searchText(this.model_name, this.pretrained, this.text, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name)
            .then(result => {
                this.searchDurationMs = performance.now() - searchStart
                return result
            })

            this.setBuffer(searchPromise)
            .then(this.initImage)
            .finally(() => { this.isSearching = false });
        },

        //画像から検索するボタンの動作
        imagesSearchButton() {
            const searchStart = performance.now()
            this.isSearching = true

            let selectedId = Object.keys(this.selectedItemId)

            const searchPromise = repository.searchImage(this.model_name, this.pretrained, selectedId, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name)
            .then(result => {
                this.searchDurationMs = performance.now() - searchStart
                return result
            })

            this.setBuffer(searchPromise)
            .then(this.initImage)
            .finally(() => { this.isSearching = false });
        },

        //アップロード画像から検索するボタンの動作
        uploadImageSearchButton() {
            if(!this.uploadFile) {
                return;
            }
            const searchStart = performance.now()
            this.isSearching = true
            const searchPromise = repository.searchUploadImage(this.model_name, this.pretrained, this.uploadFile)
            .then(result => {
                this.searchDurationMs = performance.now() - searchStart
                return result
            })

            this.setBuffer(searchPromise)
            .then(() => { this.uploadFile = null })
            .then(this.initImage)
            .finally(() => { this.isSearching = false });
        },

        //画像名前から検索するボタンの動作
        nameSearchButton() {
            const searchStart = performance.now()
            this.isSearching = true
            const searchPromise = repository.searchName(this.model_name, this.pretrained, this.text, this.isRegexp, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name)
            .then(result => {
                this.searchDurationMs = performance.now() - searchStart
                return result
            })

            this.setBuffer(searchPromise)
            .then(this.initImage)
            .finally(() => { this.isSearching = false });
        },

        allDownloadImagesButton() {
            const selectedIds = this.resultBuffer.map(result => result.item.id);

            fetch('/download_images_zip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ params: { ids: selectedIds } })
            })
            .then(response => response.blob())
            .then(blob => {
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'images.zip';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            })
            .catch(err => console.error('ZIPダウンロードエラー:', err));
        },

        //乱数から検索するボタンの動作
        randomSearchButton() {
            const searchStart = performance.now()
            this.isSearching = true
            const searchPromise = repository.searchRandom(this.model_name, this.pretrained, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name)
            .then(result => {
                this.searchDurationMs = performance.now() - searchStart
                return result
            })

            this.setBuffer(searchPromise)
            .then(this.initImage)
            .finally(() => { this.isSearching = false });
        },

        //クエリから検索するボタンの動作
        querySearchButton() {
            const searchStart = performance.now()
            this.isSearching = true
            const searchPromise = repository.searchQuery(this.model_name, this.pretrained, this.search_query, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name)
            .then(result => {
                this.searchDurationMs = performance.now() - searchStart
                return result
            })

            this.setBuffer(searchPromise)
            .then(this.initImage)
            .finally(() => { this.isSearching = false });
        },

        //クエリにテキストの特徴を足して検索するボタンの動作
        addTextFeaturesButton() {
            const searchStart = performance.now()
            this.isSearching = true
            const searchPromise = repository.addTextFeatures(this.model_name, this.pretrained, this.text, this.search_query, this.features_strength, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name)
            .then(result => {
                this.searchDurationMs = performance.now() - searchStart
                return result
            })

            this.setBuffer(searchPromise)
            .then(this.initImage)
            .finally(() => { this.isSearching = false });
        },

        //テキストからタグを検索するボタンの動作
        tagSearchButton() {
            const searchStart = performance.now()
            this.isSearching = true
            const searchPromise = repository.searchTags(this.model_name, this.pretrained, this.text, this.isRegexp, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name)
            .then(result => {
                this.searchDurationMs = performance.now() - searchStart
                return result
            })

            this.setBuffer(searchPromise)
            .then(this.initImage)
            .finally(() => { this.isSearching = false });
        },

        // style_cluster から検索するボタンの動作
        styleClusterSearchButton() {
            const searchStart = performance.now()
            this.isSearching = true
            const searchPromise = repository.searchStyleCluster(this.model_name, this.pretrained, this.text, this.isRegexp, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name)
            .then(result => {
                this.searchDurationMs = performance.now() - searchStart
                return result
            })

            this.setBuffer(searchPromise)
            .then(this.initImage)
            .finally(() => { this.isSearching = false });
        },

        openDialog(item) {
            this.setDialogState(item.id, true)
            this.fetchImageMetadata(item)
        },

        setDialogState(itemId, value) {
            this.dialogStates = { ...this.dialogStates, [itemId]: value }
        },

        fetchImageMetadata(item) {
            if (this.imageMeta[item.id]) {
                return
            }

            repository.getImageMetadata(this.model_name, this.pretrained, item.id)
            .then(meta => {
                this.imageMeta = { ...this.imageMeta, [item.id]: meta }
            })
        },

        getImageMetadata(itemId) {
            return this.imageMeta[itemId] || { tags: "", style_cluster: "", rating: "", aesthetic_quality: 0 }
        },

        onSelectItem(event) {

            const index = event.currentTarget.dataset.index;
            const item = this.displayItems[index];

            if(this.selectedItemId[item.id]) {
                delete this.selectedItemId[item.id];
            } else {
                this.selectedItemId[item.id] = true;
            }
        }
    }
})

app.use(vuetify)
app.mount('#app')
