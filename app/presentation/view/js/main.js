import natsort from 'https://cdn.jsdelivr.net/npm/natsort@2.0.3/+esm'
import jszip from 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm'


import * as repository from "./modules/repository.js"
import * as util from "./util.js"


/**
 * @typedef {{id: string, score: number, img_name: string, img: string, selected: boolean}} DisplayItem
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
            model_name: "ViT-SO400M-16-SigLIP-i18n-256",
            pretrained: "webli",
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
            selectedItemId:{}
        }
    },
    mounted() {
        // window.addEventListener("scroll", this.updateImageFromScroll)
        window.addEventListener("load", this.init)
    },
    methods:{

        /**
         * 初期化する
         */
        init() {

            this.initBuffer()
            .then(this.initImage)
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
            this.padding_bottom = this.item_height * this.resultBuffer.length / this.numCols - this.padding_top;
        },

        /**
         * バッファを初期化する
         *
         * @return {Promise<void>}
         */
        initBuffer() {

            //画像項目をすべて取得する
            return repository.getImageItems()
            .then(objs => {

                //バッファに登録
                /**
                 * @type {repository.ResultItem[]}
                 */
                this.resultBuffer = objs.map(obj => ({

                    item: obj,
                    score: 0
                }));
            })
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
                    img: repository.getImageUrl(result.item.id),
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

            return promise.then(result => {

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
                this.resultBuffer = array
            })
        },



        //テキストから検索するボタンの動作
        textSearchButton() {

            this.setBuffer(repository.searchText(this.model_name, this.pretrained, this.text, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name))
            .then(this.initImage);
        },

        //画像から検索するボタンの動作
        imagesSearchButton() {

            let selectedId = Object.keys(this.selectedItemId)

            this.setBuffer(repository.searchImage(this.model_name, this.pretrained, selectedId, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name))
            .then(this.initImage);
        },

        //アップロード画像から検索するボタンの動作
        uploadImageSearchButton() {
            if(!this.uploadFile) {
                return;
            }
            this.setBuffer(repository.searchUploadImage(this.model_name, this.pretrained, this.uploadFile))
            .then(() => { this.uploadFile = null })
            .then(this.initImage);
        },

        //画像名前から検索するボタンの動作
        nameSearchButton() {

            this.setBuffer(repository.searchName(this.model_name, this.pretrained, this.text, this.isRegexp, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name))
            .then(this.initImage);
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

            this.setBuffer(repository.searchRandom(this.model_name, this.pretrained, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name))
            .then(this.initImage);
        },

        //クエリから検索するボタンの動作
        querySearchButton() {

            this.setBuffer(repository.searchQuery(this.model_name, this.pretrained, this.search_query, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name))
            .then(this.initImage);
        },

        //クエリにテキストの特徴を足して検索するボタンの動作
        addTextFeaturesButton() {

            this.setBuffer(repository.addTextFeatures(this.model_name, this.pretrained, this.text, this.search_query, this.features_strength, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name))
            .then(this.initImage);
        },

        //テキストからタグを検索するボタンの動作
        tagSearchButton() {

            this.setBuffer(repository.searchTags(this.model_name, this.pretrained, this.text, this.isRegexp, this.aesthetic_quality_beta, this.aesthetic_quality_range, this.aesthetic_model_name))
            .then(this.initImage);
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