import natsort from 'https://cdn.jsdelivr.net/npm/natsort@2.0.3/+esm'


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
            model_name: "ViT-L-14-336",
            pretrained: "openai",
            showedItemIndex: 0,
            aesthetic_quality_beta: 0.00,
            aesthetic_quality_range: [0, 10],
            padding_top: 0,
            padding_bottom: 500,
            item_height:258,

            /**
             * @type {ResultItem[]}
             */
            resultBuffer:[],

            /**
             * @type {DisplayItem[]}
             */
            displayItems:[]
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

            return promise.then(array => {

                console.log('ソート前' + JSON.stringify(array))

                //並び替え
                array = array.sort(util.cancatComparator(
                    (a, b) => b.score - a.score,
                    (a, b) => natsort.default()(a.item.name, b.item.name)
                ))

                console.log('ソート後' + JSON.stringify(array))

                //バッファに登録
                this.resultBuffer = array
            })
        },



        //テキストから検索するボタンの動作
        textSearchButton() {

            this.setBuffer(repository.searchText(this.model_name, this.pretrained, this.text, this.aesthetic_quality_beta, this.aesthetic_quality_range))
            .then(this.initImage);
        },

        //画像から検索するボタンの動作
        imagesSearchButton() {

            let selectedId = [...this.displayItems].filter(item => item.selected).map(item => item.id)

            this.setBuffer(repository.searchImage(this.model_name, this.pretrained, selectedId, this.aesthetic_quality_beta, this.aesthetic_quality_range))
            .then(this.initImage);
        },

        //画像名前から検索するボタンの動作
        nameSearchButton() {

            this.setBuffer(repository.searchName(this.model_name, this.pretrained, this.text, this.isRegexp, this.aesthetic_quality_beta, this.aesthetic_quality_range))
            .then(this.initImage);
        },

        //画像をコピーするボタンの動作
        copyImagesButton() {

            console.log("未実装")
        },

    }
})

app.use(vuetify)
app.mount('#app')