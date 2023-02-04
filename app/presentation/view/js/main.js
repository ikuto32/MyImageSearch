
import Vue from "https://cdn.jsdelivr.net/npm/vue@2/dist/vue.esm.browser.js"
import axios from "https://cdn.jsdelivr.net/npm/axios@1.3.1/+esm"

import * as repository from "./modules/repository.js"
import * as util from "./modules/util.js"



new Vue({
	el:"#app",
	data:{
		load_size: 10,
		message: "test",
		text: "",
		isShowSetting: false,
		isRegexp: false,
		image_size: 250,
		model_name: "ViT-B-32",
		pretrained: "laion2b_s34b_b79k",
		showedItemIndex: 0,
		itemsBuffer:[],
		displayItems:[]
	},
	mounted() {
		window.addEventListener("scroll", this.updateImageFromScroll)
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
			this.showNextImg()
		},

		/**
		 * バッファを初期化する
		 * 
		 * @return {Promise<void>}
		 */
		initBuffer() {

			return repository.getImageItems()
			.then(objs => Promise.all(objs.map(obj => {

				//表示情報
				return {
					id: obj.id,
					score: 0,
					img_name: obj.name,
					img: repository.getImageUrl(obj.id),
					selected: false
				}

			}))).then(array => {

				//バッファに登録
				this.itemsBuffer = array
			})
		},


		//下までスクロールすると、次の画像を読み込む。
		updateImageFromScroll() {

			// ページ全体の高さ
			let pageHeight = document.body.clientHeight;

			// スクロールバー等を含まないブラウザの表示領域
			let viewportHeight = document.documentElement.clientHeight;
			
			// スクロールの最大値
			let scrollMaxY = pageHeight - viewportHeight;

			// 現在のスクロール値
			let scrollY = window.pageYOffset;

			console.log("スクロール: " + scrollY + " / " + scrollMaxY);

			//最後に近づいたら、更新
			if(scrollMaxY - scrollY < 20)
			{
				this.showNextImg();
			}

        },

		


		/**
		 * バッファ上の画像項目を逐次、描画する
		 */
		showNextImg(){
			
			if(this.itemsBuffer.length <= this.showedItemIndex)
			{
				console.log("これ以上描画できません。")
				return
			}


			let temp = this.showedItemIndex

			for(let i = 0; i < this.load_size; i++) {

				let displayItem = this.itemsBuffer[temp + i]
				if(displayItem == null) {
					break;
				}
				
				this.displayItems.push(displayItem)
				this.showedItemIndex++
			}

		},

		/**
		 * 検索結果をバッファに登録する
		 * @param {Promise<repository.ResultPair[]>} promise
		 * @return {Promise<void>}
		 */
		setBuffer(promise) {

			return promise.then(objs => Promise.all(objs.map(async obj => {

				//画像項目の情報を取得
				let item = await repository.getItem(obj.id)

				//表示する情報
				let displayItem = {
					id: obj.id,
					score: obj.score,
					img_name: item.name,
					img: repository.getImageUrl(obj.id),
					selected: false
				}

				console.log(`結果 ${JSON.stringify(displayItem)}`)

				return displayItem

			}))).then(array => {

				//並び替え
				array = array.sort((a, b) => b.score - a.score)
				console.log(`ソート後 ${JSON.stringify(array)}`)

				//バッファに登録
				this.itemsBuffer = array
			})
		},



		//テキストから検索するボタンの動作
		textSearchButton() {

			this.setBuffer(repository.searchText(this.model_name, this.pretrained, this.text))
			.then(this.initImage);
			
		},

		//画像から検索するボタンの動作
		imagesSearchButton() {

			let selectedItems = [...this.displayItems].filter(item => item.selected)

			this.setBuffer(repository.searchImage(this.model_name, this.pretrained, selectedItems))
			.then(this.initImage);
		},

		//画像名前から検索するボタンの動作
		nameSearchButton() {

			console.log("未実装")
		},

		//画像をコピーするボタンの動作
		copyImagesButton() {

			console.log("未実装")
		},

	}
})