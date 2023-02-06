
import Vue from "https://cdn.jsdelivr.net/npm/vue@2/dist/vue.esm.browser.js"

import * as repository from "./modules/repository.js"


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

		/**
		 * @type {{id: string, score: number}[]}
		 */
		resultBuffer:[],

		/**
		 * @type {{id: string, score: number, img_name: string, img: string, selected: boolean}[]}
		 */
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

			//画像項目をすべて取得する
			return repository.getImageItems()
			.then(objs => {

				//バッファに登録
				this.resultBuffer = objs.map(obj => ({

					id: obj.id,
					score: 0
				}))
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
			
			
			if(this.resultBuffer.length < this.showedItemIndex)
			{
				console.log("これ以上描画できません。")
				return
			}

			//今回、追加表示する範囲
			let start = this.showedItemIndex
			let end = Math.min(start + this.load_size - 1, this.resultBuffer.length)


			//不足した情報を追加して、表示
			this.resultBuffer.slice(start, end).forEach(obj => {

				repository.getItem(obj.id)
				.then(item => {

					this.displayItems.push({

						id: obj.id,
						score: obj.score,
						img: repository.getImageUrl(obj.id),
						img_name: item.name,
						selected: false
					}); 
				})
				
			});


			
			//最後のインデックス
			this.showedItemIndex = end + 1;
			

		},

		/**
		 * 検索結果をバッファに登録する
		 * @param {Promise<repository.ResultPair[]>} promise
		 * @return {Promise<void>}
		 */
		setBuffer(promise) {

			return promise.then(objs => Promise.all(objs.map(async obj => {

				return {
					
					id: obj.id,
					score: obj.score
				}

			}))).then(array => {

				//並び替え
				array = array.sort((a, b) => b.score - a.score)
				console.log(`ソート後 ${JSON.stringify(array)}`)

				//バッファに登録
				this.resultBuffer = array
			})
		},



		//テキストから検索するボタンの動作
		textSearchButton() {

			this.setBuffer(repository.searchText(this.model_name, this.pretrained, this.text))
			.then(this.initImage);
			
		},

		//画像から検索するボタンの動作
		imagesSearchButton() {

			let selectedId = [...this.displayItems].filter(item => item.selected).map(item => item.id)

			this.setBuffer(repository.searchImage(this.model_name, this.pretrained, selectedId))
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