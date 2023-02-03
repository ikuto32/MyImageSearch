
import Vue from "https://cdn.jsdelivr.net/npm/vue@2/dist/vue.esm.browser.js"
import axios from "https://cdn.jsdelivr.net/npm/axios@1.3.1/+esm"

import * as repo from "./modules/repository.js"



new Vue({
	el:"#app",
	data:{
		load_size: 10,
		message: "test",
		text: "",
		isShowSetting: false,
		isRegexp: false,
		image_size: 250,
		scrollY: 0,
		model_name: "ViT-B-32",
		pretrained: "laion2b_s34b_b79k",
		items:[

		]
	},
	mounted() {
		window.addEventListener("scroll", this.updateImageFromScroll)
		//window.addEventListener("load", this.getimgs)
    },
	methods:{

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
				//this.getimgs();
			}

        },

		//サーバから画像を取得する
		getJsonToImgs(json){
			console.log(json)

			for (let i in json){

				//idの画像を問い合わせ
				axios.get("/image_item/"+json[i]["id"]+"/image")
				.then(imgRes => {
					let img = imgRes.data
					let score = json[i]["score"]

					console.log("image call: " + json[i]["id"])
					
					axios.get("/image_item/"+json[i]["id"])
					.then(response => {
						let display_name = response.data["name"]
						console.log("item call: " + response.data["id"] + " = " + json[i]["id"])

						this.items.push({
							img: img,
							img_name: display_name,
							score: score,
							selected: false
						})
					})
				})
			}
		},

		//テキストから検索するボタンの動作
		textSearchButton() {
			let params = {params:{model_name : this.model_name, pretrained: this.pretrained, text : this.text}}
			axios.get("/search/text", params).then(response => {
				this.getJsonToImgs(response.data)
			});
			
		},

		//画像から検索するボタンの動作
		imagesSearchButton() {
			let selected_images = []
			for (let i in this.items){
				let item = this.items[i]
				console.log(item.selected)
				if (item.selected){
					selected_images.push(item.img_name)
				}
			}
			let param = {"trigger" : "ImageSearch", "meta_names" : selected_images.join(',')}
			console.log(param)
			axios.post("/search", param)
			this.initImages()
		},

		//画像名前から検索するボタンの動作
		nameSearchButton() {
			let param = {"trigger" : "NameSearch", "text" : this.text, "trueRegexp" : this.isRegexp.toString()}
			axios.post("/search", param)
			this.initImages()
		},

		//画像をコピーするボタンの動作
		copyImagesButton() {
			let selected_images = []
			for (let i in this.items){
				let item = this.items[i]
				console.log(item.selected)
				if (item.selected){
					selected_images.push(item.img_name)
				}
			}
			let param = {"trigger" : "copyImages", "meta_names" : selected_images.join(',')}
			console.log(param)
			axios.post("/search", param)
			this.initImages()
		},

		//画像の表示を初期化する
		initImages() {
			this.items = []
			this.count = 0
			//this.getimgs()
		},
	}
})