new Vue({
	el:"#app",
	data:{
		load_size: 10,
		message: "test",
		text: "",
		count: 0,
		isShowSetting: false,
		isRegexp: false,
		image_size: 250,
		items:[

		]
	},
	methods:{
		getimgs: function (){
			console.log(this.count+" to "+(this.count+this.load_size))
			axios.get("/images?min="+this.count+"&max="+(this.count+this.load_size))
			.then(response => {
				console.log(response.data)
				for (var metaName in response.data){
					score = response.data[metaName]["score"]
					img = response.data[metaName]["base64_img"]
					file_type = response.data[metaName]["file_type"]
					this.items.push({
						img: "data:image/"+file_type+";base64,"+img,
						img_name: metaName,
						score: score,
						selected: false
					})
				}
			});
			this.count += this.load_size
		},
		textSearchButton: function () {
			param = {"trigger" : "TextSearch", "text" : this.text}
			axios.post("/search", param)
		},
		imagesSearchButton: function () {
			let selected_images = []
			for (var i in this.items){
				item = this.items[i]
				console.log(item.selected)
				if (item.selected){
					selected_images.push(item.img_name)
				}
			}
			param = {"trigger" : "ImageSearch", "meta_names" : selected_images.join(',')}
			console.log(param)
			axios.post("/search", param)
		},
		nameSearchButton: function () {
			param = {"trigger" : "NameSearch", "text" : this.text, "trueRegexp" : this.isRegexp.toString()}
			axios.post("/search", param)
		},
		
	}
})