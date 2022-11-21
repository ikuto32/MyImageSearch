new Vue({
	el:"#app",
	data:{
		message: "test",
		text: "",
		count: 0,
		selectedTab: "search",
		image_size: 250,
		items:[

		]
	},
	methods:{
		getimgs: function (){
			console.log(this.count+" to "+(this.count+10))
			axios.get("/images?min="+this.count+"&max="+(this.count+10))
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
			this.count += 10
		},
		isSelect: function (tab){
			console.log(tab)
			this.selectedTab = tab
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
		}
		
	}
})