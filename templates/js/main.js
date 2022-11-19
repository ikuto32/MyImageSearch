new Vue({
	el:"#app",
	data:{
		message: "test",
		count: 0,
		selectedTab: "search",
		image_size: 250,
		items:[
			{
				img: "images/0.png",
				score: 0,
				selected: false
			},
			{
				img: "images/1.png",
				score: 0.55,
				selected: false
			}
		]
	},
	methods:{
		getimgs: function (){
			console.log(this.count+" to "+(this.count+10))
			axios.get("/images?min="+this.count+"&max="+(this.count+10))
			.then(response => {
				console.log(response.data)
				for (var metaName in response.data){
					score = response.data[metaName]
					console.log(score)
					this.items.push({
						img: "images/"+metaName,
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
		
	}
})