var app = new Vue({
	el: '#app',
	data: {
	  message: 'Hello Vue!'
	}
})

new Vue({
	el:"#itemArea",
	data:{
		message: "test",
		count: 0,
		items:[
			{
				img: "images/0.png",
				score: 0
			},
			{
				img: "images/1.png",
				score: 0.55
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
					this.items.push({img: "images/"+metaName, score: score})
				}
			});
			this.count += 10
		},
	}
})