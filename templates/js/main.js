new Veu({
	el:"#img",
	methods:{
		getimg: function(){
			url = "images/0.png"
			axios.get(url)
			.then(res => alert("get"))
		},
		
	}
	

})