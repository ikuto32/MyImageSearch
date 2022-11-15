var app = new Vue({
	el: '#app',
	data: {
	  message: 'Hello Vue!'
	}
})

new Vue({
	el:"#img",
	methods:{
		getimg: function(){
			url = "images/0.png"
			axios.get(url)
			.then(res => alert(res))
		},
	}
})