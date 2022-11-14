

var ws = new WebSocket("ws://" + location.host + "/app/connect");

/*
window.addEventListener("load", function(e){
	
	
});
*/


//Java側に指定したスクリプトの実行を要求する
function fire(paramName, eventName, paramObj) {

	console.log("スクリプトイベント発生");

	paramObj[paramName] = encodeURI(eventName)

	let param = Object.entries(paramObj)
		.map(([key, val]) => encodeURI(`${key}=${val}`))
		.join('&');


	let url = "./trigger";
	console.log(paramObj);

	return fetch(url, {
		method: "POST",
		body: JSON.stringify(param),
      	contentType: 'application/json'
	  });
}