

import axios from "https://cdn.jsdelivr.net/npm/axios@1.3.1/+esm"

/**
 * 画像項目
 * @typedef {{id:string, name:string}} ImageItem 
 * 
 * 検索結果
 * @typedef {{id:string, score:number}} ResultPair
 */


/**
 * 画像項目の一覧を取得する
 * 
 * @return {Promise<ImageItem[]>}
 */
export function getImageItems() {

    return axios.get(`/image_item/`).then(res => res.json())
}


/**
 * 指定されたimgタグに画像を適用する
 * @param {HTMLImageElement} imgElement
 * @param {string} itemId
 */
export function applyImage(imgElement, itemId) {

    imgElement.src = `/image_item/${itemId}/image`
}


/**
 * 画像項目の一覧を取得する
 * 
 * @param {string} modelName
 * @param {string} pretrained
 * @param {string} text
 * 
 * @return {Promise<ResultPair[]>}
 */
export function searchText(modelName, pretrained, text) {

    let payload = {

        params:{
            model_name: modelName, 
            pretrained: pretrained, 
            text : text
        }
    }

    return axios.get("/search/text", payload).then(res => res.json()) 
}