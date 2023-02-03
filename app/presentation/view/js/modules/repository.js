

import axios from "https://cdn.jsdelivr.net/npm/axios@1.3.1/+esm"

/**
 * 画像項目
 * @typedef {{id:string, name:string}} ImageItem 
 * 
 * 検索結果
 * @typedef {{id:string, score:number}} ResultPair
 */


/**
 * 画像項目のIDから画像のURLを取得する
 * 
 * @param {string} itemId
 * 
 * @return {string}
 */
export function getImageUrl(itemId) {

    return `/image_item/${itemId}/image`
}


/**
 * 画像項目の一覧を取得する
 * 
 * @return {Promise<ImageItem[]>}
 */
export async function getImageItems() {

    return axios.get(`/image_item/`).then(res => res.data)
}

/**
 * 画像項目を取得する
 * 
 * @param {string} itemId
 * 
 * @return {Promise<ImageItem>}
 */
export async function getItem(itemId) {

    return axios.get(`/image_item/${itemId}`).then(res => res.data)
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
export async function searchText(modelName, pretrained, text) {

    let payload = {

        params:{
            model_name: modelName, 
            pretrained: pretrained, 
            text : text
        }
    }

    return axios.get("/search/text", payload).then(res => res.data) 
}