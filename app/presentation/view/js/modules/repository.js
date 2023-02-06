

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

    return axios.get(`/image_item`).then(res => res.data)
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
 * 画像項目を文字列で検索する
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

    return axios.post("/search/text", payload).then(res => res.data) 
}


/**
 * 画像項目を画像で検索する
 * 
 * @param {string} modelName
 * @param {string} pretrained
 * @param {string[]} itemId_list
 * 
 * @return {Promise<ResultPair[]>}
 */
export async function searchImage(modelName, pretrained, itemId_list) {

    let payload = {

        params:{
            model_name: modelName, 
            pretrained: pretrained, 
            id : itemId_list
        }
    }

    return axios.post("/search/image", payload).then(res => res.data) 
}


/**
 * 画像項目を名前で検索する
 * 
 * @param {string} isRegexp
 * @param {string} text
 * 
 * @return {Promise<ResultPair[]>}
 */
export async function searchName(itemId_list, is_regexp) {

    let payload = {

        params:{
            text : itemId_list,
            is_regexp : is_regexp.toString()
        }
    }

    return axios.post("/search/name", payload).then(res => res.data) 
}