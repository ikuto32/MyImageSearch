

import axios from "https://cdn.jsdelivr.net/npm/axios@1.3.1/+esm"
import { fileToBase64 } from "./util.js"

/**
 * 画像項目
 * @typedef {{id:string, name:string}} ImageItem 
 * 
 * 検索結果
 * @typedef {{item:ImageItem, score:number}} ResultItem
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
 * @return {Promise<ResultItem[]>}
 */
export async function searchText(modelName, pretrained, text, aesthetic_quality_beta, aesthetic_quality_range, aesthetic_model_name) {

    let payload = {

        params:{
            model_name: modelName, 
            pretrained: pretrained, 
            text : text,
            aesthetic_quality_beta: aesthetic_quality_beta,
            aesthetic_quality_range: aesthetic_quality_range,
            aesthetic_model_name: aesthetic_model_name
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
 * @return {Promise<ResultItem[]>}
 */
export async function searchImage(modelName, pretrained, itemId_list, aesthetic_quality_beta, aesthetic_quality_range, aesthetic_model_name) {

    let payload = {

        params:{
            model_name: modelName,
            pretrained: pretrained,
            id : itemId_list,
            aesthetic_quality_beta: aesthetic_quality_beta,
            aesthetic_quality_range: aesthetic_quality_range,
            aesthetic_model_name: aesthetic_model_name
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
 * @return {Promise<ResultItem[]>}
 */
export async function searchName(modelName, pretrained, itemId_list, is_regexp, aesthetic_quality_beta, aesthetic_quality_range, aesthetic_model_name) {

    let payload = {

        params:{
            model_name: modelName,
            pretrained: pretrained,
            text : itemId_list,
            is_regexp : is_regexp.toString(),
            aesthetic_quality_beta: aesthetic_quality_beta,
            aesthetic_quality_range: aesthetic_quality_range,
            aesthetic_model_name: aesthetic_model_name
        }
    }

    return axios.post("/search/name", payload).then(res => res.data)
}


/**
 * 画像項目を乱数で検索する
 * 
 * @param {string} modelName
 * @param {string} pretrained
 * @param {string[]} itemId_list
 * 
 * @return {Promise<ResultItem[]>}
 */
export async function searchRandom(modelName, pretrained, aesthetic_quality_beta, aesthetic_quality_range, aesthetic_model_name) {

    let payload = {

        params:{
            model_name: modelName,
            pretrained: pretrained,
            aesthetic_quality_beta: aesthetic_quality_beta,
            aesthetic_quality_range: aesthetic_quality_range,
            aesthetic_model_name: aesthetic_model_name
        }
    }

    return axios.post("/search/random", payload).then(res => res.data)
}


/**
 * 画像項目をクエリで検索する
 * 
 * @param {string} modelName
 * @param {string} pretrained
 * @param {string[]} itemId_list
 * 
 * @return {Promise<ResultItem[]>}
 */
export async function searchQuery(modelName, pretrained, search_query, aesthetic_quality_beta, aesthetic_quality_range, aesthetic_model_name) {

    let payload = {

        params:{
            model_name: modelName,
            pretrained: pretrained,
            search_query: search_query,
            aesthetic_quality_beta: aesthetic_quality_beta,
            aesthetic_quality_range: aesthetic_quality_range,
            aesthetic_model_name: aesthetic_model_name
        }
    }

    return axios.post("/search/query", payload).then(res => res.data)
}

/**
 * 画像項目をクエリにstrengthの強さ分テキストの特徴を足してから検索する
 * 
 * @param {string} modelName
 * @param {string} pretrained
 * @param {string[]} itemId_list
 * 
 * @return {Promise<ResultItem[]>}
 */
export async function addTextFeatures(modelName, pretrained, text, search_query, features_strength, aesthetic_quality_beta, aesthetic_quality_range, aesthetic_model_name) {

    let payload = {

        params:{
            model_name: modelName,
            pretrained: pretrained,
            text : text,
            search_query: search_query,
            features_strength: features_strength,
            aesthetic_quality_beta: aesthetic_quality_beta,
            aesthetic_quality_range: aesthetic_quality_range,
            aesthetic_model_name: aesthetic_model_name
        }
    }

    return axios.post("/search/queryaddtext", payload).then(res => res.data)
}

/**
 * 画像項目をタグで検索する
 * 
 * @param {string} modelName
 * @param {string} pretrained
 * @param {string[]} itemId_list
 * 
 * @return {Promise<ResultItem[]>}
 */
export async function searchTags(modelName, pretrained, itemId_list, is_regexp, aesthetic_quality_beta, aesthetic_quality_range, aesthetic_model_name) {

    let payload = {

        params:{
            model_name: modelName,
            pretrained: pretrained,
            text : itemId_list,
            is_regexp : is_regexp.toString(),
            aesthetic_quality_beta: aesthetic_quality_beta,
            aesthetic_quality_range: aesthetic_quality_range,
            aesthetic_model_name: aesthetic_model_name
        }
    }

    return axios.post("/search/tags", payload).then(res => res.data)
}

/**
 * アップロードされた画像で検索する
 *
 * @param {string} modelName
 * @param {string} pretrained
 * @param {File} file
 * @returns {Promise<ResultItem[]>}
 */
export async function searchUploadImage(modelName, pretrained, file) {
    const dataUrl = await fileToBase64(file)
    let payload = {
        params:{
            model_name: modelName,
            pretrained: pretrained,
            base64: dataUrl.split(',')[1],
            content_type: file.type
        }
    }
    return axios.post("/search/uploadimage", payload).then(res => res.data)
}

