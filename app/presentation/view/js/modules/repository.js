import axios from "../../vendor/axios/axios.min.js"
import { fileToBase64 } from "./util.js"
export { MAX_UPLOAD_BYTES } from "./util.js"

/**
 * 画像項目。タグや評価値は getImageMetadata で画像ごとに取得する
 * @typedef {{id:string, name:string, rating:string}} ImageItem
 *
 * 検索結果
 * @typedef {{item:ImageItem, score:number}} ResultItem
 *
 * 検索結果の一覧と、同じ検索を再実行するための保存クエリ
 * @typedef {{list:ResultItem[], search_query:string}} SearchResults
 *
 * すべての検索に共通する設定
 * @typedef {{model_name:string, pretrained:string, aesthetic_quality_beta:number,
 *   aesthetic_quality_range:number[], aesthetic_model_name:string, result_size:number}} SearchSettings
 */

/**
 * 利用可能な検索モデル一覧を取得する。
 *
 * @return {Promise<{model_name:string, pretrained:string}[]>}
 */
export function getModelItems() {
    return axios.get("/model_item").then(res => res.data)
}

/**
 * 画像項目のIDから縮小画像のURLを取得する
 *
 * @param {string} itemId
 * @return {string}
 */
export function getImageSmallUrl(itemId) {
    return `/image/${itemId}/small`
}

/**
 * 画像項目のIDから元の画像のURLを取得する
 *
 * @param {string} itemId
 * @return {string}
 */
export function getImageOriginalUrl(itemId) {
    return `/image/${itemId}/original`
}

/**
 * ページ指定で画像項目の一覧を取得する。
 * includeTotalを指定すると、戻り値の配列にモデル全体の件数(totalCount)と
 * 区分に一致する件数(matchingCount)が付く。
 *
 * @param {number} page 0始まりのページ番号
 * @param {number} pageSize 取得する件数
 * @param {string} modelName
 * @param {string} pretrained
 * @param {boolean} includeTotal
 * @param {string[]} [ratings] 絞り込む画像区分。省略時は絞り込まない
 * @return {Promise<ImageItem[] & {totalCount?: number, matchingCount?: number}>}
 */
export function getImageItemsByPage(page, pageSize, modelName, pretrained, includeTotal = false, ratings) {
    const params = { page, size: pageSize, model_name: modelName, pretrained }
    if (includeTotal) params.include_total = 1
    if (ratings !== undefined) params.ratings = JSON.stringify(ratings)
    return axios.get("/image_item", { params }).then(res => {
        const total = res.headers["x-catalog-total"]
        if (total !== undefined && total !== "") res.data.totalCount = Number(total)
        const matching = res.headers["x-matching-total"]
        if (matching !== undefined && matching !== "") res.data.matchingCount = Number(matching)
        return res.data
    })
}

/**
 * 画像のタグ・評価などのメタデータを取得する
 *
 * @return {Promise<{tags:string, style_cluster:string, rating:string, aesthetic_quality:number|null}>}
 */
export function getImageMetadata(modelName, pretrained, itemId) {
    return axios.get(`/image_meta/${itemId}`, {
        params: { model_name: modelName, pretrained },
    }).then(res => res.data)
}

/**
 * 画像項目を検索する。
 *
 * kindごとに、共通の設定へ次のパラメータを加える:
 * text {text} / image {id: 画像IDの配列} / random {} / query {search_query}
 * queryaddtext {text, search_query, features_strength}
 * name・tags・style_cluster {text, is_regexp}
 *
 * @param {'text'|'image'|'random'|'query'|'queryaddtext'|'name'|'tags'|'style_cluster'|'uploadimage'} kind
 * @param {SearchSettings & Object} params
 * @return {Promise<SearchResults>}
 */
export function search(kind, params) {
    return axios.post(`/search/${kind}`, { params }).then(res => res.data)
}

/**
 * アップロードされた画像で検索する
 *
 * @param {File} file
 * @param {SearchSettings} params
 * @return {Promise<SearchResults>}
 */
export async function searchUploadImage(file, params) {
    const dataUrl = await fileToBase64(file)
    return search("uploadimage", {
        ...params,
        base64: dataUrl.split(",")[1],
        // ブラウザが形式を判別できない場合は、サーバーの既定値に任せる。
        content_type: file.type || undefined,
    })
}
