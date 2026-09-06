export const MAX_UPLOAD_BYTES = 64 * 1024 * 1024

/**
 * File object を base64 文字列に変換する
 *
 * @param {File} file
 * @returns {Promise<string>}
 */
export function fileToBase64(file) {
    if (file.size > MAX_UPLOAD_BYTES) {
        return Promise.reject(new Error('検索に使う画像は64MiB以内で指定してください。画像を縮小して再試行してください。'))
    }
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = /** @type {string} */(reader.result);
            resolve(result);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

