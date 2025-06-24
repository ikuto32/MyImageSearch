/**
 * File object を base64 文字列に変換する
 *
 * @param {File} file
 * @returns {Promise<string>}
 */
export function fileToBase64(file) {
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

