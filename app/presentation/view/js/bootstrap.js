// Keep startup failures visible outside Vue's v-cloak, including a missing
// language file or local vendor asset. A reload retries the original request.
const status = document.getElementById('startupStatus')
try {
    await import('./main.js')
    status?.remove()
} catch (error) {
    console.error('Image search startup failed:', error)
    if (status) {
        status.setAttribute('role', 'alert')
        status.replaceChildren()
        for (const [language, message] of [
            ['ja', '画面を読み込めませんでした。再読み込みして再試行してください。'],
            ['en', 'The interface could not be loaded. Reload to try again.'],
        ]) {
            const paragraph = document.createElement('p')
            paragraph.lang = language
            paragraph.textContent = message
            status.appendChild(paragraph)
        }
        const retry = document.createElement('button')
        retry.type = 'button'
        retry.textContent = '再読み込み / Reload'
        retry.addEventListener('click', () => window.location.reload())
        status.appendChild(retry)
        retry.focus()
    }
}
