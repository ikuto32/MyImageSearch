// Run with: node --test tests/test_i18n.cjs
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const crypto = require('node:crypto')
const test = require('node:test')
const viewPath = path.join(__dirname, '../app/presentation/view')
const read = file => fs.readFileSync(path.join(viewPath, file), 'utf8')
const ja = JSON.parse(read('lang/ja.json'))
const en = JSON.parse(read('lang/en.json'))

function flatten(object, prefix = '') {
    return Object.fromEntries(Object.entries(object).flatMap(([key, value]) =>
        typeof value === 'string' ? [[prefix + key, value]] : Object.entries(flatten(value, `${prefix}${key}.`))))
}

async function loadI18n(saved = null, storageUnavailable = false) {
    const document = { documentElement: { lang: '' }, title: '' }
    const writes = []
    const code = read('js/i18n.js').replace('export const i18n', 'const i18n').replaceAll('import.meta.url', '"http://localhost/js/i18n.js"')
    const i18n = await vm.runInNewContext(`(async () => { ${code}\nreturn i18n })()`, {
        URL, document,
        fetch: async url => ({ ok: true, json: async () => JSON.parse(read(`lang/${url.pathname.split('/').pop()}`)) }),
        localStorage: {
            getItem() { if (storageUnavailable) throw new Error('blocked'); return saved },
            setItem(key, value) { if (storageUnavailable) throw new Error('blocked'); writes.push([key, value]) },
        },
    })
    return { i18n, document, writes }
}

test('Japanese and English cover every static UI key and matching placeholders', () => {
    const japanese = flatten(ja), english = flatten(en)
    assert.deepEqual(Object.keys(japanese).sort(), Object.keys(english).sort())
    for (const [key, value] of Object.entries(japanese)) {
        assert.ok(value.trim(), key)
        assert.ok(english[key].trim(), key)
        const placeholders = text => [...text.matchAll(/\{(\w+)\}/g)].map(match => match[1]).sort()
        assert.deepEqual(placeholders(value), placeholders(english[key]), key)
    }
    const source = read('index.html') + read('js/main.js')
    for (const match of source.matchAll(/['"]((?:common|settings|selection|download|errors|results|image|rating|pagination|navigation|search|timing)\.[A-Za-z]\w*)['"]/g)) {
        assert.ok(Object.hasOwn(ja, match[1]), `Missing Japanese key: ${match[1]}`)
        assert.ok(Object.hasOwn(en, match[1]), `Missing English key: ${match[1]}`)
    }
})

test('locale restores across reloads and updates document language, title and component strings', async () => {
    const { i18n, document, writes } = await loadI18n('en')
    assert.equal(document.documentElement.lang, 'en')
    assert.equal(document.title, en['app.title'])
    assert.equal(i18n.vuetifyMessages.en.input.clear, 'Clear {0}')
    assert.equal(i18n.t('en', 'selection.count', { count: 17 }), '17 selected')
    i18n.setLocale('ja')
    assert.equal(document.documentElement.lang, 'ja')
    assert.equal(document.title, ja['app.title'])
    assert.deepEqual(writes.at(-1), ['myImageSearch.locale', 'ja'])
})

test('blocked storage or invalid saved locale does not prevent startup', async () => {
    assert.equal((await loadI18n(null, true)).document.documentElement.lang, 'ja')
    assert.equal((await loadI18n('unsupported')).document.documentElement.lang, 'ja')
})

test('translation parameters remain plain strings and missing keys can be diagnosed', async () => {
    const { i18n } = await loadI18n()
    assert.equal(i18n.t('en', 'image.openDetail', { name: '<img onerror=alert(1)>' }), 'Enlarge <img onerror=alert(1)>')
    assert.equal(i18n.t('en', 'missing.key'), 'missing.key')
    assert.equal(i18n.t('unsupported', 'common.close'), ja['common.close'])
})

test('startup failure displays a retry action outside the cloaked application', async () => {
    const makeNode = () => ({ children: [], appendChild(node) { this.children.push(node) }, setAttribute(key, value) { this[key] = value }, replaceChildren() { this.children = [] }, addEventListener(type, callback) { this[type] = callback }, focus() { this.focused = true }, remove() { this.removed = true } })
    const status = makeNode()
    let reloaded = false
    await vm.runInNewContext(`(async () => { ${read('js/bootstrap.js').replace("import('./main.js')", 'loadMain()')} })()`, {
        document: { getElementById: () => status, createElement: makeNode },
        loadMain: async () => { throw new Error('Language file unavailable: en') },
        console: { error() {} },
        window: { location: { reload() { reloaded = true } } },
    })
    assert.equal(status.role, 'alert')
    assert.ok(status.children.some(node => node.lang === 'ja' && node.textContent))
    assert.ok(status.children.some(node => node.lang === 'en' && node.textContent))
    const retry = status.children.at(-1)
    assert.equal(retry.type, 'button')
    assert.equal(retry.focused, true)
    retry.click()
    assert.equal(reloaded, true)
})

test('runtime entry assets use local files and bundled files match their hashes', () => {
    for (const match of read('index.html').matchAll(/(?:src|href)="(\.\/[^"{}]+)"/g)) {
        assert.ok(fs.existsSync(path.join(viewPath, match[1])), match[1])
    }
    assert.doesNotMatch(read('index.html'), /(?:src|href)="https?:/)
    assert.doesNotMatch(read('js/modules/repository.js'), /import .+ from "https?:/)
    for (const entry of JSON.parse(read('vendor/manifest.json'))) {
        const bytes = fs.readFileSync(path.join(viewPath, 'vendor', entry.path))
        assert.equal(bytes.length, entry.bytes, entry.path)
        assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), entry.sha256, entry.path)
    }
    for (const match of read('vendor/mdi/css/materialdesignicons.min.css').matchAll(/url\(["']?([^\)"']+)/g)) {
        const file = match[1].split('?')[0].split('#')[0]
        assert.ok(fs.existsSync(path.join(viewPath, 'vendor/mdi/css', file)), file)
    }
})
