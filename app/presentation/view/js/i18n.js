// UI strings live in lang/*.json. Adding a language requires a dictionary and
// an entry in SUPPORTED_LOCALES; no HTML or JavaScript strings need translating.
const SUPPORTED_LOCALES = ['ja', 'en']
const dictionaries = {}
const storageKey = 'myImageSearch.locale'

function getInitialLocale() {
    try {
        const saved = localStorage.getItem(storageKey)
        if (SUPPORTED_LOCALES.includes(saved)) return saved
    } catch (_) { /* The interface still works when storage is unavailable. */ }
    return 'ja'
}

function translate(locale, key, params = {}) {
    const message = dictionaries[locale]?.[key] ?? dictionaries.ja?.[key]
    if (typeof message !== 'string') return key
    return message.replace(/\{([\w]+)\}/g, (match, name) =>
        Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : match)
}

function setLocale(locale) {
    const supported = SUPPORTED_LOCALES.includes(locale) ? locale : 'ja'
    document.documentElement.lang = supported
    document.title = translate(supported, 'app.title')
    try { localStorage.setItem(storageKey, supported) } catch (_) {}
}

export const i18n = {
    t: translate,
    getInitialLocale,
    setLocale,
    vuetifyMessages: {},
    ready: Promise.all(SUPPORTED_LOCALES.map(async locale => {
        const response = await fetch(new URL(`../lang/${locale}.json`, import.meta.url))
        if (!response.ok) throw new Error(`Language file unavailable: ${locale}`)
        dictionaries[locale] = await response.json()
    })),
}

// Delay importing application modules until all language strings are available.
// This also gives Vuetify complete translated messages before it is created.
await i18n.ready
for (const locale of SUPPORTED_LOCALES) i18n.vuetifyMessages[locale] = dictionaries[locale].vuetify
setLocale(getInitialLocale())
