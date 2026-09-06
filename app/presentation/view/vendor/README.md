# Bundled browser dependencies

The interface serves its JavaScript, styles and icon fonts from this directory.
No CDN connection is required at runtime. These are the same dependency series
used by the application before bundling, resolved to the fixed versions below.

| Package | Version | Source | License |
| --- | --- | --- | --- |
| Vue | 3.4.38 | https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.global.prod.js | MIT; `vue/LICENSE` |
| Vuetify | 3.5.18 | https://cdn.jsdelivr.net/npm/vuetify@3.5.18/dist/vuetify.min.js and `vuetify.min.css` | MIT; `vuetify/LICENSE.md` |
| Material Design Icons font | 6.9.96 | https://cdn.jsdelivr.net/npm/@mdi/font@6.9.96/css/materialdesignicons.min.css and package `fonts/` | Apache 2.0 fonts/icons, MIT code; `mdi/LICENSE`, `mdi/LICENSE-APACHE-2.0.txt` |
| Axios | 1.3.1 | https://cdn.jsdelivr.net/npm/axios@1.3.1/dist/esm/axios.min.js and its source map | MIT; `axios/LICENSE` |

Retrieved on 2026-09-06. `manifest.json` records each distributed file's size and
SHA-256 hash. Upgrades should replace the files and licenses together, regenerate
the manifest, and verify the interface with outbound network access unavailable.

The Vue production build avoids development tooling and development-only console
messages. Axios uses its bundled browser ES module, with no transitive CDN imports.
