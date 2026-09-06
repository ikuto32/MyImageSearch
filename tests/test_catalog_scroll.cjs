// Run with: node --test tests/test_catalog_scroll.cjs
// Exercise the production input module with a clock; native gesture arbitration
// and browser scrolling remain covered by scripts/review_scroll.cjs.
const assert = require('node:assert/strict')
const path = require('node:path')
const { pathToFileURL } = require('node:url')
const test = require('node:test')

let createCatalogScroll
test.before(async () => {
    const url = pathToFileURL(path.join(__dirname, '../app/presentation/view/js/modules/catalog_scroll.js'))
    ;({ createCatalogScroll } = await import(url.href))
})

function setup() {
    let time = 0, nextFrame = 0
    const frames = new Map(), fingers = new Map(), captures = new Set()
    const state = { position: 1000, maximum: Infinity, enabled: true, reducedMotion: false }
    const target = {
        hasPointerCapture: id => captures.has(id),
        setPointerCapture: id => captures.add(id),
        releasePointerCapture: id => captures.delete(id),
    }
    const input = createCatalogScroll({
        now: () => time,
        enabled: () => state.enabled,
        reducedMotion: () => state.reducedMotion,
        requestFrame(callback) { frames.set(++nextFrame, callback); return nextFrame },
        cancelFrame: id => frames.delete(id),
        scrollBy(delta) {
            const before = state.position
            state.position = Math.max(0, Math.min(state.maximum, before + delta))
            return state.position - before
        },
    })
    const pointer = id => ({ pointerId: id, pointerType: 'touch', currentTarget: target, ...fingers.get(id) })
    const touches = () => [...fingers.values()]
    return {
        input, state,
        elapse(ms) { time += ms },
        frame(ms = 16) {
            time += ms
            const callbacks = [...frames.values()]
            frames.clear()
            callbacks.forEach(callback => callback(time))
        },
        down(id, y = 400, pointerType = 'touch') {
            fingers.set(id, { clientX: 100, clientY: y, pointerType })
            input.pointerDown(pointer(id))
            input.touchStart({ touches: touches() })
        },
        move(ids, pixels, ms = 16) {
            time += ms
            for (const id of ids) {
                fingers.get(id).clientY -= pixels
                input.pointerMove(pointer(id))
            }
            input.touchMove({ touches: touches(), preventDefault() {} })
        },
        up(id) { input.pointerUp(pointer(id)); fingers.delete(id) },
        get pendingFrames() { return frames.size },
    }
}

function flick(harness, count = 1, hold = 0) {
    const ids = Array.from({ length: count }, (_, i) => i + 1)
    ids.forEach(id => harness.down(id, 400 + id * 100))
    harness.elapse(hold)
    for (let step = 0; step < 3; step++) harness.move(ids, 24)
    ids.forEach(id => harness.up(id))
}

for (const fingers of [1, 2]) {
    test(`${fingers} finger pan follows distance and coasts with decreasing movement`, () => {
        const h = setup()
        flick(h, fingers)
        assert.equal(h.state.position, 1072, 'finger count must not multiply pan distance')
        h.frame()
        const first = h.state.position - 1072
        assert(first > 0, 'release must continue the flick')
        h.frame()
        const second = h.state.position - 1072 - first
        assert(second > 0 && second < first, 'momentum should decay')
        for (let step = 0; step < 150; step++) h.frame()
        assert.equal(h.pendingFrames, 0, 'the coast must eventually stop')
        const stopped = h.state.position
        h.frame()
        assert.equal(h.state.position, stopped)
    })
}

test('a first animation timestamp before pointer release does not discard the fling', () => {
    const h = setup()
    flick(h, 2)
    const released = h.state.position
    h.frame(-2)
    assert.equal(h.state.position, released)
    assert.equal(h.pendingFrames, 1)
    h.frame(16)
    assert(h.state.position > released)
})

test('reversing after a stationary pointer event coasts in the new direction', () => {
    const h = setup()
    h.down(1)
    for (let i = 0; i < 3; i++) h.move([1], 24)
    h.move([1], 0)
    h.move([1], -20)
    h.up(1)
    const released = h.state.position
    h.frame()
    assert(h.state.position < released, 'momentum must follow the last direction')
})

test('holding before a flick does not weaken it; holding after movement prevents a fling', () => {
    const immediate = setup(), held = setup()
    flick(immediate)
    flick(held, 1, 600)
    immediate.frame()
    held.frame()
    assert(Math.abs(immediate.state.position - held.state.position) < 0.1)

    const paused = setup()
    paused.down(1)
    paused.move([1], 48)
    paused.elapse(200)
    paused.up(1)
    paused.frame()
    assert.equal(paused.state.position, 1048, 'a stationary release must not fling')
})

test('adding and removing a finger preserves distance without a position jump', () => {
    const h = setup()
    h.down(1)
    h.move([1], 24)
    h.down(2, 600)
    assert.equal(h.state.position, 1024)
    h.move([1, 2], 24)
    assert.equal(h.state.position, 1048)
    h.up(2)
    assert.equal(h.state.position, 1048)
    h.move([1], 24)
    assert.equal(h.state.position, 1072)
    h.up(1)
})

const interruptions = {
    cancel: h => h.input.pointerCancel(),
    'new touch': h => h.down(3),
    stop: h => h.input.stop(),
    disabled: h => { h.state.enabled = false },
    'reduced motion': h => { h.state.reducedMotion = true },
    'background pause': h => h.frame(2000),
}
for (const [name, interrupt] of Object.entries(interruptions)) {
    test(`${name} stops an active coast without later movement`, () => {
        const h = setup()
        flick(h)
        h.frame()
        const before = h.state.position
        interrupt(h)
        for (let step = 0; step < 5; step++) h.frame()
        assert.equal(h.state.position, before)
        assert.equal(h.pendingFrames, 0)
    })
}

test('reduced motion keeps direct panning; ordinary lists receive no custom movement', () => {
    const reduced = setup()
    reduced.state.reducedMotion = true
    flick(reduced)
    reduced.frame()
    assert.equal(reduced.state.position, 1072)
    const ordinary = setup()
    ordinary.state.enabled = false
    flick(ordinary)
    ordinary.frame()
    assert.equal(ordinary.state.position, 1000)
})

test('momentum reaches a boundary and stops without overshooting', () => {
    const h = setup()
    flick(h)
    h.state.maximum = h.state.position + 5
    h.frame()
    assert.equal(h.state.position, h.state.maximum)
    assert.equal(h.pendingFrames, 0)
    h.frame()
    assert.equal(h.state.position, h.state.maximum)
})

test('a slow frame continues the coast while a suspended-tab frame does not jump', () => {
    const h = setup()
    flick(h)
    h.frame(150)
    assert(h.state.position > 1072)
    assert(h.pendingFrames > 0)
    const before = h.state.position
    h.frame(2000)
    assert.equal(h.state.position, before)
})

test('parallel touch movement is handled, and a later pinch relinquishes the gesture', () => {
    const h = setup()
    h.down(1, 400)
    h.down(2, 500)
    let prevented = 0
    const touchEvent = (first, second) => ({
        touches: [{ clientX: 100, clientY: first }, { clientX: 100, clientY: second }],
        preventDefault() { prevented++ },
    })
    h.input.touchMove(touchEvent(380, 480))
    assert.equal(prevented, 1)
    h.move([1, 2], 24)
    h.input.touchMove(touchEvent(360, 500))
    assert.equal(prevented, 1, 'pinch must remain available to the browser')
    const before = h.state.position
    h.up(1)
    h.up(2)
    h.frame()
    assert.equal(h.state.position, before, 'pinch must not produce a fling')
})

test('a drag suppresses delayed and detail-zero clicks; new keyboard input or a tap still work', () => {
    const h = setup()
    let prevented = 0
    const click = detail => h.input.click({
        detail, preventDefault() { prevented++ }, stopImmediatePropagation() {},
    })
    flick(h)
    h.elapse(2000)
    click(1)
    assert.equal(prevented, 1)
    click(0)
    assert.equal(prevented, 2)
    h.input.keyDown()
    click(0)
    assert.equal(prevented, 2)
    flick(h)
    h.down(1)
    h.up(1)
    click(1)
    assert.equal(prevented, 2)
})

test('a browser cancellation before movement cannot activate an image on release', () => {
    const h = setup()
    h.down(1)
    h.input.pointerCancel()
    h.elapse(2000)
    let prevented = false
    h.input.click({ detail: 0, preventDefault() { prevented = true }, stopImmediatePropagation() {} })
    assert(prevented)
    assert.equal(h.state.position, 1000)
})

test('only a moving touch cancels compatibility clicks at their source', () => {
    const h = setup()
    let prevented = 0
    const event = { touches: [{}], cancelable: true, preventDefault() { prevented++ } }
    h.down(1)
    h.input.touchMove(event)
    assert.equal(prevented, 0, 'a tap keeps native activation')
    h.move([1], 20)
    h.input.touchMove(event)
    assert.equal(prevented, 1)
    h.input.touchMove({ ...event, cancelable: false })
    h.state.enabled = false
    h.input.touchMove(event)
    assert.equal(prevented, 1, 'uncancelable events and ordinary lists stay native')
})

test('a pen follows direct movement instead of becoming a selection gesture', () => {
    const h = setup()
    h.down(1, 400, 'pen')
    h.move([1], 24)
    h.move([1], 24)
    h.up(1)
    assert.equal(h.state.position, 1048)
    h.frame()
    assert(h.state.position > 1048)
})

test('two-finger spacing drift keeps panning and slow pinching still yields to zoom', () => {
    const h = setup()
    h.down(1, 400)
    h.down(2, 500)
    for (let step = 0; step < 12; step++) {
        // The second finger alternates slightly faster/slower than the first.
        h.move([1], 20)
        h.move([2], step % 2 ? 8 : 32, 0)
    }
    assert.equal(h.state.position, 1240)
    h.up(1)
    h.up(2)
    h.frame()
    assert(h.state.position > 1240)

    const pinch = setup()
    pinch.down(1, 400)
    pinch.down(2, 500)
    let yielded = false
    for (let spread = 2; spread <= 40; spread += 2) {
        let prevented = false
        pinch.input.touchMove({ touches: [{ clientX: 100, clientY: 400 - spread },
            { clientX: 100, clientY: 500 + spread }], preventDefault() { prevented = true } })
        if (!prevented) { yielded = true; break }
    }
    assert(yielded, 'small successive scale changes must accumulate into a pinch')
})

test('queued input uses the gesture duration rather than the short dispatch duration', () => {
    const h = setup()
    h.input.pointerDown({ pointerId: 1, pointerType: 'touch', clientY: 400, timeStamp: 0 })
    h.elapse(200)
    for (let step = 1; step <= 3; step++) {
        h.elapse(1)
        h.input.pointerMove({ pointerId: 1, clientY: 400 - step * 20, timeStamp: step * 20 })
    }
    h.input.pointerUp({ pointerId: 1, timeStamp: 60 })
    assert.equal(h.state.position, 1060)
    h.frame()
    const coast = h.state.position - 1060
    assert(coast > 10 && coast < 20, `queued moves produced an accelerated coast: ${coast}`)
})

test('a fast flick can exceed the former speed limit while extreme speeds remain bounded', () => {
    for (const direction of [1, -1]) {
        const h = setup()
        h.state.position = 10000
        h.down(1)
        for (let step = 0; step < 3; step++) h.move([1], direction * 320)
        h.up(1)
        const released = h.state.position
        h.frame()
        const moved = (h.state.position - released) * direction
        assert(moved > 100 && moved <= 128, `fling speed must be raised but capped: ${moved}`)
        for (let step = 0; step < 150; step++) h.frame()
        assert.equal(h.pendingFrames, 0)
        assert((h.state.position - released) * direction < 2000, 'extreme input cannot cause an unbounded jump')
    }
})
