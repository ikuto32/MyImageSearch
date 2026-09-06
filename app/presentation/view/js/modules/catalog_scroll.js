// Only compressed catalogs need custom touch scrolling. Ordinary lists keep
// browser scrolling; trackpads keep the momentum already present in wheel events.
export function createCatalogScroll({ scrollBy, enabled, reducedMotion = () => false,
    now = () => performance.now(), requestFrame = callback => requestAnimationFrame(callback),
    cancelFrame = id => cancelAnimationFrame(id) }) {
    const points = new Map()
    let anchor = null, last = null, samples = [], distance = 0, dragging = false
    let fingerCountChanged = false, twoFingerStart = null
    let frame = null, velocity = 0, frameTime = 0, suppressClick = false

    function center() {
        return [...points.values()].reduce((sum, point) => sum + point.y, 0) / points.size
    }

    function inputTime(event) {
        const current = now()
        // Use input time, not dispatch time: rendering may queue several moves
        // and then deliver them together. Ignore legacy epoch-based timestamps.
        return Number.isFinite(event.timeStamp) && event.timeStamp >= 0 && event.timeStamp <= current
            ? event.timeStamp : current
    }

    function stopMomentum() {
        if (frame !== null) cancelFrame(frame)
        frame = null
        velocity = 0
    }

    function resetAnchor(time) {
        anchor = last = points.size ? center() : null
        distance = 0
        samples = [{ time, distance }]
    }

    function release(point, id) {
        if (point.target?.hasPointerCapture?.(id)) point.target.releasePointerCapture(id)
    }

    function stop() {
        stopMomentum()
        for (const [id, point] of points) release(point, id)
        points.clear()
        anchor = last = null
        samples = []
        dragging = false
        fingerCountChanged = false
        twoFingerStart = null
    }

    function animate(time) {
        frame = null
        const elapsed = time - frameTime
        // A suspended tab must not jump when animation callbacks resume.
        if (!enabled() || reducedMotion() || elapsed > 500) return stopMomentum()
        // RAF timestamps mark the frame's start and may precede pointerup's
        // performance.now() in the same frame. Start moving on the next frame.
        if (elapsed <= 0) {
            frame = requestFrame(animate)
            return
        }
        frameTime = time
        const decay = Math.exp(-elapsed / 240)
        const delta = velocity * 240 * (1 - decay)
        velocity *= decay
        const moved = scrollBy(delta)
        if (Math.abs(velocity) < 0.02 || Math.abs(moved - delta) > 0.1) return stopMomentum()
        frame = requestFrame(animate)
    }

    return {
        stop,
        keyDown() {
            stop()
            suppressClick = false
        },
        pointerDown(event) {
            stopMomentum()
            if (!points.size) suppressClick = false
            if (!['touch', 'pen'].includes(event.pointerType) || !enabled()) return stop()
            points.set(event.pointerId, { y: event.clientY, target: event.currentTarget })
            // A change in finger count must not become movement or release velocity.
            resetAnchor(inputTime(event))
            fingerCountChanged = false
        },
        pointerMove(event) {
            if (!points.has(event.pointerId)) return
            if (!enabled()) return stop()
            const time = inputTime(event)
            if (fingerCountChanged) {
                resetAnchor(time)
                fingerCountChanged = false
            }
            const point = points.get(event.pointerId)
            point.y = event.clientY
            const current = center()
            if (points.size > 2) return stop()
            if (!dragging && Math.abs(current - anchor) < 6) return
            dragging = true
            suppressClick = true
            for (const [id, touch] of points) {
                if (!touch.target?.hasPointerCapture?.(id)) touch.target?.setPointerCapture?.(id)
            }
            const delta = last - current
            last = current
            const moved = scrollBy(delta)
            // Pressure-only pointer events must not hide a direction reversal
            // or count as fresh movement after the finger has stopped.
            if (moved === 0) return
            const previous = samples.at(-1)
            const previousDelta = samples.length > 1 ? previous.distance - samples.at(-2).distance : 0
            if (moved * previousDelta < 0) samples = [previous]
            distance += moved
            samples.push({ time, distance })
            while (samples.length > 1 && samples[0].time < time - 100) samples.shift()
        },
        pointerUp(event) {
            const point = points.get(event.pointerId)
            if (!point) return
            release(point, event.pointerId)
            points.delete(event.pointerId)
            // A single touchend can produce multiple pointerup events. Preserve
            // velocity unless the remaining finger actually starts another pan.
            if (points.size) {
                fingerCountChanged = true
                return
            }
            const time = inputTime(event)
            const first = samples[0], latest = samples.at(-1)
            const canFling = dragging && enabled() && !reducedMotion() && latest &&
                time - latest.time < 80 && latest.time > first.time
            velocity = canFling ? Math.max(-8, Math.min(8, (latest.distance - first.distance) / (latest.time - first.time))) : 0
            dragging = false
            anchor = last = null
            samples = []
            if (Math.abs(velocity) >= 0.02) {
                frameTime = now()
                frame = requestFrame(animate)
            }
        },
        pointerCancel() {
            // Safari can cancel before the first move, then emit a click when
            // the finger eventually lifts. Keep suppression until a new input.
            if (points.size) suppressClick = true
            stop()
        },
        touchStart(event) {
            const touches = [...event.touches]
            twoFingerStart = touches.length === 2 ? {
                y: (touches[0].clientY + touches[1].clientY) / 2,
                span: Math.hypot(touches[1].clientX - touches[0].clientX, touches[1].clientY - touches[0].clientY),
            } : null
        },
        touchMove(event) {
            if (!enabled() || event.cancelable === false) return
            // Suppress the compatibility click at its source once a pan starts.
            // A stationary touch remains a normal tap.
            if (event.touches.length === 1 && dragging) {
                event.preventDefault()
                return
            }
            if (!twoFingerStart || event.touches.length !== 2) return
            const [first, second] = event.touches
            const span = Math.hypot(second.clientX - first.clientX, second.clientY - first.clientY)
            const spanDelta = Math.abs(span - twoFingerStart.span)
            const pan = Math.abs((first.clientY + second.clientY) / 2 - twoFingerStart.y)
            // Compare atomic positions and require a clear change of scale, not
            // ordinary finger spacing drift. Rebase during a pan so a subsequent
            // deliberate pinch remains available without lifting both fingers.
            const pinchThreshold = Math.max(24, Math.min(48, twoFingerStart.span * 0.12))
            if (spanDelta > pinchThreshold && spanDelta > pan * 1.6) {
                twoFingerStart = null
                this.pointerCancel()
                return
            }
            // Cancel even undecided moves: otherwise the browser can claim the
            // gesture before a slow or diagonal two-finger pan is recognized.
            event.preventDefault()
            if (pan >= 6 && pan > spanDelta * 0.75) twoFingerStart = {
                y: (first.clientY + second.clientY) / 2, span,
            }
        },
        click(event) {
            // Touch-generated clicks can have detail=0 too. Actual keyboard
            // input and the next pointerdown explicitly restore activation.
            if (suppressClick) {
                event.preventDefault()
                event.stopImmediatePropagation()
            }
        },
    }
}
