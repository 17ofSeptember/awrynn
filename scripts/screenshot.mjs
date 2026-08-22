/*
 * Visual verification driver.
 *
 * The canvas IS the product (§6), so "the tests pass" is not evidence that it
 * renders. This drives a real Chrome over the DevTools protocol, dispatches
 * real input, and reports whether the canvas actually has pixels on it.
 *
 * Chrome's own --screenshot flag is not usable here: the app runs a permanent
 * requestAnimationFrame loop, so the page never reaches the "settled" state
 * that flag waits for, and it hangs. CDP sidesteps that entirely.
 *
 * No dependencies — Node 22 ships a global WebSocket and fetch.
 *
 *   npm run build && npx vite preview --port 5178 &
 *   node scripts/screenshot.mjs <url> <out.png> [width] [height] [actions...]
 *
 * Actions are evaluated in order before the shot:
 *   MOVE:x,y        dispatch a real mouse move (CSS pixels)
 *   CLICK:x,y       move, press and release
 *   <expression>    evaluated in the page; its value is printed
 *
 * Requires Chrome listening on 9222:
 *   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
 *     --headless=new --disable-gpu --remote-debugging-port=9222 \
 *     --user-data-dir=/tmp/awry-cp about:blank &
 */

import { writeFileSync } from 'node:fs';

const [,, url, outPath, widthArg, heightArg, ...actions] = process.argv;
const width = Number(widthArg ?? 1600), height = Number(heightArg ?? 1000);

async function cdpTargets() {
  const res = await fetch('http://127.0.0.1:9222/json');
  return res.json();
}

const targets = await cdpTargets();
const page = targets.find(t => t.type === 'page');
if (!page) { console.error('no page target'); process.exit(1); }

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const events = [];

ws.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  else if (msg.method) events.push(msg);
});
await new Promise(r => ws.addEventListener('open', r, { once: true }));

function send(method, params = {}) {
  const msgId = ++id;
  return new Promise((resolve, reject) => {
    pending.set(msgId, (m) => m.error ? reject(new Error(method + ': ' + JSON.stringify(m.error))) : resolve(m.result));
    ws.send(JSON.stringify({ id: msgId, method, params }));
  });
}
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

await send('Page.enable');
await send('Runtime.enable');
await send('Log.enable');
await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 2, mobile: false });
await send('Page.navigate', { url });
await sleep(2500);

// Run any requested interactions before the shot.
// MOVE:x,y and CLICK:x,y dispatch real input through CDP rather than synthetic
// DOM events, so pointer capture and the drag/click distinction are exercised.
let clip = null;
for (const action of actions) {
  if (action.startsWith('CLIP:')) {
    // CLIP:x,y,w,h — capture only this region, for inspecting a detail at
    // full resolution instead of squinting at a full-page shot.
    const [x, y, width, height] = action.slice(5).split(',').map(Number);
    clip = { x, y, width, height, scale: 2 };
    continue;
  }
  if (action.startsWith('SELECTOR_CLIP:')) {
    // Capture the bounding box of a selector, resolved in the page.
    const selector = action.slice(14);
    const { result } = await send('Runtime.evaluate', {
      expression: `(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el) return null; const r = el.getBoundingClientRect(); return JSON.stringify({x:r.x,y:r.y,width:r.width,height:r.height}); })()`,
      returnByValue: true,
    });
    if (result.value) { clip = { ...JSON.parse(result.value), scale: 2 }; console.log('CLIP:', result.value); }
    else console.error('CLIP: selector not found', selector);
    continue;
  }
  if (action.startsWith('DRAG:')) {
    // DRAG:x1,y1,x2,y2 — press, move in steps, release. Steps matter: a single
    // jump would not exercise the per-move scrub path.
    const [x1, y1, x2, y2] = action.slice(5).split(',').map(Number);
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x1, y: y1, button: 'none', buttons: 0, pointerType: 'mouse' });
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: x1, y: y1, button: 'left', buttons: 1, clickCount: 1, pointerType: 'mouse' });
    const steps = 12;
    for (let i = 1; i <= steps; i++) {
      const x = x1 + ((x2 - x1) * i) / steps;
      const y = y1 + ((y2 - y1) * i) / steps;
      await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, button: 'left', buttons: 1, pointerType: 'mouse' });
      await sleep(30);
    }
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: x2, y: y2, button: 'left', buttons: 0, clickCount: 1, pointerType: 'mouse' });
    await sleep(400);
    continue;
  }
  if (action.startsWith('MOVE:') || action.startsWith('CLICK:')) {
    const [kind, coords] = action.split(':');
    const [x, y] = coords.split(',').map(Number);
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, button: 'none', buttons: 0, pointerType: 'mouse' });
    if (kind === 'CLICK') {
      await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', buttons: 1, clickCount: 1, pointerType: 'mouse' });
      await sleep(60);
      await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', buttons: 0, clickCount: 1, pointerType: 'mouse' });
    }
    await sleep(500);
    continue;
  }
  const { result } = await send('Runtime.evaluate', { expression: action, awaitPromise: true, returnByValue: true });
  if (result && result.subtype === 'error') console.error('ACTION ERROR:', result.description);
  else console.log('EVAL:', typeof result.value === 'string' ? result.value : JSON.stringify(result.value));
  await sleep(600);
}
await sleep(600);

// Surface anything the page complained about — a blank canvas usually has a reason.
const consoleErrors = events
  .filter(e => e.method === 'Log.entryAdded' && e.params.entry.level === 'error')
  .map(e => e.params.entry.text);
if (consoleErrors.length) console.error('CONSOLE ERRORS:\n  ' + consoleErrors.join('\n  '));

// Report whether the canvas actually has non-background pixels drawn on it.
const probe = await send('Runtime.evaluate', {
  expression: `(() => {
    const c = document.querySelector('canvas');
    if (!c) return JSON.stringify({ canvas: false });
    const ctx = c.getContext('2d');
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    const seen = new Map();
    for (let i = 0; i < d.length; i += 4 * 97) {
      const k = d[i] + ',' + d[i+1] + ',' + d[i+2];
      seen.set(k, (seen.get(k) || 0) + 1);
    }
    const top = [...seen.entries()].sort((a,b) => b[1]-a[1]).slice(0, 6);
    return JSON.stringify({ canvas: true, w: c.width, h: c.height, distinctColors: seen.size, top });
  })()`,
  returnByValue: true,
});
console.log('CANVAS PROBE:', probe.result.value);

const shot = await send('Page.captureScreenshot', clip === null ? { format: 'png' } : { format: 'png', clip });
writeFileSync(outPath, Buffer.from(shot.data, 'base64'));
console.log('saved', outPath);
ws.close();
process.exit(0);
