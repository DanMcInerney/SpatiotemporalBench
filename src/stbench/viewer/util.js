'use strict';
// Helpers for the page: DOM building and the formatting of numbers and answers. Everything from the run is untrusted
// text, so this page makes elements with createElement and puts text in with textContent or text nodes. It never makes
// markup from a string; tests/test_view_page.py checks that, and runs the page against a DOM that refuses markup.

function add(el, kid) {
  if (kid === null || kid === undefined || kid === false) return;
  if (Array.isArray(kid)) kid.forEach(k => add(el, k));
  else el.appendChild(kid instanceof Node ? kid : document.createTextNode(String(kid)));
}

// Replace what is inside EL with the kids (a null among them is nothing).
function fill(el, ...kids) {
  el.replaceChildren();
  kids.forEach(k => add(el, k));
}

// h('div', {class: 'x', title: 't', on: {click: fn}}, child, 'text', ...). A link or picture takes only file: addresses,
// which the reader wrote for the dataset's frames.
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const key of Object.keys(props || {})) {
    const v = props[key];
    if (v === null || v === undefined || v === false) continue;
    if (key === 'class') el.className = v;
    else if (key === 'on') for (const ev of Object.keys(v)) el.addEventListener(ev, v[ev]);
    else if (key === 'href' || key === 'src') { if (/^file:/i.test(v)) el.setAttribute(key, v); }
    else el.setAttribute(key, v === true ? '' : String(v));
  }
  kids.forEach(k => add(el, k));
  return el;
}

const DASH = '–', MINUS = '−', ELLIPSIS = '…', ARROW = ' → ';

function fmtInt(n) { return (typeof n === 'number' && isFinite(n)) ? n.toLocaleString('en-US') : DASH; }

function num(x, digits) {
  if (typeof x !== 'number' || !isFinite(x)) return String(x);
  const t = Number(x.toFixed(digits === undefined ? 3 : digits));
  return Object.is(t, -0) ? '0' : String(t);
}

function signed(x, digits) {
  if (typeof x !== 'number' || !isFinite(x)) return DASH;
  const t = x.toFixed(digits === undefined ? 2 : digits);
  return (x > 0 ? '+' : '') + t.replace('-', MINUS);
}

function fmtTokens(n) {
  if (typeof n !== 'number') return DASH;
  if (n >= 1e6) return num(n / 1e6, 2) + 'M';
  if (n >= 1e4) return num(n / 1e3, 1) + 'k';
  return fmtInt(n);
}

function fmtCost(x) { return typeof x === 'number' ? '$' + x.toFixed(x < 1 ? 4 : 2) : DASH; }

function fmtSeconds(s) {
  if (typeof s !== 'number') return DASH;
  if (s < 120) return num(s, 1) + ' s';
  const m = Math.floor(s / 60);
  return m + ' min ' + Math.round(s - 60 * m) + ' s';
}

function json(v) { try { return JSON.stringify(v); } catch (e) { return String(v); } }

// Text that is cut at `limit` characters with a button to show all of it.
function clipped(text, limit) {
  text = String(text);
  if (text.length <= limit) return h('span', {class: 'txt'}, text);
  const cut = text.slice(0, limit) + ELLIPSIS;
  const body = h('span', {class: 'txt'}, cut);
  const label = 'show all (' + fmtInt(text.length) + ' characters)';
  const btn = h('button', {class: 'link', type: 'button'}, label);
  let open = false;
  btn.addEventListener('click', ev => {
    ev.stopPropagation();
    open = !open;
    body.textContent = open ? text : cut;
    btn.textContent = open ? 'show less' : label;
  });
  return h('span', {class: 'clip'}, body, ' ', btn);
}

// ---- answers --------------------------------------------------------------------------------------------------
const SHOWN_ITEMS = 10;

function point(p) { return Array.isArray(p) ? '(' + p.map(v => num(v, 2)).join(', ') + ')' : String(p); }
function place(item) { return point(item && (item.xy || item.xyz)); }

function listed(items, show, sep) {
  const shown = items.slice(0, SHOWN_ITEMS).map(show).join(sep);
  return items.length > SHOWN_ITEMS ? shown + sep + ELLIPSIS + ' +' + (items.length - SHOWN_ITEMS) + ' more (' + items.length + ' in all)' : shown;
}

// An answer or a gold in words: a position as (x, y), an offset from the camera, yes or no, a clock, unknown, a list of
// places, a route.
function formatAnswer(a) {
  if (a === null || a === undefined) return 'no answer';
  if (typeof a !== 'object') return json(a);
  const items = Array.isArray(a.items) ? a.items : null;
  let text;
  switch (a.type) {
    case 'unknown': return 'unknown';
    case 'failed': return 'failed: ' + a.reason;
    case 'position': text = point(a.xy || a.xyz); break;
    case 'offset': text = num(a.ahead_m, 2) + ' m ahead, ' + num(a.left_m, 2) + ' m left'; break;
    case 'time': text = String(a.clock); break;
    case 'yesno': text = a.value === true ? 'yes' : a.value === false ? 'no' : String(a.value); break;
    case 'count': case 'number': text = num(a.value); break;
    case 'set': text = !items ? json(a) : items.length ? listed(items, i => (i.change || '?') + ' ' + place(i), '; ') : 'none'; break;
    case 'positions': text = !items ? json(a) : items.length ? listed(items, place, ' ') : 'none'; break;
    case 'cells': text = !items ? json(a) : items.length ? items.length + ' cells: ' + listed(items, c => '[' + c + ']', ' ') : 'none'; break;
    case 'route':
      if (Array.isArray(a.points)) text = listed(a.points, point, ARROW);
      else if (Array.isArray(a.valid)) text = 'one of ' + a.valid.length + ' region sequences: ' + listed(a.valid, v => [].concat(v).join(ARROW), ' | ');
      else text = json(a);
      break;
    default: return json(a);
  }
  const dont = Array.isArray(a.dont_care) ? a.dont_care.length : 0;
  return dont ? text + " (+" + dont + " don't-care)" : text;
}

// How far an answer is from the gold, where that has a meaning: metres, a count or a number, seconds.
function offBy(q) {
  const a = q.answer, g = q.gold;
  if (!a || !g || typeof a !== 'object' || typeof g !== 'object' || a.type !== g.type) return '';
  if (a.type === 'position') {
    const p = a.xy || a.xyz, r = g.xy || g.xyz;
    if (!Array.isArray(p) || !Array.isArray(r) || p.length < 2 || r.length < 2) return '';
    const flat = q.metric !== 'xyz' || p.length < 3 || r.length < 3;
    const d = flat ? Math.hypot(p[0] - r[0], p[1] - r[1]) : Math.hypot(p[0] - r[0], p[1] - r[1], p[2] - r[2]);
    return 'off by ' + num(d, 2) + ' m';
  }
  if (a.type === 'offset' && [a.ahead_m, a.left_m, g.ahead_m, g.left_m].every(v => typeof v === 'number')) {
    const d = Math.hypot(a.ahead_m - g.ahead_m, a.left_m - g.left_m);
    return 'off by ' + num(d, 2) + ' m' + (q.range_m > 0 ? ' (' + num(d / q.range_m, 2) + ' of the range)' : '');
  }
  if ((a.type === 'number' || a.type === 'count') && typeof a.value === 'number' && typeof g.value === 'number') {
    return 'off by ' + signed(a.value - g.value, 2);
  }
  if (a.type === 'time') {
    const d = (Date.parse(a.clock) - Date.parse(g.clock)) / 1000;
    return isFinite(d) ? 'off by ' + signed(d, 0) + ' s' : '';
  }
  return '';
}

// How the gold is scored: the credit and its tolerance.
function creditText(q) {
  if (!q.credit) return '';
  const t = q.tolerances || {};
  const unit = q.answer_type === 'time' ? ' s' : (q.answer_type === 'position' || q.answer_type === 'number' || q.answer_type === 'offset') ? ' m' : '';
  let s = 'credit: ' + q.credit;
  if (Array.isArray(t.ladder)) s += ' ' + t.ladder.map(v => num(v)).join(' / ') + unit;
  else if (typeof t.erode_m === 'number') s += ' (erode ' + num(t.erode_m) + ' m)';
  else if (typeof t.match_m === 'number') s += ' (match ' + num(t.match_m) + ' m)';
  if (q.metric) s += ', ' + q.metric;
  if (typeof q.range_m === 'number') s += ', range ' + num(q.range_m, 2) + ' m';
  if (Array.isArray(q.region)) s += ', region of ' + q.region.length + ' points';
  return s;
}
