// Runs the script of a page that `stbench view` wrote against a small stand-in for the browser's DOM, and prints what the
// page came to hold as JSON. node tests/view_dom_stub.js PAGE.html. It is how tests/test_view_page.py proves that the
// page's code runs from end to end on a run (expands every case and its log, filters, sorts) and that data only ever
// becomes text: the stand-in's innerHTML, outerHTML and insertAdjacentHTML throw, so a use of one fails the run.
'use strict';
const fs = require('fs');
const vm = require('vm');

class Node {}
class Text extends Node {
  constructor(data) { super(); this.nodeType = 3; this.data = String(data); }
  get textContent() { return this.data; }
}
const forbidden = name => () => { throw new Error(name + ' is not for data'); };
class Elem extends Node {
  constructor(tag) {
    super();
    Object.assign(this, {nodeType: 1, tagName: String(tag).toLowerCase(), attrs: {}, childNodes: [], listeners: {},
      parentNode: null, hidden: false, open: false, checked: false, value: '', dataset: {}, className: '', id: ''});
  }
  get firstChild() { return this.childNodes[0] || null; }
  appendChild(kid) {
    if (kid.parentNode) kid.parentNode.removeChild(kid);
    kid.parentNode = this; this.childNodes.push(kid); return kid;
  }
  removeChild(kid) {
    const i = this.childNodes.indexOf(kid);
    if (i >= 0) this.childNodes.splice(i, 1);
    kid.parentNode = null; return kid;
  }
  replaceChildren(...kids) {
    this.childNodes.slice().forEach(k => this.removeChild(k));
    kids.forEach(k => this.appendChild(k instanceof Node ? k : new Text(k)));
  }
  setAttribute(key, value) {
    this.attrs[key] = String(value);
    if (key === 'id') this.id = String(value);
    if (key === 'class') this.className = String(value);
    if (key === 'hidden') this.hidden = true;
    if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-(.)/g, (m, c) => c.toUpperCase())] = String(value);
  }
  removeAttribute(key) { delete this.attrs[key]; }
  addEventListener(name, fn) { (this.listeners[name] = this.listeners[name] || []).push(fn); }
  fire(name, extra) { (this.listeners[name] || []).forEach(fn => fn({stopPropagation() {}, preventDefault() {}, key: 'Enter', ...extra})); }
  get textContent() { return this.childNodes.map(k => k.textContent).join(''); }
  set textContent(text) { this.childNodes = []; this.appendChild(new Text(text)); }
  scrollIntoView() {}
  all(test, out = []) {
    this.childNodes.forEach(k => { if (k instanceof Elem) { if (test(k)) out.push(k); k.all(test, out); } });
    return out;
  }
}
for (const name of ['innerHTML', 'outerHTML']) {
  Object.defineProperty(Elem.prototype, name, {get: forbidden(name), set: forbidden(name)});
}
Elem.prototype.insertAdjacentHTML = forbidden('insertAdjacentHTML');

const html = fs.readFileSync(process.argv[2], 'utf8');
const blocks = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
const dataBlock = blocks.find(b => /application\/json/.test(b[1]));
const code = blocks.find(b => !/application\/json/.test(b[1]))[2];

const body = new Elem('body');
const byId = {};
for (const id of ['head', 'controls', 'cases', 'empty', 'data']) {
  const el = new Elem(id === 'cases' ? 'table' : 'div');
  el.setAttribute('id', id); byId[id] = el; body.appendChild(el);
}
byId.data.textContent = dataBlock[2];
const document = {
  createElement: tag => new Elem(tag), createTextNode: text => new Text(text), title: '',
  getElementById: id => body.all(e => e.id === id)[0] || null,
};
const context = vm.createContext({document, Node, location: {hash: ''}, history: {replaceState() {}},
  window: {addEventListener() {}}});
vm.runInContext(code, context);

const run = js => vm.runInContext(js, context);
run('setAll(true)');                                     // every case opens: facts, questions
body.all(e => e.className === 'logbox').forEach(box => { box.open = true; box.fire('toggle'); });
body.all(e => e.tagName === 'details' && e.className.startsWith('rec')).forEach(r => { r.open = true; r.fire('toggle'); });
const checkbox = body.all(e => e.attrs.type === 'checkbox' && e.parentNode.tagName === 'label');
checkbox.forEach(c => { c.checked = true; c.fire('change'); c.checked = false; c.fire('change'); });
const search = body.all(e => e.attrs.type === 'search')[0];
if (search) { search.value = 'zzz-no-such-text'; search.fire('input'); search.value = ''; search.fire('input'); }
run('headers').forEach(th => { th.fire('click'); th.fire('click'); th.fire('click'); });   // each way of sorting

const everything = body.all(() => true);
const out = {
  text: body.textContent,
  title: document.title,
  tags: everything.reduce((n, e) => { n[e.tagName] = (n[e.tagName] || 0) + 1; return n; }, {}),
  attributeNames: [...new Set(everything.flatMap(e => Object.keys(e.attrs)))].sort(),
  pictures: everything.filter(e => e.tagName === 'img').map(e => e.attrs.src),
  links: everything.filter(e => e.tagName === 'a').map(e => e.attrs.href),
  count: document.getElementById('count') ? document.getElementById('count').textContent : null,
  cases: run('rows.length'),
};
// A picture that cannot load (the page served over http): each falls back to its link as text, and a note appears.
everything.filter(e => e.tagName === 'img').forEach(img => img.fire('error'));
const after = body.all(() => true);
out.fallback = {
  pictures: after.filter(e => e.tagName === 'img').length,
  notes: after.filter(e => e.className === 'note frames' && !e.hidden).map(e => e.textContent),
  links: after.filter(e => e.tagName === 'a' && e.className === 'failed').map(e => e.textContent),
};

// The stream of each case, frame by frame (frames.js): the section opens and a reader's moves are made on it, each
// noted by what the box then shows, so a test can check the frame, the pictures, the marks and the links to questions.
body.all(e => e.className === 'scrubbox').forEach(box => { box.open = true; box.fire('toggle'); });
out.scrub = body.all(e => e.className === 'scrub').map(box => {
  const find = (test) => box.all(test);
  const act = name => find(e => e.attrs['data-act'] === name)[0];
  const read = () => ({
    counter: find(e => e.attrs['data-role'] === 'counter')[0].textContent,
    facts: find(e => e.className === 'finfo')[0].textContent,
    detections: find(e => e.className === 'fdets')[0].textContent,
    rows: find(e => e.className === 'fdets')[0].all(e => e.tagName === 'tr').length,
    pictures: find(e => e.tagName === 'img').map(e => e.attrs.src),
  });
  const slider = find(e => e.attrs.type === 'range')[0];
  const mark = e => e.className.startsWith('qmark');
  const moves = [['start', () => {}], ['next', () => act('next').fire('click')], ['next', () => act('next').fire('click')],
    ['prev', () => act('prev').fire('click')], ['next-window', () => act('next-window').fire('click')],
    ['prev-window', () => act('prev-window').fire('click')],
    ['slider to the end', () => { slider.value = slider.attrs.max; slider.fire('input'); }],
    ['key ArrowRight at the end', () => box.fire('keydown', {key: 'ArrowRight'})],
    ['key ArrowLeft', () => box.fire('keydown', {key: 'ArrowLeft'})], ['key Home', () => box.fire('keydown', {key: 'Home'})],
    ['key ArrowLeft at the start', () => box.fire('keydown', {key: 'ArrowLeft'})], ['key ArrowRight', () => box.fire('keydown', {key: 'ArrowRight'})],
    ['key End', () => box.fire('keydown', {key: 'End'})], ['key x', () => box.fire('keydown', {key: 'x'})],
    ['first during question', () => { const m = find(mark).find(e => !e.className.includes('end')); if (m) m.fire('click'); }]];
  const frames = moves.map(([name, move]) => { move(); return {move: name, ...read()}; });
  return {frames, text: box.textContent, sliderMax: slider.attrs.max,
    marks: {windowEnds: find(e => e.className === 'tick').length, tasks: find(e => e.className === 'tick task').map(e => e.attrs.title),
      questions: find(mark).map(e => ({label: e.textContent, title: e.attrs.title, click: e.listeners.click ? 1 : 0}))},
    links: find(e => e.tagName === 'button' && e.className === 'link').length,
    pictureSlots: find(e => e.tagName === 'figure').length};
});
// A question's link in a frame takes the reader to its row, which is lit up; a picture that cannot load is its link as text.
const lit = () => body.all(e => e.attrs['data-flash'] !== undefined).length;
const link = body.all(e => e.className === 'scrub').flatMap(box => box.all(e => e.tagName === 'button' && e.className === 'link'))[0];
const before = lit();
if (link) link.fire('click');
out.flash = {links: !!link, before, after: lit()};
const scrubbed = body.all(e => e.className === 'scrub')[0];
const slot = scrubbed && scrubbed.all(e => e.tagName === 'figure')[0];
if (slot) {
  const [img, miss] = slot.childNodes;
  const shown = () => scrubbed.all(e => e.className === 'note frames' && !e.hidden).length;
  const before_note = shown();
  img.fire('error');
  out.slot = {img: img.hidden, miss: miss.hidden, text: miss.textContent, notes: [before_note, shown()]};
}
console.log(JSON.stringify(out));
