// The transport log of a case: a system's own journal, one record a line. It is kept generic: a record with an `event`
// is a note (run settings, an answer's outcome, a failed exec); any other record is a call to the model, and whatever
// of `prompt`, `tools`, `messages`, `usage`, `warnings`, `stderr` and `error` it holds is shown. The rest is JSON.

const SHOWN = ['i', 't', 'purpose', 'kind', 'event', 'prompt', 'tools', 'messages', 'usage', 'latency_s', 'warnings',
  'stderr', 'error', 'event_errors', 'ask'];
const PREVIEW = 400;

const isEvent = r => typeof r.event === 'string';
const callsOf = r => Array.isArray(r.tools) ? r.tools.filter(t => t && typeof t === 'object') : [];
const scalar = v => v === null || ['string', 'number', 'boolean'].includes(typeof v);
const present = v => Array.isArray(v) ? v.length > 0 : v !== null && v !== undefined && v !== '' && v !== false;

function recordState(r) {
  const failed = (isEvent(r) && /fail|error/i.test(r.event)) || present(r.error) || present(r.event_errors) ||
    callsOf(r).some(t => present(t.error));
  return failed ? 'bad' : (present(r.warnings) || present(r.stderr)) ? 'warn' : '';
}

function recordTitle(r) {
  if (isEvent(r)) return r.event;
  return ['exec' + (r.i !== undefined ? ' #' + r.i : ''), r.purpose, r.kind].filter(Boolean).join(' · ');
}

function numbers(obj) {
  return Object.keys(obj).filter(k => typeof obj[k] === 'number').map(k => k + ' ' + fmtInt(obj[k])).join(' · ');
}

function recordChips(r) {
  const bits = [];
  if (isEvent(r)) {
    Object.keys(r).filter(k => k !== 'event' && scalar(r[k])).forEach(k => bits.push(k + ' ' + r[k]));
    return clipped(bits.join(' · '), 160);
  }
  if (typeof r.latency_s === 'number') bits.push(num(r.latency_s, 1) + ' s');
  if (r.usage && typeof r.usage === 'object') {
    const u = r.usage;
    bits.push('tokens ' + fmtTokens(u.prompt_tokens) + ' in, ' + fmtTokens(u.completion_tokens) + ' out');
  }
  if (callsOf(r).length) bits.push(callsOf(r).length + ' tool call' + (callsOf(r).length === 1 ? '' : 's'));
  if (r.attempts > 1) bits.push(r.attempts + ' attempts');
  return bits.join(' · ');
}

function part(label, ...kids) {
  return h('div', {class: 'part'}, h('div', {class: 'label'}, label), kids);
}

function textOf(v) { return typeof v === 'string' ? v : json(v); }

function renderCall(t) {
  const failed = present(t.error);
  return h('div', {class: 'call' + (failed ? ' err' : '')},
    h('span', {class: 'tool'}, t.tool === undefined ? 'call' : t.tool), ' ',
    h('span', {class: 'mono'}, clipped(textOf(t.arguments), PREVIEW)),
    t.result === undefined ? null : h('div', {class: 'muted'}, ARROW.trim() + ' ', clipped(textOf(t.result), PREVIEW)),
    failed ? h('div', {class: 'neg'}, 'error: ', clipped(textOf(t.error), PREVIEW)) : null);
}

function recordBody(r) {
  const out = [];
  if (r.ask) out.push(part('asked', h('span', {class: 'mono'}, clipped(json(r.ask), PREVIEW))));
  if (r.prompt) {
    const p = r.prompt;
    const text = typeof p === 'string' ? p : p.text;
    const images = p && Array.isArray(p.images) ? p.images.length : 0;
    out.push(part('prompt' + (images ? ' (' + images + ' images attached)' : ''),
      h('pre', null, clipped(textOf(text === undefined ? p : text), PREVIEW))));
  }
  const calls = callsOf(r);
  if (calls.length) out.push(part('tool calls', calls.map(renderCall)));
  if (Array.isArray(r.messages) && r.messages.length) out.push(part('reply', r.messages.map(m => h('div', null, clipped(textOf(m), PREVIEW)))));
  if (r.usage && typeof r.usage === 'object') out.push(part('usage', h('span', {class: 'mono'}, numbers(r.usage) || json(r.usage))));
  for (const key of ['error', 'event_errors', 'warnings', 'stderr']) {
    if (present(r[key])) out.push(part(key.replace('_', ' '), h('pre', {class: key === 'warnings' || key === 'stderr' ? '' : 'neg'}, clipped(textOf(r[key]), PREVIEW))));
  }
  const rest = {};
  Object.keys(r).filter(k => !SHOWN.includes(k) && !(isEvent(r) && scalar(r[k]))).forEach(k => { rest[k] = r[k]; });
  if (Object.keys(rest).length) out.push(part('other fields', h('pre', null, clipped(json(rest), PREVIEW))));
  return out;
}

// One record: a line that opens into its body when it is first opened.
function renderRecord(r) {
  const state = recordState(r);
  const badge = state ? h('span', {class: 'badge ' + (state === 'bad' ? 'o-wrong' : 'o-partial')}, state === 'bad' ? 'error' : 'warning') : null;
  const box = h('details', {class: 'rec ' + state, 'data-state': state},
    h('summary', null, h('strong', null, recordTitle(r)), badge, h('span', {class: 'chips'}, recordChips(r))));
  box.addEventListener('toggle', () => {
    if (box.open && !box.dataset.built) { box.dataset.built = '1'; add(box, recordBody(r)); }
  });
  return box;
}

function transportFacts(t) {
  const rows = [];
  for (const key of Object.keys(t.summary)) {
    if (key === 'wall_s') continue;
    const v = t.summary[key];
    rows.push(h('dt', null, key), h('dd', {class: 'mono'}, clipped(scalar(v) ? String(v) : json(v), 160)));
  }
  return h('div', null, h('div', {class: 'label'}, t.name + '.summary.json'), h('dl', {class: 'facts'}, rows));
}

function renderLog(c) {
  const log = c.log, recs = log.records;
  const calls = recs.reduce((n, r) => n + callsOf(r).length, 0);
  const states = recs.map(recordState);
  const bad = states.filter(s => s === 'bad').length, warn = states.filter(s => s === 'warn').length;
  const list = h('div', {class: 'recs'}, recs.map(renderRecord));
  const only = h('input', {type: 'checkbox'});
  only.addEventListener('change', () => {
    list.childNodes.forEach(node => { node.hidden = only.checked && !node.dataset.state; });
  });
  const notes = [];
  if (log.cut) notes.push(log.cut + ' long text' + (log.cut === 1 ? '' : 's') + ' cut at 20,000 characters');
  if (log.left_out.length) notes.push('left out of the page: ' + log.left_out.join(', '));
  return h('div', {class: 'log'},
    h('div', {class: 'bar'},
      h('span', null, recs.length + ' records · ' + calls + ' tool calls · ' + bad + ' with errors · ' + warn + ' with warnings'),
      h('label', null, only, ' errors and warnings only'),
      h('span', {class: 'muted small'}, log.file, notes.length ? ' · ' + notes.join(' · ') : '')),
    c.transport ? transportFacts(c.transport) : null,
    list);
}
