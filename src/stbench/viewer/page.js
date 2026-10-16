// The page: the run's header, a sortable and filterable table of its cases, and each case opened in place (facts,
// questions with answer and gold, transport log), built the first time it is opened.

const DATA = JSON.parse(document.getElementById('data').textContent);
const OUTCOMES = ['correct', 'partial', 'wrong', 'abstained'];
const state = {arm: 0, sort: null, text: '', status: 'all', layout: '', job: '', length: '', wrong: false};
let rows = [];   // one {c, body, tr, detail, built, open} a case of the arm on show
let headers = [];   // the table's column headers, in the order of COLUMNS
let multi = DATA.arms.length > 1;

const arm = () => DATA.arms[state.arm];
const tokensOf = c => c.tokens && c.tokens.tokens_in !== null && c.tokens.tokens_out !== null ? c.tokens.tokens_in + c.tokens.tokens_out : null;

const COLUMNS = [
  {key: 'id', label: 'Case', get: c => c.id},
  {key: 'layout', label: 'Layout', get: c => c.layout},
  {key: 'job', label: 'Job', get: c => c.job},
  {key: 'length', label: 'Length', get: c => c.length},
  {key: 'status', label: 'Status', get: c => c.status},
  {key: 'n', label: 'Questions', num: true, get: c => c.counts.n, show: fmtInt},
  {key: 'correct', label: 'Correct', num: true, get: c => c.counts.correct, show: fmtInt},
  {key: 'partial', label: 'Partial', num: true, get: c => c.counts.partial, show: fmtInt},
  {key: 'wrong', label: 'Wrong', num: true, get: c => c.counts.wrong, show: fmtInt},
  {key: 'abstained', label: 'Abstained', num: true, get: c => c.counts.abstained, show: fmtInt},
  {key: 'mean', label: 'Mean score', num: true, get: c => c.counts.mean, show: v => signed(v, 3)},
  {key: 'wall', label: 'Wall time', num: true, get: c => c.wall_s, show: fmtSeconds},
  {key: 'tokens', label: 'Tokens', num: true, get: tokensOf, show: fmtTokens},
  {key: 'cost', label: 'Cost', num: true, get: c => c.cost_usd, show: fmtCost},
];

// ---- the header -----------------------------------------------------------------------------------------------
function fact(label, value, wide) {
  return value === null || value === undefined || value === '' ? null : h('div', {class: wide ? 'wide' : ''}, h('dt', null, label), h('dd', null, value));
}

// An ability's line: the mean of the questions the means count (all of them if none is), and how many of its questions.
function abilityText(name, a) {
  const kept = a.counted > 0, mean = kept ? a.counted_mean : a.mean;
  return name + ' ' + signed(mean, 3) + ' over ' + fmtInt(kept ? a.counted : a.n) + (kept && a.counted < a.n ? ' of ' + fmtInt(a.n) : '') + (a.diagnostic ? ' (diagnostic)' : '');
}

// The headline is results.json's: eval writes that file when the run begins and when it ends, so a run in progress has
// none to show, and the viewer computes no statistic of its own.
function layoutTable(a) {
  const names = Object.keys(a.layouts);
  if (!names.some(name => a.layouts[name])) {
    return h('p', {class: 'muted small'}, 'No headline yet: it is read from results.json, which eval writes when the run ends' +
      (a.results.present ? ' (the one here is from the start of the run and holds none).' : ' (there is none yet).'));
  }
  const layouts = a.layouts;
  const body = names.map(name => {
    const l = layouts[name];
    if (!l) return h('tr', null, h('td', null, name), h('td', {class: 'muted', colspan: 5}, 'none in results.json'));
    const hl = l.headline || {}, ci = hl.ci, sc = l.columns || {}, ab = l.abilities || {};
    return h('tr', null, h('td', {class: 'mono'}, name),
      h('td', null, h('strong', null, signed(hl.mean, 3)), ci ? ' [' + signed(ci[0], 3) + ', ' + signed(ci[1], 3) + ']' : ' (no interval)'),
      h('td', null, ['now', 'memory'].filter(k => sc[k]).map(k => k + ' ' + signed(sc[k].mean, 3) + ' over ' + fmtInt(sc[k].n)).join(', ')),
      h('td', null, Object.keys(ab).map(k => abilityText(k, ab[k])).join(', ')),
      h('td', null, fmtInt(l.n) + ' questions, ' + fmtInt(l.cases) + ' cases'),
      h('td', null, l.counts ? OUTCOMES.map(o => l.counts[o] + ' ' + o).join(', ') : ''));
  });
  return h('div', {class: 'scroll'}, h('table', {class: 'layouts'},
    h('tr', null, h('th', null, 'Layout'), h('th', null, 'Headline (95% interval)'), h('th', null, 'now / memory columns (abilities equal in each)'), h('th', null, 'Abilities'), h('th', null, 'Scored'), h('th', null, 'Outcomes')), body));
}

// Cases finished are counted from the case folders, as the header does; results.json's own count may be older.
function runState(a) {
  const t = a.totals, r = a.results;
  if (t.finished >= t.cases) return null;
  let why = '';
  if (!r.present) why = ' There is no results.json yet: eval writes it when the run ends.';
  else if (r.finished !== t.finished) why = ' results.json lists ' + r.finished + ' finished: eval writes it when the run begins and when it ends, not in between.';
  return 'Unfinished: ' + t.finished + ' of ' + t.cases + ' cases finished (counted from the case folders), so this is not a benchmark score.' + why;
}

function renderHead() {
  const a = arm(), hd = a.header, t = a.totals, tk = t.tokens;
  const system = hd.system || '(system not recorded)';
  const tokens = tk.tokens_in === null ? null : fmtInt(tk.tokens_in) + ' in (' + fmtInt(tk.tokens_cached_in) + ' cached), ' + fmtInt(tk.tokens_out) + ' out';
  const cost = t.cost_usd === null ? null : fmtCost(t.cost_usd) + (t.unpriced ? ' + ' + t.unpriced + ' unpriced case(s)' : '');
  const state = runState(a);
  const notes = [...(state ? [[state, '']] : []), ...(hd.notes || []).map(n => [n, '']), ...a.problems.map(p => [p, ' bad'])];
  fill(document.getElementById('head'),
    h('h1', null, 'stbench run view: ' + system),
    h('div', {class: 'muted small'}, 'Written after the run, from its folders and the dataset. It holds the gold: for troubleshooting a system, never for the system.'),
    notes.map(([n, cls]) => h('div', {class: 'note' + cls}, n)),
    h('dl', {class: 'facts'},
      fact('System', system), fact('Model', hd.model), fact('Reasoning', hd.reasoning), fact('Channel', hd.channel), fact('Brief', hd.brief),
      fact('Images', hd.text_only === true ? 'withheld (text only)' : null),
      fact('During questions', hd.during === null || hd.during === undefined ? null : hd.during ? 'on' : 'off'),
      fact('Sandbox', hd.sandbox), fact('Verified', hd.verified === null || hd.verified === undefined ? null : hd.verified ? 'yes' : 'NO'),
      fact('Cases finished', t.finished + ' of ' + t.cases + (t.unlisted ? ' (and ' + t.unlisted + ' not in results.json)' : '')), fact('Tokens', tokens), fact('Cost', cost),
      fact('Wall time (messages)', t.wall_s ? fmtSeconds(t.wall_s) : null),
      fact('Run folder', h('span', {class: 'mono'}, a.dir), true), fact('Dataset', DATA.dataset ? h('span', {class: 'mono'}, DATA.dataset) : null, true)),
    layoutTable(a));
  document.title = 'stbench view: ' + (hd.system || DATA.run);
}

// ---- the controls and the table --------------------------------------------------------------------------------
function options(select, values, first) {
  fill(select, h('option', {value: ''}, first), values.map(v => h('option', {value: v}, v)));
}

function renderControls() {
  const a = arm(), uniq = key => [...new Set(a.cases.map(c => c[key]).filter(Boolean))].sort();
  const text = h('input', {type: 'search', placeholder: 'filter: case, job, status, question or answer text', value: state.text});
  text.addEventListener('input', () => { state.text = text.value; applyFilters(); });
  const status = h('select', null, h('option', {value: 'all'}, 'all statuses'), h('option', {value: 'complete'}, 'complete'), h('option', {value: 'not'}, 'not complete'));
  status.value = state.status;
  status.addEventListener('change', () => { state.status = status.value; applyFilters(); });
  const pick = (key, label) => {
    const s = h('select', null);
    options(s, uniq(key), label);
    s.value = state[key];
    s.addEventListener('change', () => { state[key] = s.value; applyFilters(); });
    return s;
  };
  const wrong = h('input', {type: 'checkbox'});
  wrong.checked = state.wrong;
  wrong.addEventListener('change', () => { state.wrong = wrong.checked; applyFilters(); });
  const armSelect = h('select', {title: 'run folder'}, DATA.arms.map((x, i) => h('option', {value: i}, x.name || '(this run)')));
  armSelect.value = state.arm;
  armSelect.addEventListener('change', () => { state.arm = Number(armSelect.value); show(); });
  fill(document.getElementById('controls'),
    multi ? armSelect : null, text, status, pick('layout', 'all layouts'), pick('job', 'all jobs'), pick('length', 'all lengths'),
    h('label', null, wrong, ' has wrong answers'),
    h('button', {class: 'tool', type: 'button', on: {click: () => setAll(true)}}, 'Expand all'),
    h('button', {class: 'tool', type: 'button', on: {click: () => setAll(false)}}, 'Collapse all'),
    h('span', {class: 'muted', id: 'count'}, ''));
}

function haystack(c) {
  if (c.hay === undefined) {
    const bits = [c.id, c.script, c.layout, c.robot, c.job, c.length, c.status, c.error, json(c.failures)];
    c.questions.forEach(q => bits.push(q.text, q.family, formatAnswer(q.answer), formatAnswer(q.gold)));
    c.hay = bits.filter(Boolean).join('\n').toLowerCase();
  }
  return c.hay;
}

function matches(c) {
  const f = state;
  if (f.status === 'complete' && c.kind !== 'complete') return false;
  if (f.status === 'not' && c.kind === 'complete') return false;
  if ((f.layout && c.layout !== f.layout) || (f.job && c.job !== f.job) || (f.length && c.length !== f.length)) return false;
  if (f.wrong && !c.counts.wrong) return false;
  return !f.text || haystack(c).includes(f.text.toLowerCase());
}

function applyFilters() {
  let shown = 0;
  rows.forEach(r => { r.body.hidden = !matches(r.c); if (!r.body.hidden) shown += 1; });
  document.getElementById('empty').hidden = shown > 0;
  const count = document.getElementById('count');
  if (count) count.textContent = 'showing ' + shown + ' of ' + rows.length;
}

function compare(col, dir) {
  return (x, y) => {
    const a = col.get(x.c), b = col.get(y.c);
    if (a === null || a === undefined || a === '') return (b === null || b === undefined || b === '') ? 0 : 1;
    if (b === null || b === undefined || b === '') return -1;
    return dir * (typeof a === 'number' ? a - b : String(a).localeCompare(String(b)));
  };
}

function sortRows() {
  const order = rows.slice();
  if (state.sort) order.sort(compare(COLUMNS.find(c => c.key === state.sort.key), state.sort.dir));
  const table = document.getElementById('cases');
  order.forEach(r => table.appendChild(r.body));
  headers.forEach((th, i) => {
    const on = state.sort && state.sort.key === COLUMNS[i].key;
    if (on) th.setAttribute('aria-sort', state.sort.dir > 0 ? 'ascending' : 'descending');
    else th.removeAttribute('aria-sort');
  });
}

function clickHeader(col) {
  const first = col.num ? -1 : 1;   // text sorts up first, numbers down; a second click reverses, a third clears
  const same = state.sort && state.sort.key === col.key;
  state.sort = !same ? {key: col.key, dir: first} : state.sort.dir === first ? {key: col.key, dir: -first} : null;
  sortRows();
}

function caseRow(c) {
  const cells = COLUMNS.map(col => {
    const v = col.get(c);
    if (col.key === 'id') return h('td', {class: 'mono'}, h('span', {class: 'chev'}), c.id, c.listed ? null : h('span', {class: 'muted small', title: 'results.json does not list this case folder'}, ' (not in results)'));
    if (col.key === 'status') return h('td', {class: 'status-cell', title: c.status + (c.error ? ': ' + c.error : '')}, h('span', {class: 'pill st-' + c.kind}, c.status));
    if (!col.num) return h('td', null, v === null || v === undefined ? DASH : v);
    const cls = 'num' + (OUTCOMES.includes(col.key) ? ' c-' + col.key : '') + (v === 0 ? ' zero' : '');
    const title = col.key === 'tokens' && c.tokens && c.tokens.tokens_in !== null
      ? fmtInt(c.tokens.tokens_in) + ' in (' + fmtInt(c.tokens.tokens_cached_in) + ' cached), ' + fmtInt(c.tokens.tokens_out) + ' out' : null;
    return h('td', {class: cls + (col.key === 'mean' && typeof v === 'number' ? (v < 0 ? ' neg' : v > 0 ? ' pos' : '') : ''), title}, v === null || v === undefined ? DASH : col.show(v));
  });
  const tr = h('tr', {class: 'case', tabindex: 0, 'aria-expanded': 'false'}, cells);
  const detail = h('tr', {class: 'detail', hidden: true}, h('td', {colspan: COLUMNS.length}));
  const row = {c, body: h('tbody', {id: 'case-' + c.id}, tr, detail), tr, detail, built: false, open: false};
  tr.addEventListener('click', () => toggle(row));
  tr.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); toggle(row); } });
  return row;
}

function setHash(row) {
  try {
    history.replaceState(null, '', '#case-' + encodeURIComponent(row.c.id) + (multi ? '@' + encodeURIComponent(arm().name) : ''));
  } catch (e) { /* a page opened from some places may not change its address */ }
}

function toggle(row, open) {
  row.open = open === undefined ? !row.open : open;
  row.tr.setAttribute('aria-expanded', String(row.open));
  row.detail.hidden = !row.open;
  if (row.open && !row.built) { row.built = true; row.detail.firstChild.appendChild(renderCase(row.c)); }
  if (row.open && open === undefined) setHash(row);
}

function setAll(open) { rows.forEach(r => { if (!r.body.hidden) toggle(r, open); }); }

function renderTable() {
  const table = document.getElementById('cases');
  headers = COLUMNS.map(col => {
    const th = h('th', {class: col.num ? 'num' : '', title: 'sort by ' + col.label}, col.label);
    th.addEventListener('click', () => clickHeader(col));
    return th;
  });
  fill(table, h('thead', null, h('tr', null, headers)));
  rows = arm().cases.map(caseRow);
  rows.forEach(r => table.appendChild(r.body));
  sortRows();
  applyFilters();
  if (rows.length === 1) toggle(rows[0], true);
}

// ---- one case ---------------------------------------------------------------------------------------------------
function facts(c) {
  const w = c.window, tk = c.tokens || {};
  const failures = Object.keys(c.failures).map(k => k + ': ' + c.failures[k]).join('; ');
  const by = Object.keys(c.wall_by).map(k => k + ' ' + fmtSeconds(c.wall_by[k])).join(' · ');
  const cn = c.counts;
  const items = [
    ['Script', h('span', {class: 'mono'}, c.script || DASH)],
    ['Place and job', [c.layout, c.robot, c.job, c.length].filter(Boolean).join(' · ')],
    ['Window', w ? 'script tasks ' + w.a + ' to ' + w.b + ' (keyframes ' + w.k0 + ' to ' + w.k1 + '), starts ' + w.start_clock : null],
    ['Status', [h('span', {class: 'pill st-' + c.kind}, c.status), c.error ? ' ' + c.error : null]],
    ['Failed questions', failures || null],
    ['Questions scored', cn.n ? cn.n + ': ' + OUTCOMES.map(o => cn[o] + ' ' + o).join(', ') + '; mean score ' + signed(cn.mean, 3) : null],
    ['Wall time', c.wall_s === null ? null : fmtSeconds(c.wall_s) + (by ? ' (' + by + ')' : '')],
    ['Tokens', tk.tokens_in === null || tk.tokens_in === undefined ? null : fmtInt(tk.tokens_in) + ' in (' + fmtInt(tk.tokens_cached_in) + ' cached, ' + fmtInt(tk.tokens_cache_write_in) + ' written to a cache), ' + fmtInt(tk.tokens_out) + ' out'],
    ['Cost', c.cost_usd === null ? c.cost_note : fmtCost(c.cost_usd) + (c.cost_note ? ' (' + c.cost_note + ')' : '')],
    ['Usage report', c.usage_problems.length ? c.usage_problems.join('; ') : null],
    ['State bytes', Object.keys(c.state_bytes).length ? Object.keys(c.state_bytes).map(k => k + ': ' + fmtInt(c.state_bytes[k])).join(' · ') : null],
  ].filter(it => it[1] !== null && it[1] !== undefined && it[1] !== '');
  return h('dl', {class: 'facts'}, items.map(it => [h('dt', null, it[0]), h('dd', null, it[1])]));
}

const tip = v => json(v).slice(0, 2000);

function whenText(q) { return q.when === 'during' ? 'during' + (q.step === null ? '' : ', after step ' + q.step) : (q.when || DASH); }

// A picture the browser cannot load (a page served over http may not read file: addresses) becomes its link as text.
function frameLink(e, note) {
  const img = h('img', {src: e.url, loading: 'lazy', alt: 'keyframe ' + e.k});
  const link = h('a', {href: e.url, target: '_blank', rel: 'noopener', title: 'keyframe ' + e.k + ' of the script'}, img, 'k' + e.k);
  img.addEventListener('error', () => {
    link.className = 'failed';
    link.replaceChildren('k' + e.k + ' ' + e.url);
    note.hidden = false;
  });
  return link;
}

function evidenceCell(q, note) {
  if (!q.evidence.length) return null;
  const shown = q.evidence.slice(0, 4);
  const more = q.evidence.length - shown.length;
  return h('div', {class: 'evid'}, shown.map(e => e.url
    ? frameLink(e, note)
    : h('span', {class: 'muted small', title: 'the dataset has no picture for this keyframe'}, 'k' + e.k)),
  more > 0 ? h('span', {class: 'muted small', title: q.evidence.map(e => 'k' + e.k).join(' ')}, '+' + more + ' more') : null);
}

function questionRow(q, i, note) {
  const skipped = !q.asked, wrong = q.outcome === 'wrong';
  const answer = skipped ? 'not asked' : formatAnswer(q.answer);
  const off = skipped ? '' : offBy(q);
  const cls = wrong ? 'wrong' : skipped ? 'skipped' : '';
  return h('tr', {class: cls},
    h('td', {class: 'nowrap muted'}, i + 1),
    h('td', {class: 'nowrap'}, whenText(q)),
    h('td', {class: 'nowrap'}, q.family || DASH, h('div', {class: 'muted small'}, [q.scope, q.reference, q.answer_class].filter(Boolean).join(' · '))),
    h('td', null, q.text === null ? h('span', {class: 'muted'}, '(no question text: the dataset does not have this case)') : q.text, h('div', {class: 'muted small mono'}, q.qid)),
    h('td', {class: 'ans', title: skipped ? null : tip(q.answer)}, clipped(answer, 300), off ? h('div', {class: 'muted small'}, off) : null),
    h('td', {class: 'ans', title: q.gold === null ? null : tip(q.gold)}, q.gold === null ? DASH : clipped(formatAnswer(q.gold), 300), h('div', {class: 'muted small'}, creditText(q))),
    h('td', {class: 'nowrap'}, h('span', {class: 'pill o-' + (q.outcome || 'none')}, q.outcome || (skipped ? 'not asked' : DASH))),
    h('td', {class: 'nowrap'}, signed(q.score, 2), q.s === null || q.s === undefined ? null : h('div', {class: 'muted small'}, 'credit ' + num(q.s, 2))),
    h('td', {class: 'nowrap'}, q.critical === true ? 'critical' : q.critical === false ? DASH : ''),
    h('td', {class: 'nowrap'}, q.slice || ''),
    h('td', null, evidenceCell(q, note)));
}

// QROWS gets each question's row by its qid, so the stream below can take the reader to it.
function questionsTable(c, qrows) {
  if (!c.questions.length) return h('p', {class: 'muted'}, 'No question for this case: the run holds no rows or answers and the dataset no questions.');
  const note = h('p', {class: 'note frames', hidden: true},
    'The keyframe pictures load when this page is opened from disk (a file:// address): a page served over http cannot read them, so their links are shown as text.');
  const heads = ['#', 'When', 'Family', 'Question', "Solver's answer", 'Gold', 'Outcome', 'Score', 'Critical', 'Slice', 'Evidence frames'];
  return h('div', {class: 'qbox'}, note, h('div', {class: 'scroll'}, h('table', {class: 'q'},
    h('thead', null, h('tr', null, heads.map(x => h('th', null, x)))),
    h('tbody', null, c.questions.map((q, i) => { const tr = questionRow(q, i, note); qrows[q.qid] = tr; return tr; })))));
}

function renderCase(c) {
  const logBox = h('details', {class: 'logbox'}, h('summary', null, c.log ? 'Transport log (' + c.log.records.length + ' records)' : 'Transport log'));
  if (c.log) {
    logBox.addEventListener('toggle', () => {
      if (logBox.open && !logBox.dataset.built) { logBox.dataset.built = '1'; logBox.appendChild(renderLog(c)); }
    });
  } else {
    logBox.appendChild(h('p', {class: 'muted'}, 'This case has no transport log: the system wrote none (a scripted system has nothing to log).'));
  }
  const qrows = {};
  return h('div', {class: 'detail-body'},
    c.problems.map(p => h('div', {class: 'note'}, p)),
    facts(c), h('h2', null, 'Questions, in the order asked'), questionsTable(c, qrows), streamBox(c, qrows, DATA.depth_dir), logBox);
}

// ---- start ------------------------------------------------------------------------------------------------------
function show() {
  renderHead();
  renderControls();
  renderTable();
}

function openFromHash() {
  let m = null;
  try { m = /^#case-([^@]+)(?:@(.*))?$/.exec(decodeURIComponent(location.hash)); } catch (e) { return; }
  if (!m) return;
  let at = DATA.arms.findIndex(a => a.name === m[2]);
  if (at < 0) at = DATA.arms[state.arm].cases.some(c => c.id === m[1]) ? state.arm : DATA.arms.findIndex(a => a.cases.some(c => c.id === m[1]));
  if (at < 0) return;
  if (at !== state.arm) { state.arm = at; show(); }
  const row = rows.find(r => r.c.id === m[1]);
  if (row) { toggle(row, true); row.tr.scrollIntoView(); }
}

if (DATA.arms.length) { show(); openFromHash(); }
else fill(document.getElementById('head'), h('h1', null, 'stbench run view'), h('div', {class: 'note bad'}, 'The run folder holds no run.'));
window.addEventListener('hashchange', openFromHash);
