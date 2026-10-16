// The stream of a case, frame by frame: the steps the harness delivered to the system, one at a time, each with its
// pictures and numbers and with where look windows, tasks and questions fall. The reader packed the steps as numbers
// (`view_frames`); only the current frame is on the page, and its pictures are the dataset's files by `file:` address
// (the depth ones drawn for viewing). A step's number is its k in the case, counting from 0; the counter counts from 1.

const PICTURE_NOTE = 'The pictures load when this page is opened from disk (a file:// address): a page served over http cannot read them, so their links are shown as text.';

const pad6 = n => String(n).padStart(6, '0');
const fileOnly = url => typeof url === 'string' && /^file:/i.test(url) ? url : '';
const pct = (i, total) => total > 1 ? Math.min(100, Math.max(0, 100 * i / (total - 1))) : 0;
const pairs = obj => Object.keys(obj || {}).sort().map(k => k + ' ' + obj[k]).join(', ');
const xyz = v => '(' + v.map(x => num(x, 3)).join(', ') + ')';
const shown = v => v === null || v === undefined ? DASH : scalar(v) ? String(v) : json(v);

// The case's clock at SECONDS after CLOCK0 (a local ISO time, to the tenth of a second as the stream delivers it;
// UTC arithmetic keeps it free of the viewer's time zone).
function clockAt(clock0, seconds) {
  const m = /^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d))?$/.exec(String(clock0));
  if (!m) return String(clock0) + ' + ' + seconds + ' s';
  const ms = Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]) + 100 * (+(m[7] || 0)) + Math.round(1000 * seconds);
  return new Date(ms).toISOString().slice(0, 21);
}

// Which look window and which task each step is in, from the ends of the windows and the first steps of the tasks.
function streamShape(s) {
  const n = s.t.length, windowOf = [], bounds = [];
  let lo = 0;
  s.windows.forEach((hi, b) => {
    bounds.push([lo, Math.min(hi, n)]);
    for (let i = lo; i < hi && i < n; i++) windowOf[i] = b;
    lo = hi;
  });
  const tasks = s.tasks.filter(t => t[1] < n);
  return {n, windowOf, bounds, tasks, taskOf: i => tasks.filter(t => t[1] <= i).pop() || null};
}

// The questions by the step they were asked after (`during`) and those asked after the last step (`end`), numbered as
// the table of questions numbers them.
function streamQuestions(c, n) {
  const during = {}, end = [];
  c.questions.forEach((q, i) => {
    const m = {q, i};
    if (q.when === 'during' && typeof q.step === 'number' && q.step >= 0 && q.step < n) (during[q.step] = during[q.step] || []).push(m);
    else if (q.when === 'end') end.push(m);
  });
  return {during, end};
}

const questionLabel = m => 'Q' + (m.i + 1);
const questionTitle = m => questionLabel(m) + ' · ' + (m.q.family || DASH) + (m.q.asked ? '' : ' · not asked in this run') + (m.q.text ? ' · ' + m.q.text : '');

// Take the reader to a question's row in the table above and light it up.
function flash(row) {
  if (!row) return;
  row.scrollIntoView({block: 'center'});
  row.removeAttribute('data-flash');
  void row.offsetWidth;   // so the light starts again when the same question is asked for twice
  row.setAttribute('data-flash', '1');
}

// A picture that follows the frame: its address is set for each frame, and one that cannot load becomes its link as text.
function pictureSlot(label, note) {
  const img = h('img', {alt: label, decoding: 'async'});
  const miss = h('a', {class: 'failed', target: '_blank', rel: 'noopener', hidden: true});
  img.addEventListener('error', () => { img.hidden = true; miss.hidden = false; note.hidden = false; });
  const el = h('figure', {class: 'shot'}, img, miss, h('figcaption', {class: 'muted small'}, label));
  return {el, set(url) {
    img.hidden = false;
    miss.hidden = true;
    img.setAttribute('src', fileOnly(url));
    miss.setAttribute('href', fileOnly(url));
    miss.textContent = label + ' ' + url;
  }};
}

function detectionsTable(s, i) {
  const found = s.detections.rows[i];
  if (!found.length) return h('p', {class: 'muted'}, 'No detections in this frame.');
  const heads = ['Kind', 'Attributes', 'Map centre (m)', 'Camera centre (m)', 'Size (m)', 'Box (px)'];
  const body = found.map(r => {
    const tag = s.detections.tags[r[0]];
    return h('tr', null, h('td', {class: 'nowrap'}, tag[0]), h('td', null, [pairs(tag[1]), pairs(tag[2])].filter(Boolean).join('; ')),
      h('td', {class: 'nowrap'}, xyz(r.slice(5, 8))), h('td', {class: 'nowrap'}, xyz(r.slice(8, 11))),
      h('td', {class: 'nowrap'}, tag[3].map(x => num(x, 3)).join(' x ')),
      h('td', {class: 'nowrap'}, '(' + r[1] + ', ' + r[2] + ') to (' + r[3] + ', ' + r[4] + ')'));
  });
  return h('div', {class: 'scroll'}, h('table', {class: 'q dets'}, h('thead', null, h('tr', null, heads.map(x => h('th', null, x)))), h('tbody', null, body)),
    h('div', {class: 'muted small'}, found.length + ' detection' + (found.length === 1 ? '' : 's') + ', in the order delivered; their keys are fresh for each frame and are not shown.'));
}

// The strip under the slider: where each look window ends, where each task begins and where questions were asked.
function marksStrip(s, shape, marks, goTo) {
  const n = shape.n, cur = h('span', {class: 'curwindow'});
  const windowEnds = s.windows.filter(hi => hi < n).map(hi => h('span', {class: 'tick', style: 'left:' + pct(hi - 0.5, n) + '%'}));
  const tasks = shape.tasks.map(t => h('span', {class: 'tick task', style: 'left:' + pct(t[1] - 0.5, n) + '%', title: 'task ' + t[0] + ' begins at step ' + t[1]},
    h('span', {class: 'tlabel'}, 'T' + t[0])));
  const off = ms => ms.every(m => !m.q.asked) ? ' off' : '';   // a question the run did not ask (--during off) is dimmed
  const asked = Object.keys(marks.during).map(Number).sort((a, b) => a - b).map(step => {
    const ms = marks.during[step];
    return h('button', {class: 'qmark' + off(ms), type: 'button', style: 'left:' + pct(step, n) + '%', title: ms.map(questionTitle).join('\n'), on: {click: () => goTo(step)}}, ms.map(questionLabel).join(' '));
  });
  const end = marks.end.length ? h('button', {class: 'qmark end' + off(marks.end), type: 'button', title: marks.end.map(questionTitle).join('\n'), on: {click: () => goTo(n - 1)}}, 'end: ' + marks.end.map(questionLabel).join(' ')) : null;
  return {el: h('div', {class: 'marks'}, cur, windowEnds, tasks, asked, end), setWindow(b) {
    const [lo, hi] = shape.bounds[b] || [0, 0];
    const from = pct(lo - 0.5, n), to = pct(hi - 0.5, n);
    cur.setAttribute('style', 'left:' + from + '%;width:' + Math.max(0, to - from) + '%');
  }};
}

function questionLinks(list, qrows) {
  return list.map(m => [h('button', {class: 'link', type: 'button', title: questionTitle(m), on: {click: () => flash(qrows[m.q.qid])}}, questionLabel(m) + ' ' + (m.q.family || '')),
    m.q.asked ? null : ' (not asked in this run)', ' ']);
}

function taskText(shape, i) {
  const task = shape.taskOf(i);
  if (!task) return null;
  const first = task[1] === i, last = i === shape.n - 1 || shape.tasks.some(t => t[1] === i + 1);
  const gap = task[2] ? ', after ' + fmtInt(task[2]) + ' s not observed' : '';
  return 'task ' + task[0] + [first ? 'its first step (the task message arrives before it' + gap + ')' : null, last ? 'its last step' : null].filter(Boolean).map(x => ', ' + x).join('');
}

function frameFacts(s, shape, marks, qrows, i) {
  const b = shape.windowOf[i] || 0, [lo, hi] = shape.bounds[b] || [i, i + 1], p = s.pose ? s.pose[i] : null;
  const status = s.status ? s.status.keys.map((k, j) => k + ' ' + shown(s.status.rows[i][j])).join(' · ') : null;
  const items = [
    ['Step', 'k = ' + i + ' in the case (counting from 0)'],
    ['Clock', clockAt(s.clock0, s.t[i])],
    ['Look window', 'look window ' + (b + 1) + ' of ' + s.windows.length + ': steps ' + lo + ' to ' + (hi - 1) + ' (a cut of the question compiler, never sent: the system gets one frame at a time); this is' + (i - lo + 1) + ' of ' + (hi - lo)],
    ['Task', taskText(shape, i)],
    ['Base', p ? 'at ' + xyz(p.slice(0, 3)) + ' m, heading ' + num(p[3], 1) + '° (counter-clockwise from +x)' : null],
    ['Camera', p ? 'at ' + xyz(p.slice(4, 7)) + ' m, pitch ' + num(p[7], 1) + '° (positive looks up)' : null],
    ['Status', status],
    ['Subgoal', s.subgoal ? s.subgoal[i] : null],
    ['Questions', marks.during[i] ? ['asked after this step: ', questionLinks(marks.during[i], qrows)] : null],
    ['End questions', i === shape.n - 1 && marks.end.length ? ['asked after this last step: ', questionLinks(marks.end, qrows)] : null],
  ].filter(it => it[1] !== null && it[1] !== undefined && it[1] !== '');
  return h('dl', {class: 'facts'}, items.map(it => [h('dt', null, it[0]), h('dd', null, it[1])]));
}

function renderScrub(c, s, qrows, depthDir) {
  const shape = streamShape(s), marks = streamQuestions(c, shape.n), last = shape.n - 1, sends = s.sends;
  const note = h('p', {class: 'note frames', hidden: true}, PICTURE_NOTE);
  const rgb = sends.includes('rgb') ? pictureSlot('camera image (rgb)', note) : null;
  const range = s.camera ? ': white ' + num(s.camera.near_m) + ' m, dark grey ' + num(s.camera.far_m) + ' m, black no reading' : '';
  const depth = sends.includes('depth') && s.depth ? pictureSlot('depth image, drawn for viewing' + range, note) : null;
  const counter = h('span', {class: 'counter', 'data-role': 'counter'});
  const slider = h('input', {type: 'range', min: 0, max: last, step: 1, value: 0, 'aria-label': 'frame'});
  const pictures = h('div', {class: 'shots'}, rgb ? rgb.el : null, depth ? depth.el : null,
    sends.includes('depth') && !s.depth ? h('p', {class: 'muted'}, 'The depth picture is not drawn in this view.') : null);
  const facts = h('div', {class: 'finfo'}), dets = h('div', {class: 'fdets'});
  const strip = marksStrip(s, shape, marks, i => show(i));
  let at = 0;

  function show(i) {
    at = Math.min(last, Math.max(0, i));
    slider.value = String(at);
    counter.textContent = 'frame ' + (at + 1) + ' of ' + shape.n;
    if (rgb) rgb.set(s.rgb ? s.rgb + pad6(s.k0 + at) + '.jpg' : '');
    if (depth) depth.set(s.depth + pad6(s.k0 + at) + '.png');
    fill(facts, frameFacts(s, shape, marks, qrows, at));
    fill(dets, s.detections ? detectionsTable(s, at) : null);
    strip.setWindow(shape.windowOf[at] || 0);
  }
  const jump = dir => {   // to the first step of the next look window, or of this one when going back from its middle
    const b = shape.windowOf[at] || 0, lo = shape.bounds[b][0];
    show(dir > 0 ? (shape.bounds[b + 1] ? shape.bounds[b + 1][0] : last) : at > lo ? lo : (shape.bounds[b - 1] || [0])[0]);
  };
  const button = (act, label, title, fn) => h('button', {class: 'tool', type: 'button', 'data-act': act, title, on: {click: fn}}, label);
  const bar = h('div', {class: 'bar'}, button('prev-window', '« window', 'to the start of this look window, or of the one before (Page Up)', () => jump(-1)),
    button('prev', '‹ prev', 'the step before (left arrow)', () => show(at - 1)), counter,
    button('next', 'next ›', 'the next step (right arrow)', () => show(at + 1)),
    button('next-window', 'window »', 'to the start of the next look window (Page Down)', () => jump(1)));
  slider.addEventListener('input', () => show(Number(slider.value)));
  const box = h('div', {class: 'scrub', tabindex: 0},
    h('div', {class: 'muted small'}, 'Channel ' + s.channel + ' delivers: ' + sends.join(', ') + '. With the focus here, the arrow keys step, Home and End go to the ends, Page Up and Page Down by look window.'),
    s.notes.map(x => h('div', {class: 'note'}, x)),
    s.missing.rgb ? h('div', {class: 'note'}, s.missing.rgb + ' of the ' + shape.n + ' camera images are not in the dataset.') : null,
    s.missing.depth ? h('div', {class: 'note'}, s.missing.depth + ' of the ' + shape.n + ' depth images are not in the dataset or cannot be drawn.') : null,
    depth ? h('div', {class: 'muted small'}, 'The depth images of the dataset are 16-bit and look black in a browser: stbench view drew these grey ones from their values, in ' + (depthDir || 'a folder beside this page') + '.') : null,
    note, bar, slider, strip.el,
    h('div', {class: 'muted small legend'}, 'Under the slider: a thin line is where a look window ends (a cut of the question compiler, up to 8 steps; the system gets one frame at a time), a blue line T1 where a task begins, Q3 where a question was asked after that step (click to go there).'),
    pictures, facts, dets);
  box.addEventListener('keydown', ev => {
    if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
    const go = {ArrowLeft: () => show(at - 1), ArrowRight: () => show(at + 1), Home: () => show(0), End: () => show(last), PageUp: () => jump(-1), PageDown: () => jump(1)}[ev.key];
    if (!go) return;
    ev.preventDefault();
    go();
  });
  show(0);
  return box;
}

// The section of a case, closed until opened; its frames are built the first time.
function streamBox(c, qrows, depthDir) {
  const s = c.stream ? DATA.streams[c.stream] : null;
  const box = h('details', {class: 'scrubbox'}, h('summary', null, 'The stream, frame by frame' + (s ? ' (' + s.t.length + ' frames)' : '')));
  box.addEventListener('toggle', () => {
    if (!box.open || box.dataset.built) return;
    box.dataset.built = '1';
    box.appendChild(s ? renderScrub(c, s, qrows, depthDir) : h('p', {class: 'muted'}, 'The stream of this case cannot be shown: see the notes above.'));
  });
  return box;
}
