// Inspector: renders a run at simulated time t from recorded events, belief snapshots
// and camera frames. Live view and replay use the same drawing code.
'use strict';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const S = { run: null, events: [], beliefs: [], frames: [], views: [], eyes: [], t: 0, end: 0, playing: false, speed: 2, live: false, follow: false, lastWall: 0 };

const VIEW = { overhead_a: 'A の真上のカメラ', side_south: '手前側', side_north: '奥側', side_west: '左側', side_east: '右側' };
const DIR = { south: '手前側', north: '奥側', west: '左側', east: '右側' };
const label = (id) => (id && id.startsWith('case_') ? `ケース ${parseInt(id.slice(5), 10)}` : id || '');
const list = (ids) => (ids || []).map(label).join('・');
const pct = (p) => `${Math.round(p * 100)}%`;
const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

// The four ways a mission ends (requirements v04), and older runs.
const STATUS = {running: '実行中', verified: '全件移載・検証済み', verified_with_handover: '移載済み・人への引き渡しあり',
  unresolved: '未解決あり', interrupted: '中断', complete: '完了', incomplete: '途中終了'};

// Each event becomes one sentence for people. Tone drives the caption colour.
function describe(e) {
  const d = e.data || {};
  switch (e.type) {
    case 'MISSION_ACCEPTED': return ['指示を受けた：パレット A の荷を B へ積み替える', 'plan'];
    case 'OBSERVED':
      if (d.view === 'under') return [d.measured ? `下のカメラで持ったケースを測った（高さ ${(d.height_m * 100).toFixed(0)} cm）` : '下のカメラで測れなかった：上からの推定で置く', d.measured ? 'plan' : 'look'];
      return [`${VIEW[d.view] || d.view}で見た（ケース ${d.cases} 個を把握）`, 'plan'];
    case 'OBSERVE_REQUESTED': return [`${DIR[d.view] || d.view}へ見に行く：${d.reason}`, 'look'];
    case 'PICK_PLANNED': {
      const fall = d.predicted_fall && d.predicted_fall.length ? `。${list(d.predicted_fall)} が動く見込み` : '';
      const fb = d.fallback ? '（ほかに安全な候補がない）' : '';
      const st = d.state ? `（${d.state}）` : '';
      return [`${label(d.case)}${st} を取る：倒れる見込み ${d.risk}${fall}${fb}`, 'plan'];
    }
    case 'GRIP_OK': return [`${label(d.case)} を吸着した`, 'plan'];
    case 'GRIP_FAILED': return [`${label(d.case)} を吸着できなかった（${d.attempt} 回目）`, 'look'];
    case 'LIFTED': {
      if (d.upright) return [`${label(d.case)} を持ち上げた → B へ`, 'plan'];
      const pose = d.up_dim > 0.33 ? '立っている' : '横倒し';
      return d.destination === 'reorient'
        ? [`${label(d.case)} は${pose}（高さ ${(d.up_dim * 100).toFixed(0)} cm）→ 起こしてから B へ`, 'plan']
        : [`${label(d.case)} は${pose} → 別置きへ（人が起こす）`, 'plan'];
    }
    case 'LOAD_MEASURED': return [`試し持ち上げで量った：${d.mass_kg.toFixed(1)} kg・重心は中心から ${Math.hypot(...d.com_offset_mm).toFixed(0)} mm`, 'plan'];
    case 'REORIENTED': return [`${label(d.case)} を台の上で底の縁を支点に倒して起こした（高さ ${(d.standing_m * 100).toFixed(0)} cm を確認）`, 'done'];
    case 'FORK_IN': return d.accepted
      ? [`底受けの爪を差し込み、重さの ${Math.round(d.share * 100)} % を爪に移した（左 ${Math.round(d.tines_N[0])} N・右 ${Math.round(d.tines_N[1])} N）`, 'plan']
      : [`爪に重さを移せなかった：${d.reason}`, 'look'];
    case 'NEAR_MISS': return [`吸着板の進路にほかの物がある（${d.clearance_mm == null ? '見えない' : d.clearance_mm + ' mm 先'}）→ 触れる前に止めた`, 'stop'];
    case 'VACUUM_LOST': return [`吸着が外れたが爪が支えている → 止めて吸い直し${d.sealed_again ? '、残りはゆっくり運ぶ' : 'できず、人を待つ'}`, 'look'];
    case 'CONTACT_LOAD': return [d.calibrated ? `接触の力：${d.kind === 'touch' ? '吸着板が天面に触れた' : '爪が下の面に着いた'}（${d.static_N} N・上限 ${d.limit_static_N} N）` : '接触の力：上下軸の読みが未校正のため N では判定しない', d.damage_risk ? 'stop' : 'done'];
    case 'DAMAGE_RISK': return [`${label(d.case)} に上限を超える力がかかった（${d.static_N} N）→ 損傷のおそれ。人が確認`, 'stop'];
    case 'FORK_SKIPPED': return [`爪を使わない：${d.reason}`, 'look'];
    case 'FORK_LANDED': return [`爪が着地したのを感じた → 重さを吸着に戻す`, 'plan'];
    case 'FORK_OUT': return [d.in_air ? `爪を空中で抜いた：${d.reason}` : '爪を空いている側へ抜いた → 吸着だけで静かに置く', 'plan'];
    case 'PLACEMENT_SETTLED': return [d.ok ? `手を離して ${d.waited_s} 秒後も動いていない（${d.drift_mm} mm）` : `手を離した後に動いた：${d.reason || `${d.drift_mm} mm・${d.lean_deg}°`}`, d.ok ? 'done' : 'look'];
    case 'PLACED': return [`${label(d.case)} を B の ${d.layer + 1} 段目に置いた`, 'done'];
    case 'PLACEMENT_CHECKED': {
      if (d.ok) return [`置いた位置と向きを確かめた（ずれ ${(d.position_error_m * 100).toFixed(1)} cm）`, 'done'];
      if (d.reason) return [`置いた位置を確かめられない：${d.reason}`, 'look'];
      return [`置いたケースがずれている（位置 ${(d.position_error_m * 100).toFixed(1)} cm・向き ${d.yaw_error_deg}°）`, 'look'];
    }
    case 'PLACEMENT_REPAIRED': return ['ずれたケースをつかみ直して、同じ区画に置き直す', 'plan'];
    case 'REGRIPPED': return [`${label(d.case)} は中心から ${Math.round(d.offset_mm)} mm ずれて吸着していた → 台に置いて持ち直す`, 'look'];
    case 'PILE_CHANGED': {
      if (!d.moved || !d.moved.length) return ['取った後に山を確かめた：ほかのケースは動いていない', 'done'];
      const un = d.unexpected && d.unexpected.length ? `（予想外：${list(d.unexpected)}）` : '';
      return [`取った後に ${list(d.moved)} が動いた${un}。見立てを直す`, 'look'];
    }
    case 'SAFETY_STOP': return [`停止：${d.reason}`, 'stop'];
    case 'SAFETY_CLEARED': return [`${d.operator}が安全を確かめ、再開した`, 'plan'];
    case 'MOVE_REJECTED': return [`動作を取りやめた：${d.reason}`, 'look'];
    case 'HANDOVER': return [`${label(d.case)} を人に渡す：${d.reason}`, 'look'];
    case 'DROP_DETECTED': return [`搬送中に ${label(d.case)} を落とした`, 'stop'];
    case 'DECISION_DEFERRED': return [`判断を見送った：${d.reason}`, 'look'];
    case 'MISSION_COMPLETE': return [`完了：B に ${d.placed} 個、人に渡した ${d.handed_over} 個`, 'done'];
    case 'MISSION_INCOMPLETE': return [`途中で終了：${d.reason}`, 'stop'];
    default: return [null, null];
  }
}

const atOrBefore = (arr, t, key) => {
  let lo = 0, hi = arr.length - 1, best = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (key(arr[m]) <= t) { best = m; lo = m + 1; } else hi = m - 1; }
  return best;
};

async function getJSON(url) { const r = await fetch(url, { cache: 'no-store' }); return r.ok ? r.json() : null; }

async function load(run) {
  S.run = run;
  let eyes;
  [S.events, S.beliefs, S.frames, S.views, eyes] = await Promise.all([
    getJSON(`/api/run/${run}/events`), getJSON(`/api/run/${run}/beliefs`), getJSON(`/api/run/${run}/frames`),
    getJSON(`/api/run/${run}/views`), getJSON(`/api/run/${run}/perception`)]);
  S.events ||= []; S.beliefs ||= []; S.frames ||= []; S.views ||= [];
  S.eyes = (eyes && eyes.views) || [];
  const result = await getJSON(`/api/run/${run}/result`);
  S.live = !result;
  S.follow = S.live;
  S.end = Math.max(S.events.length ? S.events[S.events.length - 1].t : 0, S.frames.length ? S.frames[S.frames.length - 1] / 1000 : 0);
  showMode();
  const acc = S.events.find((e) => e.type === 'MISSION_ACCEPTED');
  const POLICY = { shinai: '必要なときだけ見に行く', observe_all: '毎回すべての方向から見る', top_first: '上から順に取る' };
  $('mission').textContent = acc ? `${acc.data.instruction}（方式：${POLICY[acc.data.policy] || acc.data.policy}）` : run;
}

// While a run is still going, the page either follows its newest moment or replays what is
// recorded so far; pressing play leaves the newest moment, and reaching it follows again.
function showMode() {
  $('mode').textContent = !S.live ? '再生' : S.follow ? 'ライブ（いま撮れた最新の場面）' : '実行中の記録を再生（最後まで行くとライブに戻ります）';
}

async function poll() {
  if (!S.live || !S.run) return;
  const lastSeq = S.events.length ? S.events[S.events.length - 1].seq : -1;
  const lastB = S.beliefs.length ? S.beliefs[S.beliefs.length - 1].seq : -1;
  const [ev, be, fr, res] = await Promise.all([getJSON(`/api/run/${S.run}/events?since=${lastSeq}`),
    getJSON(`/api/run/${S.run}/beliefs?since=${lastB}`), getJSON(`/api/run/${S.run}/frames`), getJSON(`/api/run/${S.run}/result`)]);
  if (ev) S.events.push(...ev);
  if (be) S.beliefs.push(...be);
  if (fr) S.frames = fr;
  S.end = Math.max(S.end, S.events.length ? S.events[S.events.length - 1].t : 0, S.frames.length ? S.frames[S.frames.length - 1] / 1000 : 0);
  if (res) S.live = S.follow = false;
  if (S.follow) S.t = S.end;
  showMode();
  if (S.follow || res) render();
}

function render() {
  const t = S.t;
  $('clock').textContent = mmss(t);
  $('seek').value = S.end ? Math.round((t / S.end) * 1000) : 0;
  // frame
  const fi = atOrBefore(S.frames, t * 1000, (x) => x);
  if (fi >= 0) {
    const name = `${String(S.frames[fi]).padStart(8, '0')}.jpg`;
    const src = `/api/run/${S.run}/frame/${name}`;
    if ($('frame').dataset.src !== src) { $('frame').src = src; $('frame').dataset.src = src; }
    $('frame-empty').hidden = true;
    for (const v of ['side', 'closeup']) {
      const fig = $(`inset-${v}`), img = fig.querySelector('img');
      fig.hidden = !S.views.includes(v);
      if (fig.hidden) continue;
      const vsrc = `/api/run/${S.run}/frame/${v}/${name}`;
      if (img.dataset.src !== vsrc) { img.src = vsrc; img.dataset.src = vsrc; }
    }
  } else { $('frame-empty').hidden = false; }
  renderEye(t);
  // events up to t
  const ei = atOrBefore(S.events, t, (e) => e.t);
  const past = S.events.slice(0, ei + 1);
  let safety = 'ok';
  for (const e of past) { if (e.type === 'SAFETY_STOP') safety = 'stop'; if (e.type === 'SAFETY_CLEARED') safety = 'ok'; }
  $('safety').dataset.state = safety;
  $('safety').textContent = safety === 'stop' ? '安全：停止中' : '安全：異常なし';
  const lines = [];
  for (let i = past.length - 1; i >= 0 && lines.length < 7; i--) {
    const [text, tone] = describe(past[i]);
    if (text) lines.push({ t: past[i].t, text, tone });
  }
  $('now').textContent = lines.length ? lines[0].text : '待機中';
  $('now').dataset.tone = lines.length ? lines[0].tone : 'plan';
  $('timeline').replaceChildren(...lines.map((l) => {
    const li = document.createElement('li'); li.dataset.tone = l.tone;
    const tm = document.createElement('time'); tm.textContent = mmss(l.t);
    const sp = document.createElement('span'); sp.textContent = l.text;
    li.append(tm, sp); return li;
  }));
  // belief
  const bi = atOrBefore(S.beliefs, t, (b) => b.t);
  drawBelief(bi >= 0 ? S.beliefs[bi] : null);
}

// The robot's own camera at the latest capture up to t, with what it made of it (scripts/perception_views.py).
const CAMERA = { overhead_a: 'パレット A の真上', overhead_b: 'パレット B の真上', tool: '手先', under: '床（下から）', down: '吸着板の中央（下向き）' };
function renderEye(t) {
  $('eye-panel').hidden = !S.eyes.length;
  if (!S.eyes.length) return;
  const i = atOrBefore(S.eyes, t, (v) => v.t);
  if (i < 0) { $('eye-note').textContent = 'まだ何も撮っていません。'; return; }
  const v = S.eyes[i];
  const src = `/api/run/${S.run}/perception/${v.picture}`;
  if ($('eye').dataset.src !== src) { $('eye').src = src; $('eye').dataset.src = src; }
  $('eye-camera').textContent = `${CAMERA[v.camera] || v.camera}のカメラ・${mmss(v.t)}`;
  const c = v.counts || {};
  $('eye-note').textContent = `安全に取れる ${c.safe || 0} 個・倒れる見込み ${c.risky || 0} 個・今は取れない ${c.blocked || 0} 個`
    + (v.chosen ? `。次に取るのは ${v.chosen.replace('case_', 'ケース ')}。` : '。');
}

function drawBelief(b) {
  const svg = $('map');
  if (!b) { svg.replaceChildren(); $('unknown').textContent = '—'; return; }
  const [ax, ay, aw, ah] = b.pallets.A;
  const x0 = ax - aw / 2 - 0.55, x1 = ax + aw / 2 + 0.55, y0 = ay - ah / 2 - 0.4, y1 = ay + ah / 2 + 0.4;
  const W = 400, H = 230, k = Math.min(W / (x1 - x0), H / (y1 - y0));
  const ox = (W - (x1 - x0) * k) / 2, oy = (H - (y1 - y0) * k) / 2;
  const X = (x) => ox + (x - x0) * k, Y = (y) => H - (oy + (y - y0) * k);
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a) => { const e = document.createElementNS(NS, n); for (const [kk, v] of Object.entries(a)) e.setAttribute(kk, v); return e; };
  const kids = [];
  const defs = el('defs', {});
  const pat = el('pattern', { id: 'hatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' });
  pat.append(el('rect', { width: 5, height: 5, fill: '#2E2413' }), el('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: '#E9B463', 'stroke-width': 1.2 }));
  defs.append(pat); kids.push(defs);
  kids.push(el('rect', { x: X(ax - aw / 2), y: Y(ay + ah / 2), width: aw * k, height: ah * k, fill: 'none', stroke: '#5C5C5C', 'stroke-dasharray': '4 3' }));
  const c = b.unknown_cell;
  for (const [ux, uy] of b.unknown_cells) {
    if (ux < x0 || ux > x1 || uy < y0 || uy > y1) continue;
    kids.push(el('rect', { x: X(ux), y: Y(uy + c), width: c * k + 0.5, height: c * k + 0.5, fill: 'url(#hatch)', opacity: 0.85 }));
  }
  const style = { next: ['#8FAFD9', '#8FAFD9', ''], safe: ['none', '#7FBB9C', ''], falls: ['none', '#D9847E', '5 3'],
                  risky: ['none', '#E9B463', '2 2'], blocked: ['#0E0E0E', '#5C5C5C', ''] };
  const sorted = [...b.boxes].sort((p, q) => p.top - q.top);
  for (const bx of sorted) {
    const [fill, stroke, dash] = style[bx.state] || style.blocked;
    const pts = bx.poly.map(([px, py]) => `${X(px).toFixed(1)},${Y(py).toFixed(1)}`).join(' ');
    const shade = 0.35 + Math.min(1, bx.top / 1.0) * 0.5;
    kids.push(el('polygon', { points: pts, fill: fill === 'none' ? `rgba(239,239,238,${(shade * 0.18).toFixed(2)})` : fill,
                              'fill-opacity': bx.state === 'next' ? 0.85 : 1, stroke, 'stroke-width': 1.5, 'stroke-dasharray': dash }));
    if (bx.state === 'next' || bx.state === 'falls') {
      const cx = bx.poly.reduce((s, p) => s + p[0], 0) / 4, cy = bx.poly.reduce((s, p) => s + p[1], 0) / 4;
      const tx = el('text', { x: X(cx), y: Y(cy) + 4, 'text-anchor': 'middle', 'font-size': 12, fill: bx.state === 'next' ? '#0E0E0E' : '#D9847E' });
      tx.textContent = bx.label.replace('ケース ', ''); kids.push(tx);
    }
  }
  // What holds each case, and where its mass may sit (the box centre, give or take what its
  // contents allow) for the case about to be taken and for any case whose footing is uncertain.
  const byId = Object.fromEntries(b.boxes.map((x) => [x.id, x]));
  for (const bx of sorted) {
    for (const s of bx.rests_on || []) {
      const o = byId[s];
      if (!o || !o.com || !bx.com) continue;
      kids.push(el('line', { x1: X(bx.com[0]), y1: Y(bx.com[1]), x2: X(o.com[0]), y2: Y(o.com[1]), stroke: '#8A8A88', 'stroke-width': 1, opacity: 0.7 }));
    }
  }
  for (const bx of sorted) {
    if (!bx.com || !(bx.state === 'next' || bx.stability === 'uncertain')) continue;
    const m = bx.com_margin || 0, [cx, cy] = bx.com;
    const col = bx.stability === 'uncertain' ? '#E9B463' : '#0E0E0E';
    if (m > 0) kids.push(el('rect', { x: X(cx - m), y: Y(cy + m), width: 2 * m * k, height: 2 * m * k, fill: 'none', stroke: col, 'stroke-width': 1.2, 'stroke-dasharray': '3 2' }));
    kids.push(el('line', { x1: X(cx) - 4, y1: Y(cy), x2: X(cx) + 4, y2: Y(cy), stroke: col, 'stroke-width': 1.5 }));
    kids.push(el('line', { x1: X(cx), y1: Y(cy) - 4, x2: X(cx), y2: Y(cy) + 4, stroke: col, 'stroke-width': 1.5 }));
  }
  svg.replaceChildren(...kids);
  $('unknown').textContent = b.unknown_litres == null ? '—' : `${Math.round(b.unknown_litres)} L`;
  $('hidden').textContent = b.next_hidden == null ? '—' : pct(b.next_hidden);
  $('lowconf').textContent = `${b.boxes.filter((x) => x.confidence < 0.8).length} 個`;
  const nxt = b.boxes.find((x) => x.state === 'next');
  const onTop = nxt && nxt.reason === 'free' ? '上に載っているケースはなく、' : '';
  $('why').textContent = nxt ? `${nxt.label}を選んだ理由：${onTop}倒れる見込み ${nxt.risk}、位置の確からしさ ${pct(nxt.confidence)}。` : '';
  if (b.progress) {
    $('placed').textContent = `${b.progress.placed} / ${b.progress.slots}`;
    $('handed').textContent = `${b.progress.handed_over} 個`;
    $('fill').style.width = `${(100 * b.progress.placed) / b.progress.slots}%`;
  }
}

function tick(now) {
  if (S.playing) {
    const dt = (now - S.lastWall) / 1000;
    S.t = Math.min(S.end, S.t + dt * S.speed);
    if (S.t >= S.end) { S.playing = false; S.follow = S.live; $('play').textContent = '再生'; showMode(); }
    render();
  }
  S.lastWall = now;
  requestAnimationFrame(tick);
}

async function init() {
  const runs = (await getJSON('/api/runs')) || [];
  $('run').replaceChildren(...runs.map((r) => { const o = document.createElement('option'); o.value = r.name; o.textContent = `${r.name}（${STATUS[r.status] || '途中終了'}）`; return o; }));
  const want = params.get('run') || (runs.length ? runs[runs.length - 1].name : null);
  if (!want) return;
  $('run').value = want;
  await load(want);
  if (params.has('capture')) {               // deterministic stills for the video, driven by window.__seek
    document.body.classList.add('capture');
    S.speed = Number(params.get('speed') || 2);
    $('mode').textContent = S.speed === 1 ? '記録の再生（等倍）' : `記録の再生（${S.speed} 倍速）`;
    const loaded = (img) => new Promise((r) => { if (img.complete) r(); else img.onload = img.onerror = r; });
    const imageReady = () => Promise.all([$('frame'), ...document.querySelectorAll('.inset:not([hidden]) img')].map(loaded));
    window.__seek = async (t) => { S.t = t; render(); await imageReady(); };
    window.__end = S.end;
    window.__inspectorReady = true;
    return;
  }
  render();
  $('run').onchange = async () => { await load($('run').value); S.t = 0; render(); };
  $('play').onclick = () => {
    S.playing = !S.playing;
    if (S.playing && S.follow) { S.follow = false; S.t = 0; }      // following sits at the end: replay from the start
    $('play').textContent = S.playing ? '一時停止' : '再生';
    showMode();
  };
  $('speed').onchange = () => { S.speed = Number($('speed').value); $('speednote').textContent = S.speed === 1 ? '等倍で再生しています' : `${S.speed} 倍速で再生しています`; };
  $('seek').oninput = () => { S.follow = false; S.t = (Number($('seek').value) / 1000) * S.end; showMode(); render(); };
  setInterval(poll, 1000);
  requestAnimationFrame(tick);
}

init();
