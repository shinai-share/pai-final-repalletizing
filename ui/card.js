// Text cards for the video (title, "skipped" notice, results). Content comes from the URL.
'use strict';
const q = new URLSearchParams(location.search);
document.getElementById('kicker').textContent = q.get('kicker') || '';
document.getElementById('title').textContent = q.get('title') || '';
document.getElementById('note').textContent = q.get('note') || '';
document.getElementById('lines').replaceChildren(...q.getAll('line').map((t) => {
  const li = document.createElement('li'); li.textContent = t; return li;
}));
window.__inspectorReady = true;
