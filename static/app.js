import WaveSurfer from '/static/vendor/wavesurfer.esm.js';
import RegionsPlugin from '/static/vendor/regions.esm.js';

const $ = (sel) => document.querySelector(sel);

const el = {
  urlInput: $('#url-input'), loadBtn: $('#load-btn'), status: $('#status'),
  workspace: $('#workspace'), video: $('#video'),
  videoTitle: $('#video-title'), videoInfo: $('#video-info'),
  waveform: $('#waveform'),
  playBtn: $('#play-btn'), playRegionBtn: $('#play-region-btn'), loopToggle: $('#loop-toggle'),
  timeReadout: $('#time-readout'),
  regionStart: $('#region-start'), regionEnd: $('#region-end'), regionDur: $('#region-dur'),
  setStartBtn: $('#set-start-btn'), setEndBtn: $('#set-end-btn'),
  extractBtn: $('#extract-btn'), extractStatus: $('#extract-status'),
  labelPanel: $('#label-panel'), clipIdLabel: $('#clip-id-label'), qcLine: $('#qc-line'),
  transcript: $('#transcript'), asrInfo: $('#asr-info'),
  genderRadios: $('#gender-radios'), genderInfo: $('#gender-info'),
  emotionRadios: $('#emotion-radios'),
  valence: $('#valence'), valenceVal: $('#valence-val'),
  arousal: $('#arousal'), arousalVal: $('#arousal-val'),
  actor: $('#actor'), notes: $('#notes'), annotatorId: $('#annotator-id'),
  saveBtn: $('#save-btn'), discardBtn: $('#discard-btn'), saveStatus: $('#save-status'),
  clipList: $('#clip-list'),
};

const EXTRACT_TEXT = '구간 추출 및 사전 라벨';
const OVERWRITE_TEXT = '기존 구간을 덮어쓸까요?';
const DEFAULT_NOTES = 'CONF=3; QUALITY=A; AMBIG=0; SECONDARY=; FLAGS=';
const REGION_COLOR = 'rgba(36, 84, 125, 0.16)';

const state = {
  config: null,
  emotionLabel: {},   // value -> "English(한국어)" display text
  genderLabel: {},
  video: null,        // metadata from POST /api/videos
  ws: null,           // wavesurfer instance
  regions: null,      // regions plugin
  region: null,       // the single active region
  loop: false,
  clipId: null,       // current extracted clip
  pendingOverwrite: false,
  busy: false,
};

/* ---------- helpers ---------- */

function setStatus(node, text, cls = '') {
  node.textContent = text;
  node.className = `status ${cls}`.trim();
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join('; ');
    const err = new Error(detail || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

const fmt = (s) => s.toFixed(2);
const frameDur = () => 1 / (state.video?.fps || 30);

/* ---------- form rendering (from /api/config) ---------- */

function renderRadios(container, name, options) {
  container.innerHTML = '';
  for (const opt of options) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = name;
    input.value = opt.value;
    label.append(input, document.createTextNode(opt.label));
    container.append(label);
  }
}

function getRadio(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value || null;
}

function setRadio(name, value) {
  for (const input of document.querySelectorAll(`input[name="${name}"]`)) {
    input.checked = input.value === value;
  }
}

/* ---------- video load ---------- */

async function loadVideo() {
  const url = el.urlInput.value.trim();
  if (!url || state.busy) return;
  state.busy = true;
  el.loadBtn.disabled = true;
  setStatus(el.status, '영상을 불러오는 중…', 'busy');
  try {
    const meta = await api('/api/videos', { method: 'POST', body: JSON.stringify({ url }) });
    state.video = meta;
    setStatus(el.status, `${meta.video_id} 불러오기 완료`, 'ok');
    el.workspace.classList.remove('hidden');
    el.videoTitle.textContent = meta.title || meta.video_id;
    el.videoInfo.textContent =
      `${meta.channel || ''} · ${fmt(meta.duration_s)}s · ${meta.width}×${meta.height} @ ${meta.fps}fps`;
    el.video.src = meta.media_url;
    resetLabelPanel();
    renderClipList(meta.clips);
    setStatus(el.status, '파형을 만드는 중…', 'busy');
    const peaks = await api(`/api/videos/${meta.video_id}/peaks`);
    buildWaveform(peaks);
    setStatus(el.status, '준비 완료 — 파형을 드래그해 발화 구간을 선택하세요', 'ok');
  } catch (e) {
    setStatus(el.status, e.message, 'error');
  } finally {
    state.busy = false;
    el.loadBtn.disabled = false;
  }
}

function buildWaveform(peaksData) {
  if (state.ws) { state.ws.destroy(); state.ws = null; }
  state.region = null;
  updateRegionReadout();

  const ws = WaveSurfer.create({
    container: el.waveform,
    media: el.video,
    peaks: [peaksData.peaks],
    duration: peaksData.duration_s,
    height: 110,
    waveColor: '#9ba2a6',
    progressColor: '#24547d',
    cursorColor: '#20201d',
    cursorWidth: 1,
  });
  const regions = ws.registerPlugin(RegionsPlugin.create());
  regions.enableDragSelection({ color: REGION_COLOR });

  regions.on('region-created', (region) => {
    for (const r of regions.getRegions()) if (r !== region) r.remove();
    state.region = region;
    updateRegionReadout();
  });
  regions.on('region-updated', (region) => {
    state.region = region;
    updateRegionReadout();
  });
  regions.on('region-out', (region) => {
    if (state.loop && region === state.region) region.play();
  });

  ws.on('timeupdate', (t) => {
    el.timeReadout.textContent = `${fmt(t)} / ${fmt(peaksData.duration_s)}`;
  });
  ws.on('play', () => { el.playBtn.textContent = '⏸ 일시정지'; });
  ws.on('pause', () => { el.playBtn.textContent = '▶ 재생'; });

  state.ws = ws;
  state.regions = regions;
}

/* ---------- region controls ---------- */

function updateRegionReadout() {
  const r = state.region;
  el.regionStart.textContent = r ? fmt(r.start) : '–';
  el.regionEnd.textContent = r ? fmt(r.end) : '–';
  el.regionDur.textContent = r ? fmt(r.end - r.start) : '–';
  el.extractBtn.disabled = !r || state.busy;
  state.pendingOverwrite = false;
  el.extractBtn.textContent = EXTRACT_TEXT;
}

function setRegionBounds(start, end) {
  if (!state.regions) return;
  const dur = state.video.duration_s;
  start = Math.max(0, Math.min(start, dur - 0.05));
  end = Math.max(start + 0.05, Math.min(end, dur));
  if (state.region) {
    state.region.setOptions({ start, end });
  } else {
    state.regions.addRegion({ start, end, color: REGION_COLOR });
  }
  state.region = state.regions.getRegions()[0];
  updateRegionReadout();
}

function nudge(spec) {
  if (!state.region) return;
  const [edge, amountStr] = spec.split(':');
  const amount = amountStr.endsWith('f')
    ? (amountStr.startsWith('-') ? -frameDur() : frameDur())
    : parseFloat(amountStr);
  const { start, end } = state.region;
  if (edge === 'start') setRegionBounds(start + amount, end);
  else setRegionBounds(start, end + amount);
}

/* ---------- extract + prelabel ---------- */

async function extractClip() {
  const r = state.region;
  if (!r || state.busy) return;
  state.busy = true;
  el.extractBtn.disabled = true;
  setStatus(el.extractStatus, '구간을 추출하는 중…', 'busy');
  try {
    const body = {
      start_s: Math.round(r.start * 1000) / 1000,
      end_s: Math.round(r.end * 1000) / 1000,
      overwrite: state.pendingOverwrite,
    };
    const res = await api(`/api/videos/${state.video.video_id}/extract`, {
      method: 'POST', body: JSON.stringify(body),
    });
    state.clipId = res.clip_id;
    state.pendingOverwrite = false;
    el.extractBtn.textContent = EXTRACT_TEXT;
    openLabelPanel(res.clip_id);
    setStatus(el.extractStatus, `구간 추출 완료 (${fmt(res.duration_s)}초) · 사전 라벨링 중…`, 'busy');
    await prelabel(res.clip_id);
    setStatus(el.extractStatus, '사전 라벨 완료 — 검토 후 저장하세요', 'ok');
  } catch (e) {
    if (e.status === 409) {
      state.pendingOverwrite = true;
      el.extractBtn.textContent = OVERWRITE_TEXT;
      setStatus(el.extractStatus, '이미 추출된 구간입니다. 한 번 더 누르면 덮어씁니다.', 'error');
    } else {
      setStatus(el.extractStatus, e.message, 'error');
    }
  } finally {
    state.busy = false;
    el.extractBtn.disabled = !state.region;
  }
}

async function prelabel(clipId) {
  setStatus(el.asrInfo, '', '');
  el.asrInfo.textContent = '음성을 전사하는 중… (첫 실행 시 약 2.8GB 모델 다운로드)';
  el.genderInfo.textContent = '';
  const pre = await api(`/api/clips/${clipId}/prelabel`, { method: 'POST', body: '{}' });
  if (clipId !== state.clipId) return; // user moved on to another clip meanwhile

  el.transcript.value = pre.asr.text || '';
  el.asrInfo.textContent = pre.asr.text
    ? `(${pre.asr.model.split('/').pop()})`
    : '(음성이 감지되지 않았습니다 — 비워두거나 들리는 대로 입력하세요)';

  const g = pre.gender;
  const gLabel = state.genderLabel[g.label] || g.label;
  const conf = g.probs ? Math.max(...Object.values(g.probs)) : null;
  const lowConf = conf !== null && conf < state.config.gender_confidence_threshold;
  const preselect = (!pre.asr.text || lowConf || g.label === 'unknown') ? 'unknown' : g.label;
  setRadio('gender', preselect);
  el.genderInfo.textContent =
    g.probs ? `(모델: ${gLabel} ${conf.toFixed(2)})`
      : g.median_f0_hz ? `(F0 추정: ${gLabel}, ${g.median_f0_hz} Hz)`
        : '(모델 추정값 없음)';
  renderQC(pre.qc);
  updateSaveEnabled();
}

function renderQC(qc) {
  el.qcLine.innerHTML = '';
  const add = (cls, text) => {
    const s = document.createElement('span');
    s.className = cls;
    s.textContent = text;
    el.qcLine.append(s);
  };

  const f = qc?.face;
  if (f?.error) add('qc-dim', '얼굴 확인 불가');
  else if (f?.verdict === 'single') add('qc-ok', '✓ 얼굴 1명');
  else if (f?.verdict === 'none') add('qc-warn', '⚠ 얼굴 미검출');
  else if (f?.verdict === 'multiple') {
    const n = Math.max(...f.faces_per_frame);
    add('qc-warn', `⚠ 얼굴 최대 ${n}명`);
  }

  const s = qc?.speaker;
  if (s?.error) add('qc-dim', '화자 확인 불가');
  else if (s?.verdict === 'single') add('qc-ok', '✓ 화자 1명');
  else if (s?.verdict === 'no_speech') add('qc-warn', '⚠ 음성 없음');
  else if (s?.verdict === 'multiple') {
    add('qc-warn', `⚠ 화자 약 ${s.estimated_speakers}명`);
  }

  const v = qc?.vad;
  if (v && !v.error) {
    const pct = Math.round(v.speech_ratio * 100);
    add(pct >= 50 ? 'qc-ok' : 'qc-warn', `음성 비율 ${pct}%`);
  }

  el.qcLine.classList.toggle('hidden', !el.qcLine.childNodes.length);
}

/* ---------- label panel ---------- */

function openLabelPanel(clipId) {
  el.labelPanel.classList.remove('disabled');
  el.clipIdLabel.textContent = clipId;
  el.qcLine.classList.add('hidden');
  el.qcLine.innerHTML = '';
  el.transcript.value = '';
  el.asrInfo.textContent = '';
  el.genderInfo.textContent = '';
  setRadio('gender', 'unknown');
  setRadio('emotion', null);
  el.valence.value = state.config.valence_scale.neutral;
  el.arousal.value = state.config.arousal_scale.neutral;
  el.valenceVal.textContent = el.valence.value;
  el.arousalVal.textContent = el.arousal.value;
  el.actor.value = '';
  el.notes.value = DEFAULT_NOTES;
  setStatus(el.saveStatus, '');
  updateSaveEnabled();
}

function resetLabelPanel() {
  state.clipId = null;
  el.labelPanel.classList.add('disabled');
  el.clipIdLabel.textContent = '';
  setStatus(el.extractStatus, '');
}

function updateSaveEnabled() {
  el.saveBtn.disabled =
    !state.clipId || !getRadio('emotion') || !el.annotatorId.value.trim();
}

async function saveAnnotation() {
  if (el.saveBtn.disabled || state.busy) return;
  state.busy = true;
  setStatus(el.saveStatus, '저장하는 중…', 'busy');
  try {
    const body = {
      emotion: getRadio('emotion'),
      valence: parseInt(el.valence.value, 10),
      arousal: parseInt(el.arousal.value, 10),
      gender: getRadio('gender') || 'unknown',
      transcript: el.transcript.value,
      actor: el.actor.value,
      notes: el.notes.value,
      annotator_id: el.annotatorId.value.trim(),
    };
    localStorage.setItem('emolabel_annotator_id', body.annotator_id);
    await api(`/api/clips/${state.clipId}/annotation`, {
      method: 'POST', body: JSON.stringify(body),
    });
    setStatus(el.saveStatus, `${state.clipId} 저장 완료`, 'ok');
    setStatus(el.extractStatus, '저장했습니다. 다음 발화 구간을 선택하세요.', 'ok');
    resetLabelPanel();
    await refreshClipList();
  } catch (e) {
    setStatus(el.saveStatus, e.message, 'error');
  } finally {
    state.busy = false;
  }
}

async function discardClip() {
  if (!state.clipId) return;
  try {
    await api(`/api/clips/${state.clipId}`, { method: 'DELETE' });
    setStatus(el.extractStatus, '구간을 폐기했습니다', 'ok');
  } catch (e) {
    setStatus(el.extractStatus, e.message, 'error');
  }
  resetLabelPanel();
  await refreshClipList();
}

/* ---------- clip list ---------- */

function renderClipList(clips) {
  el.clipList.innerHTML = '';
  for (const c of clips) {
    const li = document.createElement('li');
    const range = document.createElement('span');
    range.className = 'range';
    range.textContent = `${fmt(c.start_s)} – ${fmt(c.end_s)}`;
    const tag = document.createElement('span');
    tag.className = c.annotated ? 'tag done' : 'tag';
    tag.textContent = c.annotated
      ? (state.emotionLabel[c.emotion] || c.emotion || '라벨 완료')
      : '미라벨';
    li.append(range, tag);
    li.addEventListener('click', () => {
      setRegionBounds(c.start_s, c.end_s);
      state.ws?.setTime(c.start_s);
    });
    el.clipList.append(li);
  }
  if (!clips.length) {
    const li = document.createElement('li');
    li.className = 'dim';
    li.textContent = '저장된 구간이 없습니다';
    el.clipList.append(li);
  }
}

async function refreshClipList() {
  if (!state.video) return;
  renderClipList(await api(`/api/clips?video_id=${state.video.video_id}`));
}

/* ---------- wiring ---------- */

async function init() {
  state.config = await api('/api/config');
  state.emotionLabel = Object.fromEntries(state.config.emotions.map((o) => [o.value, o.label]));
  state.genderLabel = Object.fromEntries(state.config.gender_options.map((o) => [o.value, o.label]));
  renderRadios(el.genderRadios, 'gender', state.config.gender_options);
  renderRadios(el.emotionRadios, 'emotion', state.config.emotions);
  el.valence.min = state.config.valence_scale.min;
  el.valence.max = state.config.valence_scale.max;
  el.arousal.min = state.config.arousal_scale.min;
  el.arousal.max = state.config.arousal_scale.max;
  el.annotatorId.value = localStorage.getItem('emolabel_annotator_id') || '';

  el.loadBtn.addEventListener('click', loadVideo);
  el.urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') loadVideo(); });

  el.playBtn.addEventListener('click', () => state.ws?.playPause());
  el.video.addEventListener('click', () => state.ws?.playPause());
  el.playRegionBtn.addEventListener('click', () => state.region?.play());
  el.loopToggle.addEventListener('change', () => { state.loop = el.loopToggle.checked; });

  el.setStartBtn.addEventListener('click', () => {
    if (state.ws) setRegionBounds(state.ws.getCurrentTime(), state.region?.end ?? state.ws.getCurrentTime() + 1);
  });
  el.setEndBtn.addEventListener('click', () => {
    if (state.ws && state.region) setRegionBounds(state.region.start, state.ws.getCurrentTime());
  });
  for (const btn of document.querySelectorAll('[data-nudge]')) {
    btn.addEventListener('click', () => nudge(btn.dataset.nudge));
  }

  el.extractBtn.addEventListener('click', extractClip);
  el.saveBtn.addEventListener('click', saveAnnotation);
  el.discardBtn.addEventListener('click', discardClip);
  el.emotionRadios.addEventListener('change', updateSaveEnabled);
  el.annotatorId.addEventListener('input', updateSaveEnabled);
  el.valence.addEventListener('input', () => { el.valenceVal.textContent = el.valence.value; });
  el.arousal.addEventListener('input', () => { el.arousalVal.textContent = el.arousal.value; });

  document.addEventListener('keydown', (e) => {
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.code === 'Space') { e.preventDefault(); state.ws?.playPause(); }
    else if (e.key === 'l' || e.key === 'L') { el.loopToggle.checked = !el.loopToggle.checked; state.loop = el.loopToggle.checked; }
    else if (e.key === '[') el.setStartBtn.click();
    else if (e.key === ']') el.setEndBtn.click();
  });
}

init().catch((e) => setStatus(el.status, `초기화 실패: ${e.message}`, 'error'));
