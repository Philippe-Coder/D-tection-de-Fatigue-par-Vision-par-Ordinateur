// Fatigue Guard pour navigateur : lit l'état de l'application locale et
// l'affiche sur l'icône. Aucune image ne transite ici, seulement des chiffres.
'use strict';

const DEFAULTS = { port: 8765, notifications: 'alertes' };
const FAST_MS = 2000;          // rythme de lecture quand l'application répond
const ALARM = 'fatigue-guard-poll';
const COLORS = { ok: '#34D399', warn: '#FBBF24', danger: '#FB5C74', idle: '#8B98AD', info: '#5B8DEF' };

let lastStatus = null;         // dernier état connu (null = application injoignable)
let loopTimer = null;

async function settings() {
  return { ...DEFAULTS, ...(await chrome.storage.local.get(DEFAULTS)) };
}

async function call(method, path) {
  const { port } = await settings();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 1500);
  try {
    const response = await fetch(`http://127.0.0.1:${port}${path}`, {
      method,
      headers: method === 'POST' ? { 'X-Fatigue-Guard': '1' } : {},
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

function badge(text, color) {
  chrome.action.setBadgeText({ text });
  if (color) chrome.action.setBadgeBackgroundColor({ color });
  if (chrome.action.setBadgeTextColor) chrome.action.setBadgeTextColor({ color: '#0B0F17' });
}

function render(status) {
  if (!status) {
    badge('', null);
    chrome.action.setTitle({ title: "Fatigue Guard — l'application n'est pas lancée" });
    return;
  }
  if (!status.running) {
    badge('II', COLORS.idle);
    chrome.action.setTitle({ title: 'Fatigue Guard — surveillance en pause' });
    return;
  }
  const score = Math.round(status.vigilance);
  if (status.state === 'CALIBRATION') badge('…', COLORS.info);
  else if (status.state === 'ABSENT') badge('–', COLORS.idle);
  else {
    const alert = status.state === 'SOMNOLENT';
    const color = alert || score < 40 ? COLORS.danger : score < 70 || status.state !== 'ATTENTIF' ? COLORS.warn : COLORS.ok;
    badge(alert ? '!' : String(score), color);
  }
  chrome.action.setTitle({ title: `Fatigue Guard — ${status.state_label}, vigilance ${score} %` });
}

function wanted(level, mode) {
  if (mode === 'aucune') return false;
  if (mode === 'alarmes') return level === 'alarm';
  if (mode === 'alertes') return level !== 'info';
  return true;
}

async function notify(alerts) {
  if (!Array.isArray(alerts) || alerts.length === 0) return;
  const newest = alerts[alerts.length - 1].id;
  // Le service worker peut être relancé : on mémorise le dernier identifiant vu.
  const { lastAlertId = null } = await chrome.storage.session.get('lastAlertId');
  await chrome.storage.session.set({ lastAlertId: newest });
  if (lastAlertId === null || newest < lastAlertId) return; // première lecture ou appli redémarrée
  const { notifications } = await settings();
  for (const alert of alerts) {
    if (alert.id <= lastAlertId || !wanted(alert.level, notifications)) continue;
    chrome.notifications.create(`fatigue-guard-${alert.id}`, {
      type: 'basic',
      iconUrl: chrome.runtime.getURL('icons/icon128.png'),
      title: alert.level === 'alarm' ? 'Fatigue Guard — ALERTE' : 'Fatigue Guard',
      message: alert.message,
      priority: alert.level === 'alarm' ? 2 : 0,
    });
  }
}

async function poll() {
  try {
    const status = await call('GET', '/status');
    if (status.app !== 'fatigue-guard') throw new Error('autre service');
    lastStatus = status;
    render(status);
    if (status.running) await notify(status.recent_alerts);
  } catch (err) {
    lastStatus = null;
    render(null);
  }
  return lastStatus;
}

// Tant que l'application répond, on relit toutes les 2 s ; chaque appel à l'API
// du navigateur garde le service worker éveillé. Sinon on laisse l'alarme
// (toutes les 30 s) revérifier, pour ne rien consommer inutilement.
async function loop() {
  clearTimeout(loopTimer);
  const status = await poll();
  if (status) loopTimer = setTimeout(loop, FAST_MS);
}

function schedule() {
  chrome.alarms.create(ALARM, { periodInMinutes: 0.5 });
  loop();
}

chrome.runtime.onInstalled.addListener(schedule);
chrome.runtime.onStartup.addListener(schedule);
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === ALARM) loop();
});

chrome.runtime.onMessage.addListener((message, _sender, reply) => {
  (async () => {
    if (message.type === 'status') {
      const status = await poll();
      if (status) loop();
      reply({ status, settings: await settings() });
    } else if (message.type === 'command') {
      try {
        await call('POST', `/${message.name}`);
        reply({ ok: true });
      } catch (err) {
        reply({ ok: false });
      }
      setTimeout(loop, 400);
    } else if (message.type === 'settings') {
      await chrome.storage.local.set(message.values);
      reply({ ok: true });
      loop();
    }
  })();
  return true; // réponse asynchrone
});

schedule();
