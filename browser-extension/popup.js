'use strict';

const $ = (id) => document.getElementById(id);
const COLORS = { ok: '#34D399', warn: '#FBBF24', danger: '#FB5C74', idle: '#8B98AD', info: '#5B8DEF' };
const STATE_COLORS = {
  ATTENTIF: COLORS.ok,
  DISTRAIT: COLORS.warn,
  FATIGUE: COLORS.warn,
  SOMNOLENT: COLORS.danger,
  ABSENT: COLORS.idle,
  CALIBRATION: COLORS.info,
};

let running = false;
let settingsLoaded = false;

function send(message) {
  return chrome.runtime.sendMessage(message).catch(() => null);
}

function duration(seconds) {
  const s = Math.floor(seconds || 0);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h) return `${h} h ${String(m).padStart(2, '0')}`;
  if (m) return `${m} min`;
  return `${s} s`;
}

function renderAlerts(alerts) {
  const list = $('alerts');
  list.textContent = '';
  const recent = (alerts || []).slice(-6).reverse();
  if (recent.length === 0) {
    const li = document.createElement('li');
    li.className = 'muted';
    li.textContent = 'Aucune alerte';
    list.append(li);
    return;
  }
  for (const alert of recent) {
    const li = document.createElement('li');
    li.className = alert.level;
    const time = document.createElement('time');
    time.textContent = new Date(alert.time * 1000).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
    li.append(time, alert.message); // texte brut : jamais interprété comme du HTML
    list.append(li);
  }
}

function render(status) {
  $('live').hidden = !status;
  $('offline').hidden = Boolean(status);
  if (!status) {
    $('profile').textContent = 'Détection de fatigue et de distraction';
    return;
  }
  running = Boolean(status.running);
  $('profile').textContent = status.profile_label || '';
  $('toggle').textContent = running ? 'Arrêter la surveillance' : 'Démarrer la surveillance';
  $('toggle').classList.toggle('running', running);
  $('recalibrate').disabled = !running;

  const measuring = running && status.state !== 'CALIBRATION' && status.state !== 'ABSENT';
  const score = Math.round(status.vigilance);
  const color = !running ? COLORS.idle
    : status.state !== 'ATTENTIF' ? STATE_COLORS[status.state] || COLORS.warn
    : score >= 70 ? COLORS.ok : score >= 40 ? COLORS.warn : COLORS.danger;

  $('arc').style.strokeDasharray = `${measuring ? (score * 0.75).toFixed(1) : 0} 100`;
  $('arc').style.opacity = measuring && score > 0 ? '1' : '0'; // sinon l'extrémité arrondie laisse un point
  $('arc').style.stroke = color;
  $('score').textContent = measuring ? String(score) : '—';
  $('state').textContent = running ? status.state_label : 'En pause';
  $('state').style.fill = color;

  if (!running) $('reason').textContent = "La surveillance n'est pas démarrée";
  else if (status.state === 'CALIBRATION') $('reason').textContent = `Regardez l'écran normalement… ${Math.round(status.calibration * 100)} %`;
  else if (status.state === 'ATTENTIF') $('reason').textContent = 'Tout va bien';
  else $('reason').textContent = status.reason || status.state_label;

  $('perclos').textContent = running ? `${Math.round(status.perclos * 100)} %` : '—';
  $('blinks').textContent = running && status.blink_rate ? String(Math.round(status.blink_rate)) : '—';
  $('yawns').textContent = running ? String(status.yawns) : '—';
  $('session').textContent = running ? duration(status.session_s) : '—';
  renderAlerts(running ? status.recent_alerts : []);
}

async function refresh() {
  const answer = await send({ type: 'status' });
  if (!answer) return render(null);
  if (!settingsLoaded) {
    settingsLoaded = true;
    $('port').value = answer.settings.port;
    $('notifications').value = answer.settings.notifications;
    $('command').textContent = answer.settings.port === 8765
      ? 'fatigue-guard gui --minimized'
      : `fatigue-guard gui --minimized --port ${answer.settings.port}`;
  }
  render(answer.status);
}

async function command(name) {
  await send({ type: 'command', name });
  setTimeout(refresh, 500);
}

$('toggle').addEventListener('click', () => command(running ? 'stop' : 'start'));
$('show').addEventListener('click', () => command('show'));
$('recalibrate').addEventListener('click', () => command('recalibrate'));
$('settings-toggle').addEventListener('click', () => {
  $('settings').hidden = !$('settings').hidden;
});
$('copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('command').textContent);
  $('copy').textContent = 'Copié';
  setTimeout(() => ($('copy').textContent = 'Copier'), 1500);
});
$('port').addEventListener('change', async () => {
  const port = Number($('port').value);
  if (Number.isInteger(port) && port >= 1024 && port <= 65535) {
    await send({ type: 'settings', values: { port } });
    settingsLoaded = false;
    refresh();
  }
});
$('notifications').addEventListener('change', () =>
  send({ type: 'settings', values: { notifications: $('notifications').value } }));

refresh();
setInterval(refresh, 1000);
