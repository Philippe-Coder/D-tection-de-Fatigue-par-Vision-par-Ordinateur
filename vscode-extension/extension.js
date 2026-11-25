// Fatigue Guard pour VS Code : lit l'état de l'application locale et l'affiche
// dans la barre d'état. Aucune image ne transite ici, seulement des chiffres.
const vscode = require('vscode');
const http = require('http');
const { spawn } = require('child_process');

const POLL_MS = 1500;
const ICONS = {
  ATTENTIF: '$(eye)',
  CALIBRATION: '$(sync~spin)',
  DISTRAIT: '$(eye-closed)',
  FATIGUE: '$(warning)',
  SOMNOLENT: '$(alert)',
  ABSENT: '$(circle-slash)',
};

let item;
let timer;
let lastAlertId = null;
let launching = false;

function config() {
  return vscode.workspace.getConfiguration('fatigueGuard');
}

function request(method, path) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        host: '127.0.0.1',
        port: config().get('port', 8765),
        path,
        method,
        timeout: 1200,
        headers: { 'X-Fatigue-Guard': '1' },
      },
      (res) => {
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => (body += chunk));
        res.on('end', () => {
          try {
            resolve(JSON.parse(body));
          } catch (err) {
            reject(err);
          }
        });
      }
    );
    req.on('timeout', () => req.destroy(new Error('timeout')));
    req.on('error', reject);
    req.end();
  });
}

function wanted(level) {
  const mode = config().get('notifications', 'alertes');
  if (mode === 'aucune') return false;
  if (mode === 'alarmes') return level === 'alarm';
  if (mode === 'alertes') return level !== 'info';
  return true;
}

function notify(alerts) {
  if (!Array.isArray(alerts) || alerts.length === 0) {
    return;
  }
  const newest = alerts[alerts.length - 1].id;
  if (lastAlertId === null || newest < lastAlertId) {
    // Première lecture, ou application redémarrée : on ne rejoue pas l'historique.
    lastAlertId = newest;
    return;
  }
  for (const alert of alerts) {
    if (alert.id <= lastAlertId || !wanted(alert.level)) continue;
    const text = `Fatigue Guard — ${alert.message}`;
    if (alert.level === 'alarm') vscode.window.showErrorMessage(text);
    else if (alert.level === 'warn') vscode.window.showWarningMessage(text);
    else vscode.window.showInformationMessage(text);
  }
  lastAlertId = newest;
}

function render(status) {
  if (!status) {
    item.text = launching ? '$(sync~spin) Fatigue Guard' : '$(eye-closed) Fatigue Guard';
    item.tooltip = launching
      ? 'Démarrage de Fatigue Guard…'
      : "Fatigue Guard n'est pas lancé — cliquez pour démarrer la surveillance";
    item.backgroundColor = undefined;
    item.command = 'fatigueGuard.start';
    return;
  }
  item.command = 'fatigueGuard.menu';
  if (!status.running) {
    item.text = '$(debug-pause) Fatigue Guard';
    item.tooltip = 'Surveillance en pause — cliquez pour les options';
    item.backgroundColor = undefined;
    return;
  }
  const icon = ICONS[status.state] || '$(eye)';
  const showScore = status.state !== 'CALIBRATION' && status.state !== 'ABSENT';
  item.text = showScore
    ? `${icon} ${Math.round(status.vigilance)} % ${status.state_label}`
    : `${icon} ${status.state_label}`;
  item.backgroundColor =
    status.state === 'SOMNOLENT'
      ? new vscode.ThemeColor('statusBarItem.errorBackground')
      : status.state === 'FATIGUE' || status.state === 'DISTRAIT'
      ? new vscode.ThemeColor('statusBarItem.warningBackground')
      : undefined;

  const tip = new vscode.MarkdownString();
  tip.appendMarkdown(`**Fatigue Guard** — ${status.profile_label}\n\n`);
  tip.appendMarkdown(`| | |\n|---|---|\n`);
  tip.appendMarkdown(`| Vigilance | ${Math.round(status.vigilance)} % |\n`);
  tip.appendMarkdown(`| Yeux fermés (1 min) | ${Math.round(status.perclos * 100)} % |\n`);
  tip.appendMarkdown(`| Clignements | ${Math.round(status.blink_rate)} / min |\n`);
  tip.appendMarkdown(`| Bâillements (5 min) | ${status.yawns} |\n`);
  tip.appendMarkdown(`| Session | ${Math.floor(status.session_s / 60)} min |\n`);
  if (status.reason) tip.appendMarkdown(`\n${status.reason}`);
  item.tooltip = tip;
}

async function poll() {
  try {
    const status = await request('GET', '/status');
    if (status.app !== 'fatigue-guard') throw new Error('autre service');
    launching = false;
    render(status);
    if (status.running) notify(status.recent_alerts);
    return status;
  } catch (err) {
    render(null);
    return null;
  }
}

async function command(name) {
  try {
    await request('POST', `/${name}`);
    setTimeout(poll, 400);
    return true;
  } catch (err) {
    return false;
  }
}

async function start() {
  const status = await poll();
  if (status) {
    if (!status.running) await command('start');
    return;
  }
  if (launching) return;
  launching = true;
  render(null);

  const exe = config().get('command', 'fatigue-guard');
  const profile = config().get('profile', 'bureau');
  const port = config().get('port', 8765);
  const quoted = /\s/.test(exe) && !exe.startsWith('"') ? `"${exe}"` : exe;
  const line = `${quoted} gui --minimized --profile ${profile} --port ${port}`;
  let failed = false;
  const fail = (detail) => {
    if (failed) return;
    failed = true;
    launching = false;
    render(null);
    vscode.window
      .showErrorMessage(
        `Impossible de lancer Fatigue Guard (${detail}). Vérifiez le réglage « fatigueGuard.command ».`,
        'Ouvrir les réglages'
      )
      .then((choice) => {
        if (choice) {
          vscode.commands.executeCommand('workbench.action.openSettings', 'fatigueGuard.command');
        }
      });
  };

  const child = spawn(line, { shell: true, detached: true, stdio: 'ignore', windowsHide: true });
  child.on('error', (err) => fail(err.message));
  child.on('exit', (code) => {
    if (code && launching) fail(`code ${code}`);
  });
  child.unref();

  // Le premier lancement charge les modèles : on laisse jusqu'à 40 s.
  const deadline = Date.now() + 40000;
  const wait = async () => {
    if (failed) return;
    if (await poll()) return;
    if (Date.now() > deadline) return fail('aucune réponse');
    setTimeout(wait, 1000);
  };
  setTimeout(wait, 1500);
}

async function menu() {
  const status = await poll();
  if (!status) return start();
  const choices = [
    status.running
      ? { label: '$(debug-stop) Arrêter la surveillance', run: () => command('stop') }
      : { label: '$(play) Démarrer la surveillance', run: () => command('start') },
    { label: '$(dashboard) Ouvrir le tableau de bord', run: () => command('show') },
    { label: '$(target) Recalibrer', run: () => command('recalibrate') },
    {
      label: '$(gear) Réglages',
      run: () => vscode.commands.executeCommand('workbench.action.openSettings', 'fatigueGuard'),
    },
  ];
  const pick = await vscode.window.showQuickPick(choices, { placeHolder: 'Fatigue Guard' });
  if (pick) pick.run();
}

async function needsApp(name) {
  if (!(await command(name))) {
    vscode.window.showInformationMessage("Fatigue Guard n'est pas lancé.");
  }
}

function activate(context) {
  item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 1000);
  render(null);
  item.show();

  context.subscriptions.push(
    item,
    vscode.commands.registerCommand('fatigueGuard.start', start),
    vscode.commands.registerCommand('fatigueGuard.stop', () => needsApp('stop')),
    vscode.commands.registerCommand('fatigueGuard.showDashboard', () => needsApp('show')),
    vscode.commands.registerCommand('fatigueGuard.recalibrate', () => needsApp('recalibrate')),
    vscode.commands.registerCommand('fatigueGuard.menu', menu)
  );

  timer = setInterval(poll, POLL_MS);
  context.subscriptions.push({ dispose: () => clearInterval(timer) });
  poll().then((status) => {
    if (!status && config().get('autoStart', false)) start();
  });
}

function deactivate() {
  if (timer) clearInterval(timer);
}

module.exports = { activate, deactivate };
