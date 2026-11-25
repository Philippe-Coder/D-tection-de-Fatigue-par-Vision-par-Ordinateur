# Fatigue Guard — Détection de fatigue et de distraction par vision par ordinateur

Fatigue Guard surveille votre visage avec la webcam et vous alerte quand vous vous endormez, que vous fatiguez ou que votre attention décroche. Tout le traitement se fait sur votre machine : **aucune image n'est enregistrée ni envoyée**.

- Tableau de bord graphique moderne (thème sombre, vidéo annotée, jauge de vigilance, courbes, historique)
- Mode terminal et mode discret dans la zone de notification
- Extension VS Code qui affiche votre vigilance dans la barre d'état
- Extension de navigateur (Chrome, Edge) avec badge de vigilance et notifications
- Cinq profils : bureau, conduite, confort visuel, étude, examen

---

## Ce qui est surveillé

| Zone | Signal | Ce que cela révèle |
|---|---|---|
| Yeux | fermeture prolongée | micro-sommeil (alarme immédiate) |
| Yeux | part du temps yeux fermés sur 1 min (PERCLOS) | somnolence qui s'installe |
| Yeux | fréquence et durée des clignements | fatigue, fatigue visuelle |
| Bouche | bâillements (ouverture maintenue, pas la parole) | fatigue |
| Tête | inclinaison vers le bas | tête qui tombe, téléphone |
| Tête | basculée en arrière, penchée sur l'épaule | posture de sommeil ; alarme si elle est marquée, tenue et immobile |
| Tête | visage perdu juste après une chute de tête ou des yeux fermés | affaissement hors du champ (alarme, et non « Absent ») |
| Tête et regard | rotation, regard latéral | distraction |
| Visage | absence, plusieurs visages | départ, autre personne (examen) |
| Position | distance à l'écran | confort visuel |

Ces signaux sont combinés en un **score de vigilance de 0 à 100** et un état : *Attentif*, *Distrait*, *Fatigue*, *Somnolence* ou *Absent*. Une calibration de 5 secondes au démarrage adapte les seuils à votre visage.

## Profils

| Profil | Clé | Comportement |
|---|---|---|
| Travail au bureau | `bureau` | somnolence, rappel de pause toutes les 45 min, tolérant sur les regards hors écran |
| Conduite | `conduite` | alarme dès 0,8 s d'yeux fermés, 2 s de regard hors de la route, tête qui tombe |
| Confort visuel | `ecran` | clignements, distance à l'écran, règle 20-20-20 |
| Étude / concentration | `etude` | signale quand l'attention quitte le support |
| Examen | `examen` | regard hors écran, absence, présence d'une autre personne |

> **Conduite** : Fatigue Guard est une aide, pas un dispositif de sécurité homologué. Au moindre signe de somnolence, arrêtez-vous.
>
> **Examen** : n'utilisez ce profil qu'avec l'accord de la personne surveillée.

---

## Installation

Prérequis : **Python 3.10 à 3.12** (version officielle de python.org) et une webcam.

```powershell
git clone https://github.com/Philippe-Coder/D-tection-de-Fatigue-par-Vision-par-Ordinateur.git
cd D-tection-de-Fatigue-par-Vision-par-Ordinateur
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Sous Linux ou macOS : `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`

Au premier lancement, le modèle de visage (environ 4 Mo) est téléchargé une seule fois dans `~/.fatigue_guard/models`.

Vérifiez que tout fonctionne :

```powershell
fatigue-guard doctor
```

## Utilisation

```powershell
fatigue-guard                              # tableau de bord graphique
fatigue-guard gui --profile conduite --start
fatigue-guard gui --minimized              # discret, dans la zone de notification
fatigue-guard run --profile etude          # dans le terminal, sans fenêtre
fatigue-guard evaluate                     # test guidé posture par posture (2 min)
fatigue-guard stats                        # historique des sessions
fatigue-guard profiles                     # liste des profils
```

Options communes à `gui` et `run` : `--profile`, `--camera N`, `--no-sound`, `--no-cnn`, `--port`.

### Test guidé

`fatigue-guard evaluate` vous fait prendre dix poses (yeux fermés, tête en arrière, tête sur l'épaule, bâillement…), signalées par des bips pour celles où vos yeux sont fermés. Il affiche pour chacune si elle a été détectée, et enregistre les mesures chiffrées (aucune image) dans `~/.fatigue_guard/evaluations/`. Ce fichier sert à régler les seuils sur votre visage, votre caméra et votre éclairage.

Après un changement de réglage, `fatigue-guard evaluate --replay <fichier.csv>` repasse un enregistrement dans le code actuel, sans caméra, et réaffiche le tableau.

Quand la surveillance tourne, fermer la fenêtre la laisse active en arrière-plan ; l'icône de la zone de notification change de couleur selon votre état et permet de quitter.

## Extension VS Code

L'extension lit l'état de l'application et l'affiche dans la barre d'état (par exemple `87 % Attentif`). Elle peut démarrer et arrêter la surveillance, et relaie les alertes en notifications.

```powershell
cd vscode-extension
npx @vscode/vsce package
code --install-extension fatigue-guard-1.0.0.vsix
```

Puis, dans les réglages de VS Code, indiquez le chemin de la commande si elle n'est pas dans le `PATH` :

```json
"fatigueGuard.command": "C:\\chemin\\du\\projet\\.venv\\Scripts\\fatigue-guard.exe"
```

## Extension de navigateur

L'extension affiche le score de vigilance sur l'icône du navigateur, ouvre une fenêtre de suivi et relaie les alertes. Testée sur Chrome et Edge.

1. Ouvrez `chrome://extensions` (ou `edge://extensions`) et activez le **mode développeur**.
2. Cliquez sur **Charger l'extension non empaquetée** et choisissez le dossier [browser-extension/](browser-extension/).
3. Lancez l'application : `fatigue-guard gui --minimized`.

Détails dans [browser-extension/README.md](browser-extension/README.md).

### Serveur local

L'application expose son état sur `http://127.0.0.1:8765/status` (JSON, chiffres uniquement). Il n'est joignable que depuis votre machine. Les commandes (`/start`, `/stop`, `/show`, `/recalibrate`) exigent un en-tête qu'une page web ne peut pas envoyer, et seules les extensions de navigateur sont autorisées en accès croisé : un site web quelconque ne peut ni lire votre état ni piloter l'application.

---

## Entraînement du CNN des yeux (facultatif)

Sans entraînement, l'application fonctionne déjà avec MediaPipe. Le CNN ajoute un second avis sur l'état des yeux (ouvert / fermé), utile avec des lunettes ou en faible lumière.

```powershell
pip install -e ".[train]"
fatigue-guard train                 # télécharge le MRL Eye Dataset puis entraîne
fatigue-guard train --limit 8000    # essai rapide
```

Le script :

1. télécharge le **MRL Eye Dataset** (environ 85 000 images d'yeux, 37 personnes) ;
2. sépare entraînement, validation et test **par personne**, pour mesurer la généralisation à des visages jamais vus ;
3. entraîne un petit CNN (4 couches de convolution, entrée 32×32 en niveaux de gris) avec arrêt anticipé ;
4. exporte le modèle dans `models/eye_state.onnx` et vérifie qu'OpenCV reproduit les résultats ;
5. écrit le rapport de précision dans `models/eye_state_report.json`.

L'application détecte et utilise automatiquement le modèle au lancement suivant (`--no-cnn` pour comparer sans).

Le modèle fourni dans `models/` a été entraîné ainsi (12 époques, environ 15 minutes sur processeur). Sur les 7 190 images de test, issues de personnes absentes de l'entraînement : **94,0 % de précision**, 93,7 % des yeux fermés détectés, 96,8 % de fiabilité quand il annonce « fermé ». Les images du jeu de données sont en infrarouge : sur une webcam ordinaire, les performances peuvent différer.

---

## Architecture

```
fatigue_guard/
├── config.py      profils et constantes
├── detector.py    visage : 478 points, blendshapes, pose (MediaPipe Face Landmarker)
├── features.py    mesures géométriques (yeux, bouche, angles de la tête)
├── eye_cnn.py     CNN d'état des yeux, inférence ONNX via OpenCV
├── analyzer.py    analyse temporelle, score de vigilance, alertes
├── engine.py      boucle caméra dans un thread dédié
├── alerts.py      sons d'alerte
├── storage.py     historique SQLite
├── server.py      serveur d'état local pour les extensions
├── gui.py         tableau de bord PySide6
├── cli.py         ligne de commande
└── training.py    entraînement du CNN
vscode-extension/  extension VS Code
browser-extension/ extension Chrome / Edge (Manifest V3)
tests/             tests du cœur d'analyse (sans caméra)
```

| Brique | Choix | Raison |
|---|---|---|
| Suivi du visage | MediaPipe Face Landmarker | 478 points, pose et expressions en temps réel sur simple processeur |
| Interface | PySide6 (Qt 6) | liaison officielle Qt, rendu net sur écrans haute densité |
| Inférence du CNN | ONNX + OpenCV DNN | aucune dépendance lourde à l'exécution |
| Entraînement | PyTorch | uniquement pour entraîner, pas nécessaire pour utiliser |
| Historique | SQLite | intégré à Python, un seul fichier local |

## Tests

```powershell
pip install -e ".[dev]"
pytest
```

Les tests simulent des scénarios complets (clignements normaux, micro-sommeil, bâillements, parole, regard détourné, absence…) avec une horloge virtuelle.

## Dépannage

- **Caméra inaccessible** : fermez les autres applications qui l'utilisent et autorisez l'accès dans *Paramètres → Confidentialité → Caméra*. Essayez `--camera 1`.
- **`pip install` échoue** : vérifiez que vous utilisez un Python officiel (`py -0p`), pas celui de MSYS2 ou du Microsoft Store.
- **Fausses alertes** : cliquez sur *Recalibrer* en regardant l'écran normalement, et vérifiez que votre visage est bien éclairé.
- **Port occupé** : lancez avec `--port 8766` et reportez ce port dans le réglage `fatigueGuard.port` de VS Code.

## Vie privée

Les images de la webcam sont analysées en mémoire puis jetées. Seuls des chiffres (durée de session, score, nombre d'alertes) sont conservés, dans `~/.fatigue_guard/history.db`.

## Licence

Distribué sous **licence MIT** — voir [LICENSE](LICENSE).

Copyright (c) 2026 Philippe-Coder
