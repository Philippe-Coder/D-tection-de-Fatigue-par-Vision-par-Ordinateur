# Fatigue Guard pour navigateur

Affiche votre niveau de vigilance sur l'icône du navigateur et vous prévient en cas de fatigue ou de distraction pendant que vous naviguez, étudiez ou passez un examen en ligne.

Cette extension est un afficheur : l'analyse de la webcam est faite par l'application **Fatigue Guard** installée sur l'ordinateur. L'extension ne demande pas l'accès à la caméra, ne lit aucune page web et ne communique qu'avec `127.0.0.1`.

## Fonctions

- **Badge sur l'icône** : score de vigilance (vert, orange, rouge), `!` en cas de somnolence, `II` quand la surveillance est en pause
- **Fenêtre** : jauge, yeux fermés, clignements, bâillements, durée de session, alertes récentes
- **Commandes** : démarrer ou arrêter la surveillance, ouvrir le tableau de bord, recalibrer
- **Notifications** du navigateur pour les alertes (niveau réglable)

## Installation

Navigateurs testés : **Chrome** et **Edge** (version 158). Brave, Opera et Vivaldi utilisent le même moteur.

1. Ouvrez `chrome://extensions` (ou `edge://extensions`).
2. Activez le **mode développeur**.
3. Cliquez sur **Charger l'extension non empaquetée** et choisissez ce dossier `browser-extension`.
4. Épinglez l'icône Fatigue Guard dans la barre d'outils.

Puis lancez l'application :

```powershell
fatigue-guard gui --minimized
```

Firefox (121 ou plus récent) : le manifeste est prévu pour, via `about:debugging` → *Ce Firefox* → *Charger un module complémentaire temporaire*, mais ce navigateur n'a pas encore été testé.

## Réglages

Le bouton ⚙ de la fenêtre permet de changer le port de l'application (8765 par défaut) et le niveau des notifications.

## Permissions

| Permission | Usage |
|---|---|
| `http://127.0.0.1/*` | lire l'état de l'application locale |
| `notifications` | afficher les alertes |
| `storage` | mémoriser vos réglages |
| `alarms` | revérifier périodiquement si l'application est lancée |
