# Fatigue Guard pour VS Code

Affiche votre niveau de vigilance dans la barre d'état et vous prévient en cas de fatigue ou de distraction pendant que vous codez.

Cette extension est un afficheur : l'analyse de la webcam est faite par l'application **Fatigue Guard**, qui doit être installée sur la machine (`pip install -e .` dans le dépôt du projet). Aucune image ne passe par l'extension.

## Fonctions

- Barre d'état : `87 % Attentif`, colorée en orange (fatigue, distraction) ou en rouge (somnolence)
- Infobulle détaillée : yeux fermés, clignements, bâillements, durée de session
- Notifications pour les alertes
- Commandes : démarrer, arrêter, ouvrir le tableau de bord, recalibrer

## Réglages

| Réglage | Défaut | Rôle |
|---|---|---|
| `fatigueGuard.command` | `fatigue-guard` | commande ou chemin complet de l'application |
| `fatigueGuard.profile` | `bureau` | profil utilisé au démarrage |
| `fatigueGuard.port` | `8765` | port du serveur local |
| `fatigueGuard.notifications` | `alertes` | niveau de notification |
| `fatigueGuard.autoStart` | `false` | démarrer à l'ouverture de VS Code |
