# Détection de Fatigue par Vision par Ordinateur

Application de détection de fatigue en temps réel à partir d'un flux vidéo (webcam), basée sur un réseau de neurones convolutif (CNN) entraîné avec TensorFlow/Keras et une interface graphique PyQt5.

---

## Table des matières

1. [Présentation du projet](#1-présentation-du-projet)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Prérequis](#3-prérequis)
4. [Installation](#4-installation)
5. [Préparation des données](#5-préparation-des-données)
6. [Entraînement du modèle](#6-entraînement-du-modèle)
7. [Lancement de l'application](#7-lancement-de-lapplication)
8. [Explication technique des modules](#8-explication-technique-des-modules)
9. [Dépannage (FAQ)](#9-dépannage-faq)
10. [Améliorations possibles](#10-améliorations-possibles)

---

## 1. Présentation du projet

Ce projet a pour objectif de **détecter automatiquement les signes de fatigue** chez une personne filmée par une webcam (par exemple un conducteur). Le système :

- capture le flux vidéo en temps réel ;
- transmet chaque image (frame) à un modèle CNN ;
- affiche le résultat **"Fatigué"** ou **"Pas fatigué"** dans une fenêtre PyQt5.

Le modèle est un **classifieur binaire** entraîné sur deux classes : `fatigued` et `non_fatigued`.

---

## 2. Architecture du projet

```
fatigue_detection_project/
│
├── main.py                       # Point d'entrée — lance l'interface graphique
├── requirements.txt              # Dépendances Python
│
├── data/                         # Jeu de données (images)
│   ├── train/
│   │   ├── fatigued/
│   │   └── non_fatigued/
│   ├── val/
│   │   ├── fatigued/
│   │   └── non_fatigued/
│   └── test/
│       ├── fatigued/
│       └── non_fatigued/
│
├── models/                       # Modèles entraînés (.h5)
│   └── fatigue_detection_model.h5
│
└── src/                          # Code source
    ├── train_model.py            # Entraînement du CNN
    ├── predict_fatigue.py        # Inférence sur une image
    ├── gui.py                    # Interface graphique PyQt5
    └── utils.py                  # Fonctions utilitaires (prétraitement)
```

---

## 3. Prérequis

- **Python 3.9 – 3.11** (recommandé pour TensorFlow 2.18)
- **Une webcam** fonctionnelle
- **Windows / macOS / Linux**
- **GPU optionnel** (NVIDIA + CUDA) pour accélérer l'entraînement

Bibliothèques principales (voir [requirements.txt](requirements.txt)) :

| Bibliothèque | Version | Rôle |
|---|---|---|
| tensorflow | 2.18.0 | Entraînement et inférence du CNN |
| opencv-python | 4.7.0.72 | Capture vidéo et traitement d'image |
| PyQt5 | 5.15.7 | Interface graphique |
| numpy | 1.23.5 | Manipulation des tableaux |

---

## 4. Installation

### 4.1. Cloner ou télécharger le projet

Placez-vous dans le dossier `fatigue_detection_project`.

### 4.2. Créer un environnement virtuel (recommandé)

**Windows (PowerShell) :**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS :**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 5. Préparation des données

Le projet attend un jeu de données organisé selon la structure suivante :

```
data/
├── train/        # ≈ 70-80 % des images
│   ├── fatigued/
│   └── non_fatigued/
├── val/          # ≈ 10-15 % des images
│   ├── fatigued/
│   └── non_fatigued/
└── test/         # ≈ 10-15 % des images
    ├── fatigued/
    └── non_fatigued/
```

**Conseils :**
- Les images peuvent être en JPG ou PNG.
- Elles seront automatiquement redimensionnées en **128×128 pixels** lors de l'entraînement.
- Plus le dataset est équilibré et varié (luminosité, angles, sujets), meilleurs sont les résultats.
- Sources possibles de données : datasets publics (NTHU-DDD, YawDD, etc.) ou enregistrement personnel.

---

## 6. Entraînement du modèle

Une fois les données placées dans `data/train` et `data/val`, lancez :

```bash
cd src
python train_model.py
```

Ce script :
1. Applique de la **data augmentation** sur le set d'entraînement (rotations, zoom, flip horizontal, etc.) ;
2. Construit un **CNN séquentiel** :
   - 2 blocs `Conv2D + MaxPooling`
   - `Flatten → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)`
3. Compile avec l'optimiseur **Adam** et la perte **binary_crossentropy** ;
4. Entraîne pendant **10 epochs** ;
5. Sauvegarde le modèle dans [models/fatigue_detection_model.h5](models/fatigue_detection_model.h5).

Vous pouvez modifier le nombre d'epochs, la taille du batch ou l'architecture directement dans [src/train_model.py](src/train_model.py).

---

## 7. Lancement de l'application

Une fois le modèle entraîné, lancez l'interface depuis la **racine du projet** :

```bash
python main.py
```

**Utilisation :**
1. La fenêtre **"Détection de Fatigue"** s'ouvre.
2. Cliquez sur **"Démarrer"** pour activer la webcam.
3. Le statut **"Fatigué"** ou **"Pas fatigué"** s'affiche en temps réel.
4. Appuyez sur la touche **`q`** dans la fenêtre vidéo OpenCV pour quitter.

---

## 8. Explication technique des modules

### `main.py`
Point d'entrée minimal qui importe et lance `start_gui()` depuis le module GUI.

### [src/gui.py](src/gui.py)
- Définit la classe `FatigueApp` (héritière de `QtWidgets.QWidget`).
- Charge le modèle entraîné au démarrage.
- Sur clic du bouton, ouvre la webcam via `cv2.VideoCapture(0)` et appelle `predict_fatigue()` sur chaque frame.

### [src/predict_fatigue.py](src/predict_fatigue.py)
- Fonction `predict_fatigue(frame, model)` :
  1. Redimensionne l'image en 128×128 ;
  2. Normalise les pixels (`/255`) ;
  3. Effectue l'inférence ;
  4. Retourne `True` si la probabilité est > 0.5 (seuil de décision).

### [src/train_model.py](src/train_model.py)
- Définit `train_cnn_model(train_dir, val_dir, model_path)` ;
- Utilise `ImageDataGenerator` pour le chargement par lots et l'augmentation ;
- Sauvegarde le modèle au format HDF5 (`.h5`).

### [src/utils.py](src/utils.py)
- Fonction `preprocess_image(image)` : redimensionne + normalise + reshape pour l'inférence (réutilisable).

---

## 9. Dépannage (FAQ)

### Le modèle n'est pas trouvé au lancement
Vérifiez que `models/fatigue_detection_model.h5` existe. Sinon, lancez d'abord l'entraînement (section 6). Le chemin est résolu automatiquement par rapport à l'emplacement du fichier `gui.py`, donc le lancement fonctionne aussi bien depuis la racine que depuis `src/`.

### La webcam ne s'ouvre pas
- Vérifiez que rien d'autre n'utilise la caméra.
- Sur Windows, autorisez l'accès caméra dans **Paramètres → Confidentialité**.
- Essayez `cv2.VideoCapture(1)` si plusieurs caméras sont présentes.

### TensorFlow trop lent
- Utilisez un GPU compatible CUDA, ou
- Réduisez la résolution d'entrée et la taille du batch dans `train_model.py`.

---

## 10. Améliorations possibles

- **Détection faciale préalable** (Haar Cascade ou MTCPnn) pour cibler uniquement le visage avant la classification.
- **Détection des yeux fermés** via Eye Aspect Ratio (EAR) avec dlib.
- **Alerte sonore** lorsque l'état "Fatigué" persiste plusieurs secondes.
- **Lissage temporel** des prédictions (moyenne mobile) pour éviter les oscillations.
- **Export du modèle en TFLite** pour un déploiement mobile (Android/iOS).
- **Suivi des métriques** (TensorBoard) et **callbacks** (EarlyStopping, ModelCheckpoint).

---

## Licence

Projet académique.