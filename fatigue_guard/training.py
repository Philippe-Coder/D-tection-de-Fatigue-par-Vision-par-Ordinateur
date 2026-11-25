"""Entraînement du CNN d'état des yeux (ouvert / fermé) sur le MRL Eye Dataset.

Points importants :
- la séparation entraînement / validation / test se fait PAR PERSONNE, pour que
  le score mesure la généralisation à des visages jamais vus ;
- le prétraitement est exactement celui de l'inférence (`eye_cnn.preprocess`) ;
- le modèle est exporté en ONNX et relu avec OpenCV pour vérifier que
  l'application obtiendra bien les mêmes résultats.

Nécessite les dépendances d'entraînement : pip install -e ".[train]"
"""
from __future__ import annotations

import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .config import PROJECT_MODELS
from .eye_cnn import INPUT_SIZE, MODEL_NAME, preprocess

MRL_URL = "https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
SEED = 42


# ------------------------------------------------------------------ données
def download_mrl(data_dir: Path) -> Path:
    """Télécharge et décompresse le jeu MRL Eye (~85 000 images d'yeux)."""
    root = data_dir / "mrl_eyes"
    if any(root.rglob("*.png")):
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "mrlEyes_2018_01.zip"
    if not archive.exists():
        print(f"Téléchargement de {MRL_URL}")
        tmp = archive.with_suffix(".part")
        last = [0.0]

        def hook(blocks, block_size, total):
            now = time.monotonic()
            if total > 0 and now - last[0] > 2:
                last[0] = now
                done = blocks * block_size / 1e6
                print(f"  {done:6.0f} / {total / 1e6:.0f} Mo", flush=True)

        urllib.request.urlretrieve(MRL_URL, tmp, hook)
        tmp.replace(archive)
    print("Décompression…")
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    return root


def load_mrl(root: Path, limit: int = 0):
    """Renvoie (images uint8 N×1×32×32 prétraitées, labels, sujets).

    Nom de fichier : s0001_00001_0_0_<état>_0_0_01.png, état 0 = fermé, 1 = ouvert.
    Label renvoyé : 1 = fermé, 0 = ouvert.
    """
    # Lire 85 000 petits fichiers est lent : on garde le résultat en cache.
    cache = root.parent / f"mrl_cache_{INPUT_SIZE}_{limit}.npz"
    if cache.exists():
        data = np.load(cache)
        return data["x"], data["y"], data["subjects"]
    x, y, subjects = _read_mrl(root, limit)
    np.savez_compressed(cache, x=x, y=y, subjects=subjects)
    return x, y, subjects


def _read_mrl(root: Path, limit: int):
    files = sorted(root.rglob("*.png"))
    if not files:
        raise RuntimeError(f"Aucune image trouvée dans {root}")
    if limit:
        random.Random(SEED).shuffle(files)
        files = files[:limit]
    xs, ys, subjects = [], [], []
    for i, path in enumerate(files):
        parts = path.stem.split("_")
        if len(parts) < 8:
            continue
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        xs.append((preprocess(img) * 255).astype(np.uint8))
        ys.append(1 - int(parts[4]))
        subjects.append(parts[0])
        if i and i % 20000 == 0:
            print(f"  {i} / {len(files)} images chargées", flush=True)
    return np.stack(xs), np.array(ys, dtype=np.int64), np.array(subjects)


def split_by_subject(subjects: np.ndarray):
    """~70 % / 15 % / 15 % des personnes, sans recouvrement."""
    ids = sorted(set(subjects.tolist()))
    random.Random(SEED).shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    test_ids, val_ids = set(ids[:n_test]), set(ids[n_test:n_test + n_val])
    test = np.array([s in test_ids for s in subjects])
    val = np.array([s in val_ids for s in subjects])
    return ~(test | val), val, test


# ------------------------------------------------------------------- modèle
def build_model():
    import torch.nn as nn

    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True), nn.MaxPool2d(2)]

    return nn.Sequential(
        *block(1, 24), *block(24, 48), *block(48, 96),
        nn.Conv2d(96, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2),
    )


def augment(x):
    """Augmentations pensées pour passer de l'infrarouge du jeu de données à une
    webcam ordinaire : luminosité, contraste, flou, bruit, décalages, miroir."""
    import torch
    import torch.nn.functional as F

    n = x.shape[0]
    flip = torch.rand(n) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    gamma = torch.empty(n, 1, 1, 1).uniform_(0.6, 1.6)
    x = x.clamp(1e-4, 1).pow(gamma)
    contrast = torch.empty(n, 1, 1, 1).uniform_(0.7, 1.3)
    bright = torch.empty(n, 1, 1, 1).uniform_(-0.12, 0.12)
    x = (x - 0.5) * contrast + 0.5 + bright
    # Décalage, zoom et légère rotation.
    angle = torch.empty(n).uniform_(-0.2, 0.2)
    scale = torch.empty(n).uniform_(0.85, 1.15)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = scale * torch.cos(angle)
    theta[:, 0, 1] = -scale * torch.sin(angle)
    theta[:, 1, 0] = scale * torch.sin(angle)
    theta[:, 1, 1] = scale * torch.cos(angle)
    theta[:, :, 2] = torch.empty(n, 2).uniform_(-0.15, 0.15)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    blur = torch.rand(n) < 0.3
    if blur.any():
        x[blur] = F.avg_pool2d(x[blur], 3, stride=1, padding=1)
    x = x + torch.randn_like(x) * 0.03
    return x.clamp(0, 1)


def evaluate(model, x, y, batch: int = 1024):
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).float() / 255.0
            preds.append(model(xb).argmax(1).numpy())
    pred = np.concatenate(preds)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "accuracy": (tp + tn) / max(1, len(y)),
        "recall_closed": tp / max(1, tp + fn),      # yeux fermés bien détectés
        "precision_closed": tp / max(1, tp + fp),   # fausses alertes
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n": int(len(y)),
    }


# -------------------------------------------------------------- entraînement
def train(data: str = "data", epochs: int = 12, limit: int = 0, batch: int = 256) -> int:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print('PyTorch est requis pour l\'entraînement : pip install -e ".[train]"')
        return 1

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    root = download_mrl(Path(data))
    print("Chargement et prétraitement des images…")
    x, y, subjects = load_mrl(root, limit)
    tr, va, te = split_by_subject(subjects)
    print(f"{len(y)} images, {len(set(subjects.tolist()))} personnes · "
          f"entraînement {tr.sum()} / validation {va.sum()} / test {te.sum()} · "
          f"{y.mean() * 100:.0f} % d'yeux fermés")

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * (int(tr.sum()) // batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    x_tr, y_tr = x[tr], y[tr]
    best_acc, best_state, patience = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.monotonic()
        order = np.random.permutation(len(y_tr))
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xb = augment(torch.from_numpy(x_tr[idx]).float() / 255.0)
            yb = torch.from_numpy(y_tr[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        val = evaluate(model, x[va], y[va])
        print(f"Époque {epoch:2d}/{epochs} · perte {total / len(order):.4f} · "
              f"validation {val['accuracy'] * 100:.2f} % · {time.monotonic() - t0:.0f} s",
              flush=True)
        if val["accuracy"] > best_acc:
            best_acc, patience = val["accuracy"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                print("Arrêt anticipé : la validation ne progresse plus.")
                break

    model.load_state_dict(best_state)
    model.eval()
    test = evaluate(model, x[te], y[te])
    print(f"\nTest (personnes jamais vues) : précision globale {test['accuracy'] * 100:.2f} % · "
          f"yeux fermés détectés {test['recall_closed'] * 100:.2f} % · "
          f"fiabilité des « fermés » {test['precision_closed'] * 100:.2f} %")

    # Export ONNX + vérification avec le moteur réellement utilisé par l'application.
    PROJECT_MODELS.mkdir(exist_ok=True)
    out = PROJECT_MODELS / MODEL_NAME
    dummy = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(model, dummy, str(out), input_names=["eye"], output_names=["logits"],
                      dynamic_axes={"eye": {0: "n"}, "logits": {0: "n"}}, opset_version=13,
                      dynamo=False)
    net = cv2.dnn.readNetFromONNX(str(out))
    sample = x[te][:512].astype(np.float32) / 255.0
    net.setInput(sample)
    cv_pred = net.forward().argmax(1)
    with torch.no_grad():
        th_pred = model(torch.from_numpy(sample)).argmax(1).numpy()
    agree = float((cv_pred == th_pred).mean())
    print(f"Export ONNX : {out} · accord OpenCV / PyTorch {agree * 100:.1f} %")

    report = {"dataset": "MRL Eye Dataset", "images": int(len(y)),
              "validation_accuracy": best_acc, "test": test, "onnx_agreement": agree,
              "epochs_run": epoch, "limit": limit}
    (PROJECT_MODELS / "eye_state_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if agree < 0.99:
        print("ATTENTION : l'export ONNX ne reproduit pas fidèlement le modèle.")
        return 1
    return 0
