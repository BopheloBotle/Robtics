"""
training.py
===========
Neural network training pipeline for Component B.

Dataset generation
------------------
Training examples are generated from a rule-based teacher policy.

Labels
------
  front_dist < TEACHER_OBSTACLE_THRESHOLD                         -> STOP (3)
  front_dist < TEACHER_TURN_THRESHOLD and left_dist >= right_dist -> TURN_LEFT (1)
  front_dist < TEACHER_TURN_THRESHOLD and left_dist < right_dist  -> TURN_RIGHT (2)
  else                                                             -> FORWARD (0)

The dataset uses per-class targeted sampling so all four classes are
approximately equally represented (500 samples each, 2000 total).

Reproducibility: fixed seed from ConfigB.NN_RANDOM_SEED.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np

from neural_network import NumpyMLP
from configuration_b import ConfigB


# ------------------------------------------------------------------ #
#  Teacher policy                                                      #
# ------------------------------------------------------------------ #

def teacher_label(front: float, left: float, right: float, cfg: ConfigB) -> int:
    """
    Rule-based teacher policy.

    Returns 0=forward, 1=turn_left, 2=turn_right, 3=stop.
    """
    ot = cfg.TEACHER_OBSTACLE_THRESHOLD
    tt = cfg.TEACHER_TURN_THRESHOLD
    if front < ot:
        return 3
    if front < tt:
        return 1 if left >= right else 2
    return 0


# ------------------------------------------------------------------ #
#  Ray-casting helpers                                                 #
# ------------------------------------------------------------------ #

def _slab(ox: float, oy: float, dx: float, dy: float,
          x0: float, x1: float, y0: float, y1: float) -> float | None:
    """Ray-AABB intersection using the slab method."""
    tmin, tmax = 0.0, float("inf")
    if abs(dx) < 1e-12:
        if not (x0 <= ox <= x1):
            return None
    else:
        tx1, tx2 = (x0 - ox) / dx, (x1 - ox) / dx
        tmin = max(tmin, min(tx1, tx2))
        tmax = min(tmax, max(tx1, tx2))
    if abs(dy) < 1e-12:
        if not (y0 <= oy <= y1):
            return None
    else:
        ty1, ty2 = (y0 - oy) / dy, (y1 - oy) / dy
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))
    if tmax < tmin or tmax < 0:
        return None
    return tmin if tmin >= 0 else tmax


def _ray_to_obstacles(
    ox: float, oy: float, angle: float,
    obstacles: list[dict],
    max_range: float,
    wall: float,
) -> float:
    """Cast one ray from (ox, oy) at heading angle and return distance to nearest hit."""
    dx, dy = math.cos(angle), math.sin(angle)
    best = max_range

    for obs in obstacles:
        cx, cy = obs["x"], obs["y"]
        hw, hh = obs["w"] / 2.0, obs["h"] / 2.0
        t = _slab(ox, oy, dx, dy, cx - hw, cx + hw, cy - hh, cy + hh)
        if t is not None and 0.0 < t < best:
            best = t

    for axis, val in [("x", -wall), ("x", wall), ("y", -wall), ("y", wall)]:
        if axis == "x":
            if abs(dx) < 1e-12:
                continue
            t = (val - ox) / dx
        else:
            if abs(dy) < 1e-12:
                continue
            t = (val - oy) / dy
        if t > 0.0 and t < best:
            best = t

    return best


# ------------------------------------------------------------------ #
#  Per-class targeted sampling                                         #
# ------------------------------------------------------------------ #

def _sample_for_class(
    target_class: int,
    rng: random.Random,
    cfg: ConfigB,
    target_pos: tuple[float, float],
) -> list[float] | None:
    """
    Generate one feature vector whose teacher label equals target_class.

    Returns None after max_tries attempts if a matching pose cannot be found.
    """
    max_range = 2.5
    wall = cfg.WORLD_WIDTH / 2.0
    margin = 0.4

    for _ in range(300):
        if target_class == 0:
            # Forward: open space well away from obstacles
            rx = rng.uniform(cfg.WORLD_ORIGIN_X + 1.5,
                             cfg.WORLD_ORIGIN_X + cfg.WORLD_WIDTH - 1.5)
            ry = rng.uniform(cfg.WORLD_ORIGIN_Y + 1.5,
                             cfg.WORLD_ORIGIN_Y + cfg.WORLD_HEIGHT - 1.5)
            heading = rng.uniform(-math.pi, math.pi)
        else:
            # Classes 1,2,3: place robot near an obstacle
            obs = rng.choice(cfg.OBSTACLES)
            dist = rng.uniform(
                cfg.TEACHER_OBSTACLE_THRESHOLD + 0.02,
                cfg.TEACHER_TURN_THRESHOLD + 0.4,
            )
            ang = rng.uniform(-math.pi, math.pi)
            rx = obs["x"] + math.cos(ang) * dist
            ry = obs["y"] + math.sin(ang) * dist
            rx = max(cfg.WORLD_ORIGIN_X + margin,
                     min(cfg.WORLD_ORIGIN_X + cfg.WORLD_WIDTH - margin, rx))
            ry = max(cfg.WORLD_ORIGIN_Y + margin,
                     min(cfg.WORLD_ORIGIN_Y + cfg.WORLD_HEIGHT - margin, ry))
            # Heading toward obstacle with small perturbation
            heading = math.atan2(obs["y"] - ry, obs["x"] - rx) + rng.gauss(0, 0.3)

        front = _ray_to_obstacles(rx, ry, heading,              cfg.OBSTACLES, max_range, wall)
        left  = _ray_to_obstacles(rx, ry, heading + math.pi/2,  cfg.OBSTACLES, max_range, wall)
        right = _ray_to_obstacles(rx, ry, heading - math.pi/2,  cfg.OBSTACLES, max_range, wall)

        label = teacher_label(front, left, right, cfg)
        if label != target_class:
            continue

        tx, ty = target_pos
        herr = math.atan2(ty - ry, tx - rx) - heading
        while herr > math.pi:
            herr -= 2.0 * math.pi
        while herr < -math.pi:
            herr += 2.0 * math.pi

        return [
            max(0.0, min(1.0, front / max_range + rng.gauss(0, 0.01))),
            max(0.0, min(1.0, left  / max_range + rng.gauss(0, 0.01))),
            max(0.0, min(1.0, right / max_range + rng.gauss(0, 0.01))),
            max(-1.0, min(1.0, herr / math.pi   + rng.gauss(0, 0.01))),
            max(-1.0, min(1.0, herr / math.pi   + rng.gauss(0, 0.01))),
        ]

    return None   # failed to find valid pose in 300 tries


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def generate_training_data(cfg: ConfigB) -> Path:
    """
    Generate a balanced synthetic training dataset and save to CSV.

    Each of the four action classes receives exactly NN_NUM_SAMPLES//4
    samples.  Samples are generated using targeted per-class sampling so
    that the dataset is balanced regardless of the arena layout.

    Returns the path to the saved CSV file.
    """
    rng = random.Random(cfg.NN_RANDOM_SEED)
    n = cfg.NN_NUM_SAMPLES
    per_class = n // 4
    targets = list(cfg.TARGETS)

    X_all: list[list[float]] = []
    y_all: list[int] = []

    for cls in range(4):
        collected = 0
        t_idx = 0
        while collected < per_class:
            target = targets[t_idx % len(targets)]
            sample = _sample_for_class(cls, rng, cfg, target)
            if sample is not None:
                X_all.append(sample)
                y_all.append(cls)
                collected += 1
            t_idx += 1

    # Shuffle
    pairs = list(zip(X_all, y_all))
    rng.shuffle(pairs)
    X_all = [p[0] for p in pairs]
    y_all = [p[1] for p in pairs]

    csv_path = cfg.DATA_TRAINING / "nn_training_data.csv"
    header = ["front_norm", "left_norm", "right_norm",
              "goal_angle_norm", "heading_err_norm", "label"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for feats, lbl in zip(X_all, y_all):
            writer.writerow(feats + [lbl])

    meta = {
        "n_samples": len(y_all),
        "features": header[:-1],
        "classes": ["forward", "turn_left", "turn_right", "stop"],
        "label_counts": {str(c): y_all.count(c) for c in range(4)},
        "teacher_policy": "rule_based_threshold_balanced",
        "random_seed": cfg.NN_RANDOM_SEED,
        "source": "synthetic_offline_balanced",
    }
    with open(cfg.DATA_TRAINING / "nn_training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [data] {len(y_all)} samples saved to {csv_path}")
    print(f"  [data] Class distribution: {meta['label_counts']}")
    return csv_path


def train_and_evaluate_model(cfg: ConfigB) -> dict:
    """
    Load data, train NumpyMLP and MLPClassifier, evaluate both.

    Returns a comprehensive results dict and saves models + metrics.
    """
    csv_path = cfg.DATA_TRAINING / "nn_training_data.csv"
    if not csv_path.exists():
        generate_training_data(cfg)

    data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
    X_all = data[:, :cfg.NN_INPUT_SIZE].astype(float)
    y_all = data[:, cfg.NN_INPUT_SIZE].astype(int)

    rng = np.random.default_rng(cfg.NN_RANDOM_SEED)
    n = len(y_all)
    perm  = rng.permutation(n)
    split = int(n * cfg.NN_TRAIN_RATIO)
    X_train, y_train = X_all[perm[:split]], y_all[perm[:split]]
    X_test,  y_test  = X_all[perm[split:]], y_all[perm[split:]]

    print(f"  [nn] Training: {len(y_train)} samples, Test: {len(y_test)} samples")

    # ---- NumpyMLP ----
    net = NumpyMLP(
        input_size=cfg.NN_INPUT_SIZE,
        hidden_sizes=cfg.NN_HIDDEN_SIZES,
        output_size=cfg.NN_OUTPUT_SIZE,
        lr=cfg.NN_LEARNING_RATE,
        seed=cfg.NN_RANDOM_SEED,
    )
    net.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=cfg.NN_EPOCHS,
        batch_size=cfg.NN_BATCH_SIZE,
    )

    y_pred_numpy = net.predict(X_test)
    numpy_acc = float(np.mean(y_pred_numpy == y_test))

    model_path = cfg.MODELS_DIR / "numpy_mlp.npz"
    net.save(model_path)
    print(f"  [nn] NumpyMLP saved: {model_path}")
    print(f"  [nn] NumpyMLP test accuracy: {numpy_acc:.4f}")

    # ---- scikit-learn MLPClassifier ----
    sklearn_acc = None
    sklearn_report = None
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import classification_report
        from sklearn.preprocessing import StandardScaler
        import pickle

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        sk_net = MLPClassifier(
            hidden_layer_sizes=tuple(cfg.NN_HIDDEN_SIZES),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=cfg.NN_LEARNING_RATE,
            max_iter=cfg.NN_EPOCHS,
            random_state=cfg.NN_RANDOM_SEED,
        )
        sk_net.fit(X_tr_sc, y_train)
        y_pred_sk = sk_net.predict(X_te_sc)
        sklearn_acc = float(np.mean(y_pred_sk == y_test))
        sklearn_report = classification_report(
            y_test, y_pred_sk,
            target_names=["forward", "turn_left", "turn_right", "stop"],
            output_dict=True,
        )

        sk_path = cfg.MODELS_DIR / "sklearn_mlp.pkl"
        with open(sk_path, "wb") as f:
            pickle.dump({"model": sk_net, "scaler": scaler}, f)

        scaler_path = cfg.MODELS_DIR / "feature_scaler.json"
        with open(scaler_path, "w") as f:
            json.dump({
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }, f, indent=2)

        print(f"  [nn] scikit-learn MLPClassifier accuracy: {sklearn_acc:.4f}")
        print(f"  [nn] sklearn model saved: {sk_path}")

    except Exception as e:
        print(f"  [nn] scikit-learn training skipped: {e}")

    # ---- Confusion matrix (NumpyMLP) ----
    n_classes = cfg.NN_OUTPUT_SIZE
    numpy_cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test, y_pred_numpy):
        numpy_cm[int(true), int(pred)] += 1

    # ---- Per-class metrics ----
    names = ["forward", "turn_left", "turn_right", "stop"]
    class_metrics: dict = {}
    for c in range(n_classes):
        tp = int(numpy_cm[c, c])
        fp = int(numpy_cm[:, c].sum() - tp)
        fn = int(numpy_cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_metrics[names[c]] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": int(numpy_cm[c].sum()),
        }

    results = {
        "accuracy": numpy_acc,
        "sklearn_accuracy": sklearn_acc,
        "confusion_matrix": numpy_cm.tolist(),
        "class_metrics": class_metrics,
        "train_losses": net.train_losses,
        "val_losses":   net.val_losses,
        "sklearn_report": sklearn_report,
        "n_train": int(len(y_train)),
        "n_test":  int(len(y_test)),
        "epochs": cfg.NN_EPOCHS,
        "architecture": {
            "input": cfg.NN_INPUT_SIZE,
            "hidden": cfg.NN_HIDDEN_SIZES,
            "output": cfg.NN_OUTPUT_SIZE,
        },
    }

    metrics_path = cfg.RESULTS_METRICS / "nn_evaluation.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [nn] Evaluation metrics saved: {metrics_path}")

    return results
