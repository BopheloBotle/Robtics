"""
neural_network.py
=================
Lightweight NumPy feed-forward neural network for robot action classification.

Architecture
------------
  Input  : 5 features
             0. front_dist_norm  — normalised front LiDAR distance [0,1]
             1. left_dist_norm   — normalised left LiDAR distance [0,1]
             2. right_dist_norm  — normalised right LiDAR distance [0,1]
             3. goal_angle_norm  — goal bearing / π ∈ [−1, 1]
             4. heading_err_norm — heading error / π ∈ [−1, 1]

  Hidden : two layers, sizes [16, 8] with ReLU activation
  Output : 4-class softmax
             0 → forward
             1 → turn_left
             2 → turn_right
             3 → stop

Training
--------
Mini-batch stochastic gradient descent with cross-entropy loss.
No external deep-learning library is required.  Uses only NumPy.

This intentionally lightweight implementation is fully explainable
and does not depend on scikit-learn (though scikit-learn's MLPClassifier
is used alongside it in training.py for comparison).

Design note
-----------
The network is deliberately small so that a student can trace forward
and backward passes manually during a demonstration.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from configuration_b import ConfigB


# ------------------------------------------------------------------ #
#  Activation functions                                                #
# ------------------------------------------------------------------ #

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_deriv(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where x > 0, else 0."""
    return (x > 0.0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean cross-entropy loss.

    probs  : (N, C) softmax probabilities
    labels : (N,)  integer class labels
    """
    n = probs.shape[0]
    log_p = np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0))
    return float(-log_p.mean())


# ------------------------------------------------------------------ #
#  Network                                                             #
# ------------------------------------------------------------------ #

class NumpyMLP:
    """
    Two-hidden-layer multi-layer perceptron trained with mini-batch SGD.

    Parameters
    ----------
    input_size   : number of input features
    hidden_sizes : list of hidden layer widths
    output_size  : number of output classes
    lr           : learning rate
    seed         : random seed for weight initialisation
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.lr = lr
        rng = np.random.default_rng(seed)

        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.weights: list[np.ndarray] = []
        self.biases:  list[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            # He initialisation for ReLU layers; Xavier for output
            scale = math.sqrt(2.0 / fan_in) if i < len(layer_sizes) - 2 else math.sqrt(1.0 / fan_in)
            W = rng.standard_normal((fan_in, layer_sizes[i + 1])) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)

        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def _forward(
        self, X: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Full forward pass.

        Returns
        -------
        activations : list of post-activation values (including input)
        pre_act     : list of pre-activation (z) values
        """
        activations = [X]
        pre_act = []
        a = X
        n_layers = len(self.weights)

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            pre_act.append(z)
            if i < n_layers - 1:
                a = relu(z)
            else:
                a = softmax(z)
            activations.append(a)

        return activations, pre_act

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for input array X (N, input_size)."""
        activations, _ = self._forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class predictions."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------ #
    #  Backward pass (backpropagation)                                     #
    # ------------------------------------------------------------------ #

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_act: list[np.ndarray],
        labels: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compute gradients via backpropagation.

        Returns (grad_W, grad_b) lists in forward order.
        """
        n = labels.shape[0]
        n_layers = len(self.weights)
        grad_W = [None] * n_layers
        grad_b = [None] * n_layers

        # Output layer gradient (softmax + cross-entropy combined)
        delta = activations[-1].copy()     # (N, C)
        delta[np.arange(n), labels] -= 1.0
        delta /= n

        for i in reversed(range(n_layers)):
            a_prev = activations[i]
            grad_W[i] = a_prev.T @ delta
            grad_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_deriv(pre_act[i - 1])

        return grad_W, grad_b

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> "NumpyMLP":
        """
        Train the network using mini-batch SGD.

        Parameters
        ----------
        X_train, y_train : training features and labels
        X_val, y_val     : optional validation set
        epochs           : number of full passes over the training data
        batch_size       : mini-batch size

        Returns self for chaining.
        """
        n = X_train.shape[0]
        rng = np.random.default_rng(42)

        for epoch in range(epochs):
            idx = rng.permutation(n)
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]

            for start in range(0, n, batch_size):
                Xb = X_shuf[start: start + batch_size]
                yb = y_shuf[start: start + batch_size]

                activations, pre_act = self._forward(Xb)
                grad_W, grad_b = self._backward(activations, pre_act, yb)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * grad_W[i]
                    self.biases[i]  -= self.lr * grad_b[i]

            # Epoch loss
            probs_train = self.predict_proba(X_train)
            self.train_losses.append(cross_entropy_loss(probs_train, y_train))

            if X_val is not None and y_val is not None:
                probs_val = self.predict_proba(X_val)
                self.val_losses.append(cross_entropy_loss(probs_val, y_val))

        return self

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        """Save weights, biases, and training metadata to a .npz file."""
        arrays = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        arrays["train_losses"] = np.array(self.train_losses)
        arrays["val_losses"]   = np.array(self.val_losses)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path: Path, cfg: "ConfigB") -> "NumpyMLP":
        """Load a saved model from a .npz file."""
        data = np.load(str(path))
        hidden_sizes = cfg.NN_HIDDEN_SIZES
        net = cls(
            input_size=cfg.NN_INPUT_SIZE,
            hidden_sizes=hidden_sizes,
            output_size=cfg.NN_OUTPUT_SIZE,
            lr=cfg.NN_LEARNING_RATE,
            seed=cfg.NN_RANDOM_SEED,
        )
        n_layers = len(net.weights)
        for i in range(n_layers):
            net.weights[i] = data[f"W{i}"]
            net.biases[i]  = data[f"b{i}"]
        if "train_losses" in data:
            net.train_losses = list(data["train_losses"])
        if "val_losses" in data:
            net.val_losses = list(data["val_losses"])
        return net

    # ------------------------------------------------------------------ #
    #  Inference helper for controller                                     #
    # ------------------------------------------------------------------ #

    def safe_predict_action(
        self,
        features: list[float],
        obstacle_threshold: float = 0.3,
    ) -> int:
        """
        Predict the navigation action with a safety fallback.

        If the front obstacle distance (feature 0, normalised) indicates
        an imminent collision, override with STOP (3) or TURN_LEFT (1)
        regardless of network output.

        This ensures the trained policy cannot command the robot into an
        obstacle even if the network is uncertain.
        """
        front_dist_norm = features[0]
        if front_dist_norm < obstacle_threshold:
            left_dist_norm  = features[1]
            right_dist_norm = features[2]
            return 1 if right_dist_norm < left_dist_norm else 2

        x = np.array([features], dtype=float)
        return int(self.predict(x)[0])
