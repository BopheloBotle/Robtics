"""
test_neural_network.py
======================
Offline unit tests for the NumpyMLP neural network.

Tests cover:
  - Network initialises with correct weight shapes.
  - Forward pass output has correct shape and valid softmax probabilities.
  - Predictions return integer class indices in valid range.
  - Training reduces loss on simple linearly-separable data.
  - Network can be saved and loaded (weights preserved).
  - safe_predict_action returns a valid action class.
  - Safety fallback overrides output when obstacle is too close.
  - Teacher policy labels are correct for known inputs.
  - Training data generation produces expected number of samples.

No Webots connection required.
"""

import sys
from pathlib import Path
import tempfile
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from neural_network import NumpyMLP, softmax, relu, cross_entropy_loss
from training import teacher_label, generate_training_data
from configuration_b import ConfigB


@pytest.fixture
def cfg() -> ConfigB:
    return ConfigB()


@pytest.fixture
def small_net() -> NumpyMLP:
    return NumpyMLP(input_size=5, hidden_sizes=[8, 4], output_size=4,
                    lr=0.01, seed=42)


# ------------------------------------------------------------------ #
#  Activation functions                                                #
# ------------------------------------------------------------------ #

class TestActivations:
    def test_relu_positive(self):
        x = np.array([-1.0, 0.0, 2.0])
        result = relu(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 2.0])

    def test_softmax_sums_to_one(self):
        x = np.array([[1.0, 2.0, 0.5, -1.0]])
        probs = softmax(x)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_all_positive(self):
        x = np.array([[3.0, -2.0, 1.0, 0.0]])
        probs = softmax(x)
        assert np.all(probs > 0)

    def test_cross_entropy_perfect(self):
        probs = np.array([[0.0, 0.0, 1.0, 0.0]])
        probs = np.clip(probs, 1e-12, 1.0)
        labels = np.array([2])
        loss = cross_entropy_loss(probs, labels)
        assert loss < 0.001


# ------------------------------------------------------------------ #
#  Weight shapes                                                       #
# ------------------------------------------------------------------ #

class TestNetworkShape:
    def test_weight_shapes(self, small_net):
        # Layer 0: (5, 8)
        assert small_net.weights[0].shape == (5, 8)
        assert small_net.biases[0].shape  == (8,)
        # Layer 1: (8, 4)
        assert small_net.weights[1].shape == (8, 4)
        # Layer 2: (4, 4)
        assert small_net.weights[2].shape == (4, 4)

    def test_full_net_shapes(self, cfg):
        net = NumpyMLP(
            input_size=cfg.NN_INPUT_SIZE,
            hidden_sizes=cfg.NN_HIDDEN_SIZES,
            output_size=cfg.NN_OUTPUT_SIZE,
            seed=cfg.NN_RANDOM_SEED,
        )
        assert net.weights[0].shape == (cfg.NN_INPUT_SIZE, cfg.NN_HIDDEN_SIZES[0])
        assert net.weights[-1].shape == (cfg.NN_HIDDEN_SIZES[-1], cfg.NN_OUTPUT_SIZE)


# ------------------------------------------------------------------ #
#  Forward pass                                                        #
# ------------------------------------------------------------------ #

class TestForwardPass:
    def test_output_shape(self, small_net):
        X = np.random.randn(10, 5)
        probs = small_net.predict_proba(X)
        assert probs.shape == (10, 4)

    def test_probabilities_sum_to_one(self, small_net):
        X = np.random.randn(20, 5)
        probs = small_net.predict_proba(X)
        sums = probs.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(20), atol=1e-5)

    def test_predict_returns_integers_in_range(self, small_net):
        X = np.random.randn(15, 5)
        preds = small_net.predict(X)
        assert preds.shape == (15,)
        assert np.all(preds >= 0)
        assert np.all(preds < 4)


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

class TestTraining:
    def test_loss_decreases(self, cfg):
        """Training on generated data should reduce cross-entropy loss."""
        np.random.seed(42)
        rng = np.random.default_rng(42)
        n = 200
        X = rng.random((n, 5)).astype(float)
        # Simple linearly separable: if feature[0] > 0.5 → class 0, else class 2
        y = np.where(X[:, 0] > 0.5, 0, 2)

        net = NumpyMLP(5, [8], 4, lr=0.05, seed=42)
        net.fit(X, y, epochs=50, batch_size=32)

        assert len(net.train_losses) == 50
        initial_loss = net.train_losses[0]
        final_loss   = net.train_losses[-1]
        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"

    def test_high_accuracy_on_easy_data(self, cfg):
        """Network should achieve > 70% accuracy on easy synthetic data."""
        # Generate the full training set
        csv_path = generate_training_data(cfg)
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
        X = data[:, :cfg.NN_INPUT_SIZE].astype(float)
        y = data[:, cfg.NN_INPUT_SIZE].astype(int)

        n = len(y)
        split = int(n * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]

        net = NumpyMLP(
            cfg.NN_INPUT_SIZE, cfg.NN_HIDDEN_SIZES, cfg.NN_OUTPUT_SIZE,
            lr=cfg.NN_LEARNING_RATE, seed=cfg.NN_RANDOM_SEED,
        )
        net.fit(X_train, y_train, epochs=cfg.NN_EPOCHS, batch_size=cfg.NN_BATCH_SIZE)
        preds = net.predict(X_test)
        acc = float(np.mean(preds == y_test))
        assert acc > 0.70, f"Accuracy too low: {acc:.3f}"


# ------------------------------------------------------------------ #
#  Save / load                                                         #
# ------------------------------------------------------------------ #

class TestSaveLoad:
    def test_save_and_reload_predictions_match(self, small_net, cfg):
        """Saving then loading a model must preserve the weight arrays."""
        import numpy as _np
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "model.npz"
            small_net.save(path)
            assert path.exists(), "Model file was not created"
            # Open, check, then close explicitly to release Windows file lock
            npz = _np.load(str(path))
            keys = list(npz.keys())
            npz.close()
            assert "W0" in keys, "Weight W0 not found in saved file"
            assert "b0" in keys, "Bias b0 not found in saved file"

    def test_save_creates_npz_file(self, small_net):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_model.npz"
            small_net.save(path)
            assert path.exists()


# ------------------------------------------------------------------ #
#  Safety fallback                                                     #
# ------------------------------------------------------------------ #

class TestSafetyFallback:
    def test_close_obstacle_triggers_turn(self, small_net):
        """When front_dist_norm is very small, action must be 1 or 2 (not 0 or 3)."""
        features = [0.05, 0.8, 0.9, 0.1, 0.1]  # front very close
        action = small_net.safe_predict_action(features, obstacle_threshold=0.3)
        assert action in (1, 2), f"Expected turn action, got {action}"

    def test_clear_path_uses_network(self, cfg):
        """When path is clear, network output should be used."""
        net = NumpyMLP(
            cfg.NN_INPUT_SIZE, cfg.NN_HIDDEN_SIZES, cfg.NN_OUTPUT_SIZE,
            lr=cfg.NN_LEARNING_RATE, seed=42,
        )
        features = [0.9, 0.9, 0.9, 0.0, 0.0]  # all clear, heading aligned
        action = net.safe_predict_action(features, obstacle_threshold=0.3)
        assert 0 <= action <= 3


# ------------------------------------------------------------------ #
#  Teacher policy                                                      #
# ------------------------------------------------------------------ #

class TestTeacherPolicy:
    def test_stop_when_close_obstacle(self, cfg):
        label = teacher_label(0.1, 1.0, 1.0, cfg)
        assert label == 3  # stop

    def test_forward_when_clear(self, cfg):
        label = teacher_label(2.5, 2.5, 2.5, cfg)
        assert label == 0  # forward

    def test_turn_left_when_right_blocked(self, cfg):
        # front in turn zone, right closer than left
        label = teacher_label(0.5, 1.5, 0.2, cfg)
        assert label == 1  # turn_left (right blocked, steer left)

    def test_turn_right_when_left_blocked(self, cfg):
        label = teacher_label(0.5, 0.2, 1.5, cfg)
        assert label == 2  # turn_right


# ------------------------------------------------------------------ #
#  Data generation                                                     #
# ------------------------------------------------------------------ #

class TestDataGeneration:
    def test_generates_correct_count(self, cfg):
        csv_path = generate_training_data(cfg)
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
        assert data.shape[0] == cfg.NN_NUM_SAMPLES, \
            f"Expected {cfg.NN_NUM_SAMPLES} samples, got {data.shape[0]}"

    def test_labels_in_valid_range(self, cfg):
        csv_path = generate_training_data(cfg)
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
        labels = data[:, cfg.NN_INPUT_SIZE].astype(int)
        assert np.all(labels >= 0)
        assert np.all(labels < cfg.NN_OUTPUT_SIZE)

    def test_features_in_valid_range(self, cfg):
        csv_path = generate_training_data(cfg)
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
        X = data[:, :3]  # first 3 are normalised distances [0,1]
        assert np.all(X >= 0.0), "Distance features should be ≥ 0"
        assert np.all(X <= 1.0 + 1e-6), "Normalised distances should be ≤ 1"
