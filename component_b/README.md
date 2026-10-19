# Component B — Cooperative Swarm Robotics

## Overview

Three autonomous robots cooperate to visit target locations in a 12 m × 12 m arena.

## Key modules

| Module | Purpose |
|--------|---------|
| `src/configuration_b.py` | All swarm parameters |
| `src/agents.py` | AgentState and AgentMessage data structures |
| `src/communication.py` | Emitter/Receiver encoding + LocalBus for offline |
| `src/consensus.py` | Average consensus algorithm (ring topology) |
| `src/particle_swarm.py` | PSO for task allocation |
| `src/task_allocation.py` | PSO + greedy allocation + comparison |
| `src/neural_network.py` | NumpyMLP (pure NumPy, 5→[16,8]→4) |
| `src/training.py` | Balanced dataset generation + training pipeline |
| `src/evaluation.py` | Offline experiments, 6 figures, centralised vs distributed |

## Offline pipeline

```
python run_component_b.py
```

## Webots simulation

Open `worlds/component_b_world.wbt` in Webots R2025a and press Play.
All three robots start automatically using `swarm_controller.py`.

## Tests (55 total, all passing)

```
python -m pytest component_b/tests/ -v
```

## Model artifacts

- `models/numpy_mlp.npz` — NumpyMLP weights (95.5% test accuracy)
- `models/sklearn_mlp.pkl` — scikit-learn MLPClassifier (95.25%)
- `models/feature_scaler.json` — StandardScaler parameters
