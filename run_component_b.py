"""
run_component_b.py
==================
Offline experiment runner for Component B.

Executes without starting Webots:
  - Generates and saves a synthetic training dataset for the neural network.
  - Trains and evaluates the neural network classifier.
  - Runs average-consensus experiments and records convergence.
  - Runs PSO-assisted task allocation and compares with greedy baseline.
  - Compares centralised vs distributed decision-making.
  - Generates all required Matplotlib figures and saves metrics.

To run the actual Webots multi-robot simulation, open Webots and load:
  component_b/worlds/component_b_world.wbt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "component_b" / "src"))

from training import generate_training_data, train_and_evaluate_model
from consensus import run_consensus_experiment
from task_allocation import run_allocation_experiment
from evaluation import (
    run_centralised_vs_distributed,
    generate_all_component_b_figures,
)
from configuration_b import ConfigB


def main() -> None:
    cfg = ConfigB()
    print("[Component B] Starting offline experiment pipeline...")

    # --- Neural network ---
    print("[Component B] Generating training dataset...")
    dataset_path = generate_training_data(cfg)
    print(f"[Component B] Dataset saved: {dataset_path}")

    print("[Component B] Training and evaluating neural network...")
    nn_results = train_and_evaluate_model(cfg)
    print(f"[Component B] Test accuracy : {nn_results['accuracy']:.4f}")

    # --- Consensus ---
    print("[Component B] Running consensus experiment...")
    consensus_results = run_consensus_experiment(cfg)
    print(f"[Component B] Converged in {consensus_results['iterations']} iterations, "
          f"final error={consensus_results['final_error']:.6f}")

    # --- Task allocation ---
    print("[Component B] Running PSO vs greedy task allocation...")
    alloc_results = run_allocation_experiment(cfg)
    print(f"[Component B] PSO cost    : {alloc_results['pso']['cost']:.4f}")
    print(f"[Component B] Greedy cost : {alloc_results['greedy']['cost']:.4f}")

    # --- Centralised vs distributed ---
    print("[Component B] Comparing centralised vs distributed...")
    cd_results = run_centralised_vs_distributed(cfg)

    # --- Figures and metrics ---
    print("[Component B] Generating figures and saving metrics...")
    generate_all_component_b_figures(
        nn_results, consensus_results, alloc_results, cd_results, cfg
    )

    print("[Component B] Offline pipeline complete.")
    print(f"[Component B] Figures saved to : {ROOT / 'component_b' / 'results' / 'figures'}")
    print(f"[Component B] Metrics saved to : {ROOT / 'component_b' / 'results' / 'metrics'}")
    print(f"[Component B] Models  saved to : {ROOT / 'component_b' / 'models'}")


if __name__ == "__main__":
    main()
