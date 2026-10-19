"""
evaluation.py
=============
Offline evaluation and visualisation for Component B.

Functions
---------
run_centralised_vs_distributed  — compare decision strategies
generate_all_component_b_figures — produce and save all figures
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from configuration_b import ConfigB


# ------------------------------------------------------------------ #
#  Centralised vs distributed comparison                               #
# ------------------------------------------------------------------ #

def run_centralised_vs_distributed(cfg: "ConfigB") -> dict:
    """
    Compare centralised assignment (optimal greedy, full knowledge) vs
    distributed PSO-based assignment.

    Centralised: a single planner with full knowledge assigns robots
    optimally using the Hungarian-like greedy on all targets.

    Distributed: robots use PSO-assisted allocation with neighbour
    communication (as implemented in task_allocation.py).

    Both are evaluated on the same scenario (cfg.ROBOT_STARTS, cfg.TARGETS).
    """
    from task_allocation import greedy_allocation, allocation_cost, duplicate_count
    from particle_swarm import run_pso

    robot_pos  = list(cfg.ROBOT_STARTS)
    target_pos = list(cfg.TARGETS)

    # Centralised: sorted greedy with full knowledge
    cent_assign = greedy_allocation(robot_pos, target_pos)
    cent_cost   = allocation_cost(cent_assign, robot_pos, target_pos)
    cent_dups   = duplicate_count(cent_assign, len(target_pos))

    # Distributed: PSO
    pso_result = run_pso(robot_pos, target_pos, cfg)
    from task_allocation import allocation_cost as _ac
    dist_cost = _ac(pso_result.assignment, robot_pos, target_pos)
    dist_dups = pso_result.duplicate_count

    return {
        "centralised": {
            "assignment": cent_assign,
            "cost": cent_cost,
            "duplicate_count": cent_dups,
            "method": "greedy_full_knowledge",
        },
        "distributed": {
            "assignment": pso_result.assignment,
            "cost": dist_cost,
            "duplicate_count": dist_dups,
            "method": "pso_with_communication",
        },
        "robot_positions": robot_pos,
        "target_positions": target_pos,
    }


# ------------------------------------------------------------------ #
#  Figure generation                                                   #
# ------------------------------------------------------------------ #

def generate_all_component_b_figures(
    nn_results: dict,
    consensus_results: dict,
    alloc_results: dict,
    cd_results: dict,
    cfg: "ConfigB",
) -> None:
    """Generate and save all required Component B figures."""
    cfg.RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_METRICS.mkdir(parents=True, exist_ok=True)

    fig_b1_consensus(consensus_results, cfg)
    fig_b2_pso_fitness(alloc_results, cfg)
    fig_b3_task_allocation(alloc_results, cfg)
    fig_b4_nn_training_curve(nn_results, cfg)
    fig_b5_confusion_matrix(nn_results, cfg)
    fig_b6_centralised_vs_distributed(cd_results, cfg)

    save_component_b_metrics(nn_results, consensus_results, alloc_results, cd_results, cfg)


def fig_b1_consensus(results: dict, cfg: "ConfigB") -> None:
    """Figure B1: Agent consensus state over iterations."""
    histories = results["agent_histories"]
    errors    = results["errors"]
    true_avg  = results["true_average"]
    converged = results["iterations"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    colors = ["crimson", "seagreen", "royalblue"]
    for i, hist in enumerate(histories):
        ax.plot(hist, color=colors[i % len(colors)],
                linewidth=1.5, label=f"Agent {i+1}")
    ax.axhline(true_avg, color="black", linestyle="--", linewidth=1.0,
               label=f"True avg = {true_avg:.2f}")
    if results["converged"]:
        ax.axvline(converged, color="gray", linestyle=":", linewidth=1.0,
                   label=f"Converged at iter {converged}")
    ax.set_title("Figure B1a — Agent States (Consensus)", fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("State value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(errors, color="purple", linewidth=1.5)
    ax.axhline(cfg.CONSENSUS_EPSILON, color="red", linestyle="--",
               label=f"ε = {cfg.CONSENSUS_EPSILON}")
    if results["converged"]:
        ax.axvline(converged, color="gray", linestyle=":",
                   label=f"Converged at iter {converged}")
    ax.set_title("Figure B1b — Consensus Error (log scale)", fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative max deviation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure B1 — Average Consensus Experiment", fontsize=13)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig_b1_consensus.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig_b1_consensus.png saved")


def fig_b2_pso_fitness(alloc_results: dict, cfg: "ConfigB") -> None:
    """Figure B2: PSO fitness convergence over iterations."""
    history = alloc_results["pso"]["fitness_history"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color="darkorange", linewidth=1.5)
    ax.set_title("Figure B2 — PSO Fitness Convergence", fontsize=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (travel distance + penalty)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig_b2_pso_fitness.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig_b2_pso_fitness.png saved")


def fig_b3_task_allocation(alloc_results: dict, cfg: "ConfigB") -> None:
    """Figure B3: PSO vs greedy task allocation visualisation."""
    robot_pos  = alloc_results["robot_positions"]
    target_pos = alloc_results["target_positions"]
    pso_assign   = alloc_results["pso"]["assignment"]
    greed_assign = alloc_results["greedy"]["assignment"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    titles = ["PSO Allocation", "Greedy Allocation"]
    assignments = [pso_assign, greed_assign]
    costs = [alloc_results["pso"]["cost"], alloc_results["greedy"]["cost"]]
    colors_r = ["#e74c3c", "#27ae60", "#2980b9"]

    for ax, title, assign, cost in zip(axes, titles, assignments, costs):
        # Draw arena boundary
        half = cfg.WORLD_WIDTH / 2
        rect = plt.Rectangle((-half, -half), cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT,
                              fill=False, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)

        # Targets
        for t_idx, (tx, ty) in enumerate(target_pos):
            ax.plot(tx, ty, "y*", markersize=14, zorder=3)
            ax.annotate(f"T{t_idx+1}", (tx, ty), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

        # Robots + assignment arrows
        for r_idx, (rx, ry) in enumerate(robot_pos):
            ax.plot(rx, ry, "o", color=colors_r[r_idx], markersize=10, zorder=4)
            ax.annotate(f"R{r_idx+1}", (rx, ry), textcoords="offset points",
                        xytext=(5, -10), fontsize=8)
            t_idx = assign[r_idx]
            tx, ty = target_pos[t_idx]
            ax.annotate("", xy=(tx, ty), xytext=(rx, ry),
                        arrowprops=dict(arrowstyle="->",
                                        color=colors_r[r_idx], lw=1.5))

        ax.set_xlim(-half - 0.5, half + 0.5)
        ax.set_ylim(-half - 0.5, half + 0.5)
        ax.set_title(f"{title}\nTotal cost = {cost:.2f}m", fontsize=11)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle("Figure B3 — Task Allocation: PSO vs Greedy", fontsize=13)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig_b3_task_allocation.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig_b3_task_allocation.png saved")


def fig_b4_nn_training_curve(nn_results: dict, cfg: "ConfigB") -> None:
    """Figure B4: Neural network training and validation loss curves."""
    train_losses = nn_results.get("train_losses", [])
    val_losses   = nn_results.get("val_losses",   [])

    if not train_losses:
        print("  [fig] fig_b4 — no training loss data, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train loss", color="steelblue", linewidth=1.5)
    if val_losses:
        ax.plot(val_losses, label="Val loss", color="darkorange",
                linewidth=1.5, linestyle="--")
    ax.set_title("Figure B4 — Neural Network Training Curve", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig_b4_nn_training_curve.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig_b4_nn_training_curve.png saved")


def fig_b5_confusion_matrix(nn_results: dict, cfg: "ConfigB") -> None:
    """Figure B5: Confusion matrix for neural network classifier."""
    cm = np.array(nn_results.get("confusion_matrix", []))
    if cm.size == 0:
        print("  [fig] fig_b5 — no confusion matrix data, skipping")
        return

    class_names = ["Forward", "Turn L", "Turn R", "Stop"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, str(cm[r, c]),
                    ha="center", va="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontsize=11)

    acc = nn_results.get("accuracy", 0.0)
    ax.set_title(f"Figure B5 — Confusion Matrix (NumpyMLP, acc={acc:.3f})", fontsize=11)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig_b5_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig_b5_confusion_matrix.png saved")


def fig_b6_centralised_vs_distributed(cd_results: dict, cfg: "ConfigB") -> None:
    """Figure B6: Bar chart comparing centralised vs distributed allocation cost."""
    cent_cost = cd_results["centralised"]["cost"]
    dist_cost = cd_results["distributed"]["cost"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Centralised\n(Greedy)", "Distributed\n(PSO)"],
        [cent_cost, dist_cost],
        color=["steelblue", "darkorange"],
        width=0.4,
    )
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.2f}m",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=11,
        )
    ax.set_title("Figure B6 — Centralised vs Distributed Allocation Cost", fontsize=12)
    ax.set_ylabel("Total travel distance (m)")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig_b6_centralised_vs_distributed.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig_b6_centralised_vs_distributed.png saved")


# ------------------------------------------------------------------ #
#  Metrics persistence                                                 #
# ------------------------------------------------------------------ #

def save_component_b_metrics(
    nn_results: dict,
    consensus_results: dict,
    alloc_results: dict,
    cd_results: dict,
    cfg: "ConfigB",
) -> None:
    """Save combined Component B metrics to CSV and JSON."""
    # CSV
    rows = [
        ["metric", "value", "unit"],
        ["nn_test_accuracy",          nn_results.get("accuracy", 0.0),       "fraction"],
        ["nn_sklearn_accuracy",       nn_results.get("sklearn_accuracy") or "N/A", "fraction"],
        ["nn_train_samples",          nn_results.get("n_train", 0),           "samples"],
        ["nn_test_samples",           nn_results.get("n_test", 0),            "samples"],
        ["consensus_converged",       int(consensus_results.get("converged", False)), "bool"],
        ["consensus_iterations",      consensus_results.get("iterations", 0), "iter"],
        ["consensus_final_error",     consensus_results.get("final_error", 0.0), "rel"],
        ["pso_allocation_cost",       alloc_results["pso"]["cost"],           "m"],
        ["greedy_allocation_cost",    alloc_results["greedy"]["cost"],        "m"],
        ["pso_duplicate_count",       alloc_results["pso"]["duplicate_count"], "count"],
        ["cent_cost",                 cd_results["centralised"]["cost"],      "m"],
        ["dist_cost",                 cd_results["distributed"]["cost"],      "m"],
    ]
    csv_path = cfg.RESULTS_METRICS / "component_b_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # JSON
    json_data = {
        "neural_network": {
            "numpy_mlp_accuracy": nn_results.get("accuracy", 0.0),
            "sklearn_accuracy":   nn_results.get("sklearn_accuracy"),
            "confusion_matrix":   nn_results.get("confusion_matrix"),
            "class_metrics":      nn_results.get("class_metrics"),
        },
        "consensus": {
            "converged":     consensus_results.get("converged"),
            "iterations":    consensus_results.get("iterations"),
            "final_error":   consensus_results.get("final_error"),
            "true_average":  consensus_results.get("true_average"),
            "init_values":   consensus_results.get("init_values"),
        },
        "task_allocation": {
            "pso": {
                "assignment":    alloc_results["pso"]["assignment"],
                "cost_m":        alloc_results["pso"]["cost"],
                "duplicates":    alloc_results["pso"]["duplicate_count"],
            },
            "greedy": {
                "assignment":    alloc_results["greedy"]["assignment"],
                "cost_m":        alloc_results["greedy"]["cost"],
                "duplicates":    alloc_results["greedy"]["duplicate_count"],
            },
        },
        "centralised_vs_distributed": {
            "centralised_cost_m": cd_results["centralised"]["cost"],
            "distributed_cost_m": cd_results["distributed"]["cost"],
        },
    }
    json_path = cfg.RESULTS_METRICS / "component_b_metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  [metrics] component_b_metrics.csv saved")
    print(f"  [metrics] component_b_metrics.json saved")
