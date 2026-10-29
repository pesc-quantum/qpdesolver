"""
Publication-ready plotting routines for trainability studies and ansatz benchmarks.
"""
from typing import List, Dict, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_barren_plateau_scaling(
    records: List[Dict[str, Any]],
    save_path: Union[str, Path],
) -> None:
    """
    Plots gradient variance scaling vs number of qubits on logarithmic scale.
    Shows the exponential decay of global cost vs the polynomial mitigation of local cost.
    """
    # Group by ansatz
    ansatze = sorted(list(set(r["ansatz"] for r in records)))

    plt.figure(figsize=(9, 5), dpi=150)
    colors = ["#ef4444", "#3b82f6", "#10b981", "#8b5cf6"]

    for idx, ans in enumerate(ansatze):
        ans_recs = [r for r in records if r["ansatz"] == ans]
        ans_recs = sorted(ans_recs, key=lambda x: x["n_qubits"])
        qubits = [r["n_qubits"] for r in ans_recs]
        var_glob = [r["var_grad_global"] for r in ans_recs]
        var_loc = [r["var_grad_local"] for r in ans_recs]

        c = colors[idx % len(colors)]
        plt.plot(qubits, var_glob, "o--", color=c, label=f"{ans} (Global Cost)")
        plt.plot(qubits, var_loc, "s-", color=c, alpha=0.7, label=f"{ans} (Local Cost)")

    # Haar reference 2^{-n}
    all_q = sorted(list(set(r["n_qubits"] for r in records)))
    if all_q:
        haar_ref = [2.0**(-q) for q in all_q]
        plt.plot(all_q, haar_ref, "k:", label=r"Haar reference $2^{-n}$", linewidth=1.5)

    plt.yscale("log")
    plt.xlabel("Number of Qubits $n$", fontsize=12)
    plt.ylabel(r"Gradient Variance $\operatorname{Var}[\partial_{\theta} C]$", fontsize=12)
    plt.title("Barren Plateau Scaling: Global vs Local Cost Functions", fontsize=13, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_ansatz_comparison(
    records: List[Dict[str, Any]],
    save_path: Union[str, Path],
) -> None:
    """
    Bar plots comparing fidelity, residual, and circuit evaluations across ansatze.
    """
    ansatze = [r["ansatz"] for r in records]
    fidelities = [r["fidelity"] * 100 for r in records]
    residuals = [r["residual"] for r in records]

    x = np.arange(len(ansatze))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Fidelity bar
    ax1.bar(x, fidelities, color="#3b82f6", width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ansatze, rotation=15)
    ax1.set_ylabel("Solution Fidelity (%)", fontsize=11)
    ax1.set_title("Reconstruction Fidelity", fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis="y", alpha=0.3)

    # Residual bar
    ax2.bar(x, residuals, color="#10b981", width=0.5, edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ansatze, rotation=15)
    ax2.set_yscale("log")
    ax2.set_ylabel("Relative Residual (log scale)", fontsize=11)
    ax2.set_title("Equation Residual $\|A x - b\| / \|b\|$", fontweight="bold")
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
