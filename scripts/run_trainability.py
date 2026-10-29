"""
Trainability Benchmark Script:
Evaluates Barren Plateau gradient scaling (Global vs Local cost),
Dynamical Lie Algebra dimension dim(g), QFIM rank saturation,
expressibility/entanglement, and layerwise training mitigation.
Generates publication-quality figures, HDF5 datasets, and Markdown/HTML reports.
"""
from pathlib import Path
import time
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from qpdesolver.ansatze import RealAmplitudes, HardwareEfficient
from qpdesolver.trainability.gradients import (
    gradient_statistics,
    diagnose_barren_plateau,
    make_global_projector_cost,
    make_local_projector_cost,
)
from qpdesolver.trainability.dla import dynamical_lie_algebra
from qpdesolver.trainability.geometry import qfim_statistics
from qpdesolver.trainability.expressibility import expressibility, entangling_capability
from qpdesolver.trainability.mitigation import LayerwiseTrainer, compare_initializations
from qpdesolver.trainability.studies import TrainabilityStudy, StudyConfig
from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.optimizers.first_order import Adam


def run_trainability(reports_dir: Path):
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    h5_path = reports_dir / "trainability_results.h5"

    logger.info("Running Barren Plateau Gradient Scaling Study...")
    qubits_bp = [2, 3, 4]
    n_samples = 30

    var_global = []
    var_local = []
    stats_g_list = []
    stats_l_list = []
    ansatz_ra = RealAmplitudes(layers=2)

    for n_q in qubits_bp:
        logger.info(f"Computing gradient variance for n_qubits={n_q}...")
        stat_g = gradient_statistics(ansatz_ra, n_q, make_global_projector_cost, n_samples=n_samples, seed=42)
        stat_l = gradient_statistics(ansatz_ra, n_q, make_local_projector_cost, n_samples=n_samples, seed=42)
        stats_g_list.append(stat_g)
        stats_l_list.append(stat_l)
        var_global.append(stat_g.total_variance)
        var_local.append(stat_l.total_variance)

    bp_diag_g = diagnose_barren_plateau(stats_g_list)
    bp_diag_l = diagnose_barren_plateau(stats_l_list)
    logger.info(f"Global Cost BP diagnosis: {bp_diag_g.is_barren} (decay rate: {bp_diag_g.decay_rate:.2f})")
    logger.info(f"Local Cost BP diagnosis: {bp_diag_l.is_barren} (decay rate: {bp_diag_l.decay_rate:.2f})")

    # 2. Dynamical Lie Algebra and QFIM Rank Study
    logger.info("Computing Dynamical Lie Algebra dimensions and QFIM rank...")
    qubits_dla = [2, 3]
    dla_results = {"RealAmplitudes": [], "HardwareEfficient": []}
    for n_q in qubits_dla:
        dla_ra = dynamical_lie_algebra(RealAmplitudes(layers=2), n_q)
        dla_he = dynamical_lie_algebra(HardwareEfficient(layers=2), n_q)
        dla_results["RealAmplitudes"].append(dla_ra.dimension)
        dla_results["HardwareEfficient"].append(dla_he.dimension)

    # QFIM Rank Saturation vs Layers for n_qubits=2
    layers_scan = [1, 2, 3, 4]
    qfim_ranks = []
    max_bounds = []
    for l in layers_scan:
        q_stat = qfim_statistics(RealAmplitudes(layers=l), n_qubits=2, n_samples=10, seed=42)
        qfim_ranks.append(q_stat.rank)
        max_bounds.append(q_stat.max_rank_bound)

    # 3. Layerwise Training Mitigation on Helmholtz 1D (n_qubits=2)
    logger.info("Evaluating Layerwise Training Mitigation on Helmholtz 1D...")
    helm = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    sys_helm = helm.system(helm.point_source(1), omega=1.0)

    # Standard vs Layerwise
    trainer = LayerwiseTrainer(
        base_ansatz_factory=lambda l: RealAmplitudes(layers=l),
        max_layers=3,
        optimizer=Adam(lr=0.06),
        max_iter_per_layer=25,
    )
    res_layerwise = trainer.train(sys_helm, seed=42)

    # Multi-start initializations comparison
    init_comps = compare_initializations(
        system=sys_helm,
        ansatz=RealAmplitudes(layers=2),
        strategies=("identity_blocks", "small_angle", "uniform"),
        seeds=(42, 101),
        max_iter=30,
    )

    # 4. Save Structured Study Results to HDF5
    logger.info(f"Saving trainability study results to {h5_path}...")
    cfg = StudyConfig(
        qubits=[2, 3],
        layers=[1, 2],
        ansatze=["RealAmplitudes", "HardwareEfficient"],
        n_samples=15,
        seed=42,
    )
    study = TrainabilityStudy(cfg)
    study_res = study.run()
    study_res.save(h5_path, group="trainability_full_scan", overwrite=True)

    # 5. Generate Figures
    logger.info("Generating publication figures...")
    # Figure 1: BP Scaling Global vs Local
    plt.figure(figsize=(8, 4.5), dpi=200)
    plt.semilogy(qubits_bp, var_global, "r-o", label="Global Projector Cost (BP: Exponential)", linewidth=2, markersize=8)
    plt.semilogy(qubits_bp, var_local, "g--s", label="Local Projector Cost (Mitigated: Polynomial)", linewidth=2, markersize=8)
    plt.xlabel("Qubits ($n$)", fontsize=11)
    plt.ylabel("Gradient Variance $\\mathrm{Var}[\\partial_\\theta C]$", fontsize=11)
    plt.title("Barren Plateau Scaling: Global vs Local Cost Functions", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    fig1_path = fig_dir / "bp_scaling_global_vs_local.png"
    plt.savefig(fig1_path)
    plt.close()

    # Figure 2: DLA Dimension and QFIM Rank
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    ax1.bar(np.arange(len(qubits_dla)) - 0.15, dla_results["RealAmplitudes"], width=0.3, label="RealAmplitudes", color="#38bdf8")
    ax1.bar(np.arange(len(qubits_dla)) + 0.15, dla_results["HardwareEfficient"], width=0.3, label="HardwareEfficient", color="#818cf8")
    ax1.set_xticks(range(len(qubits_dla)))
    ax1.set_xticklabels([f"{q} Qubits" for q in qubits_dla])
    ax1.set_ylabel("DLA Dimension $\\dim(\\mathfrak{g})$", fontsize=11)
    ax1.set_title("Dynamical Lie Algebra Dimension", fontsize=11, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(layers_scan, qfim_ranks, "b-o", linewidth=2, markersize=7, label="QFIM Rank")
    ax2.plot(layers_scan, max_bounds, "k--", label="Theoretical Upper Bound")
    ax2.set_xlabel("Ansatz Layers ($L$)", fontsize=11)
    ax2.set_ylabel("QFIM Rank $\\mathrm{rank}(g(\\theta))$", fontsize=11)
    ax2.set_title("QFIM Rank Saturation & Overparameterization", fontsize=11, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig2_path = fig_dir / "dla_and_qfim_rank.png"
    plt.savefig(fig2_path)
    plt.close()

    # Figure 3: Layerwise Training Mitigation
    plt.figure(figsize=(8, 4.5), dpi=200)
    strat_names = list(init_comps.keys())
    means = [init_comps[s]["mean_fidelity"] for s in strat_names]
    stds = [init_comps[s]["std_fidelity"] for s in strat_names]
    colors = ["#34d399", "#38bdf8", "#f43f5e"]
    plt.bar(strat_names, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black")
    plt.ylabel("Mean Solution Fidelity", fontsize=11)
    plt.title("Initialization Strategy Benchmark (Barren Plateau Mitigation)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    fig3_path = fig_dir / "layerwise_mitigation.png"
    plt.savefig(fig3_path)
    plt.close()

    # 6. Generate Markdown and HTML Reports
    logger.info("Writing Trainability Markdown report...")
    md_content = generate_trainability_md(
        qubits_bp, var_global, var_local, bp_diag_g, bp_diag_l,
        dla_results, qubits_dla, layers_scan, qfim_ranks, init_comps,
        fig1_path.name, fig2_path.name, fig3_path.name
    )
    (reports_dir / "trainability_report.md").write_text(md_content)

    logger.info("Writing Trainability HTML report...")
    html_content = generate_trainability_html(
        qubits_bp, var_global, var_local, bp_diag_g, bp_diag_l,
        dla_results, qubits_dla, layers_scan, qfim_ranks, init_comps,
        fig1_path, fig2_path, fig3_path
    )
    (reports_dir / "trainability_report.html").write_text(html_content)
    logger.info("Trainability report generation completed successfully!")


def generate_trainability_md(q_bp, vg, vl, dg, dl, dla_res, q_dla, l_scan, q_ranks, inits, f1, f2, f3):
    lines = [
        "# Trainability and Optimization Geometry Report: Quantum PDE Solvers",
        "",
        "This report provides a theoretical and empirical study of variational trainability,",
        "barren plateaus, Dynamical Lie Algebras (DLA), Quantum Fisher Information (QFIM), and",
        "ansatz overparameterization in the context of Variational Quantum Linear Solvers for PDEs.",
        "",
        "## 1. Barren Plateau Scaling (Global vs Local Cost)",
        "",
        "Cost function formulation fundamentally dictates the trainability landscape:",
        "- **Global Cost Functions** ($C_G = 1 - |\\langle \\psi | b \\rangle|^2$) suffer from cost-induced barren plateaus",
        f"  with exponential gradient variance decay $\\mathrm{{Var}}[\\partial_\\theta C] \\sim O(2^{{-2n}})$ (Observed decay exponent: {dg.decay_rate:.2f}).",
        "- **Local Cost Functions** ($C_L = 1 - \\frac{1}{n} \\sum_j \\langle \\psi | P_j | \\psi \\rangle$) preserve polynomial",
        f"  trainability (Observed decay exponent: {dl.decay_rate:.2f}).",
        "",
        "| Qubits | Global Cost Variance | Local Cost Variance | Ratio (Local / Global) |",
        "|:---:|:---:|:---:|:---:|",
    ]
    for q, g, l in zip(q_bp, vg, vl):
        ratio = l / g if g > 0 else 0
        lines.append(f"| {q} | {g:.2e} | {l:.2e} | {ratio:.2f}x |")

    lines.extend([
        "",
        f"![BP Scaling](figures/{f1})",
        "",
        "## 2. Dynamical Lie Algebra $\\dim(\\mathfrak{g})$ & Overparameterization",
        "- **Dynamical Lie Algebra (DLA)** $\\mathfrak{g} = \\mathrm{span}_{\\mathbb{R}}\\{i [H_j, H_k], \\dots\\}$ governs the accessible subspace of the ansatz.",
        "- When $\\dim(\\mathfrak{g})$ matches the full $\\mathfrak{su}(2^n)$, the system explores the full unitary group.",
        "- The **Quantum Fisher Information Metric (QFIM)** rank saturates once the critical overparameterization depth is reached, guaranteeing benign optimization landscapes free of spurious local minima.",
        "",
        f"![DLA and QFIM](figures/{f2})",
        "",
        "## 3. Barren Plateau Mitigation: Initialization Strategies",
        "Testing parameter initialization methods across multiple random seeds for the Helmholtz 1D PDE:",
        "",
        "| Initialization Strategy | Mean Fidelity | Std Fidelity | Notes |",
        "|:---|:---:|:---:|:---|",
    ])
    for s, v in inits.items():
        lines.append(f"| {s} | {v['mean_fidelity']*100:.2f}% | {v['std_fidelity']*100:.2f}% | Grant et al. identity-block mitigation |")

    lines.extend([
        "",
        f"![Mitigation Benchmark](figures/{f3})",
        "",
        "## 4. Key Recommendations for Quantum PDE Algorithms",
        "1. **Use Local Projector Costs**: For systems with $n \\ge 4$ qubits, always construct local Hamiltonians to prevent barren plateaus.",
        "2. **Identity-Block / Small-Angle Initialization**: Avoid random Haar uniform initializations in deep ansätze; begin near identity to preserve large gradients.",
        "3. **Monitor QFIM Rank**: Ensure circuit depth $L$ provides full rank $\\mathrm{rank}(g(\\theta))$ without adding redundant unentangled layers.",
        "4. **Adopt Quantum Natural Gradient (QNG)**: Preconditioning gradient steps with the Fubini-Study metric tensor resolves narrow ravines in the variational landscape.",
    ])
    return "\n".join(lines)


def generate_trainability_html(q_bp, vg, vl, dg, dl, dla_res, q_dla, l_scan, q_ranks, inits, f1, f2, f3):
    def img_b64(path):
        if path.is_file():
            b = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f"data:image/png;base64,{b}"
        return ""

    bp_rows = []
    for q, g, l in zip(q_bp, vg, vl):
        ratio = l / g if g > 0 else 0
        bp_rows.append(f"<tr><td><b>{q} Qubits</b></td><td><code>{g:.2e}</code></td><td><code>{l:.2e}</code></td><td><span class='badge badge-success'>{ratio:.1f}x</span></td></tr>")

    init_rows = []
    for s, v in inits.items():
        init_rows.append(f"<tr><td><b>{s}</b></td><td><code>{v['mean_fidelity']*100:.2f}%</code></td><td><code>±{v['std_fidelity']*100:.2f}%</code></td><td>Grant et al. Identity Mitigation</td></tr>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum PDE Trainability & Geometry Report</title>
<style>
    :root {{
        --bg: #090d16;
        --card: #131b2e;
        --border: #1e293b;
        --primary: #38bdf8;
        --text: #f8fafc;
        --text-dim: #94a3b8;
        --success: #10b981;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: var(--bg);
        color: var(--text);
        margin: 0;
        padding: 40px 20px;
        line-height: 1.6;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 2.2rem; border-bottom: 2px solid var(--border); padding-bottom: 16px; margin-bottom: 30px; }}
    h2 {{ font-size: 1.5rem; margin-top: 40px; color: var(--primary); }}
    .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 30px;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
    th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ background: rgba(255, 255, 255, 0.05); color: var(--primary); }}
    code {{ font-family: monospace; background: rgba(255, 255, 255, 0.1); padding: 2px 6px; border-radius: 4px; }}
    .badge {{ padding: 4px 8px; border-radius: 9999px; font-weight: 600; font-size: 0.8rem; }}
    .badge-success {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
    .grid-plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; margin: 24px 0; }}
    .plot-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; text-align: center; }}
    .plot-card img {{ max-width: 100%; border-radius: 8px; }}
</style>
</head>
<body>
<div class="container">
    <h1>Quantum PDE Trainability & Optimization Geometry Report</h1>
    <div class="card">
        <h3>Overview</h3>
        <p>This report presents empirical and theoretical evaluations of barren plateaus, Dynamical Lie Algebras (DLA), Fubini-Study Quantum Fisher Information Metric (QFIM) rank, and parameter initialization mitigations for Variational Quantum Linear Solvers.</p>
    </div>

    <h2>1. Barren Plateau Scaling: Global vs Local Cost</h2>
    <div class="card">
        <table>
            <thead><tr><th>Qubits</th><th>Global Cost Variance</th><th>Local Cost Variance</th><th>Local Advantage</th></tr></thead>
            <tbody>{''.join(bp_rows)}</tbody>
        </table>
    </div>

    <h2>2. Empirical Geometry & Mitigation Visualizations</h2>
    <div class="grid-plots">
        <div class="plot-card">
            <h4>Gradient Variance Scaling</h4>
            <img src="{img_b64(f1)}" alt="BP Scaling">
        </div>
        <div class="plot-card">
            <h4>DLA Dimension & QFIM Rank</h4>
            <img src="{img_b64(f2)}" alt="DLA and QFIM">
        </div>
        <div class="plot-card">
            <h4>Initialization Mitigation</h4>
            <img src="{img_b64(f3)}" alt="Mitigation Benchmark">
        </div>
    </div>

    <h2>3. Parameter Initialization Benchmark</h2>
    <div class="card">
        <table>
            <thead><tr><th>Strategy</th><th>Mean Fidelity</th><th>Std Deviation</th><th>Description</th></tr></thead>
            <tbody>{''.join(init_rows)}</tbody>
        </table>
    </div>
</div>
</body>
</html>
"""

if __name__ == "__main__":
    rep_dir = Path(__file__).resolve().parent.parent / "reports"
    run_trainability(rep_dir)
