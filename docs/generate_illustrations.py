"""
Generates publication-quality pedagogical illustrations and schematics
for the qpdesolver academic documentation and guides.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    })

def make_ansatz_circuit_diagram(out_path: Path):
    """Illustration 1: Visual circuit diagram of Chained Ry-Rz-CZ ansatz."""
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=220)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.8, 3.8)
    ax.axis("off")

    qubits = [3, 2, 1, 0]
    q_labels = [r"$|q_0\rangle$", r"$|q_1\rangle$", r"$|q_2\rangle$", r"$|q_3\rangle$"]

    # Draw horizontal wire lines
    for idx, y in enumerate(qubits):
        ax.plot([0.5, 9.0], [y, y], color="#475569", lw=2, zorder=1)
        ax.text(0.1, y, q_labels[idx], va="center", ha="right", fontsize=11, fontweight="bold", color="#0f172a")

    # Helper for gate boxes
    def draw_gate(x, y, label, color="#38bdf8", text_color="#0f172a", w=0.6, h=0.5):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor="#0284c7", lw=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color=text_color, zorder=4)

    # Helper for CZ connection
    def draw_cz(x, y1, y2, color="#818cf8"):
        ax.plot([x, x], [y1, y2], color=color, lw=2.5, zorder=2)
        ax.plot(x, y1, "o", color=color, markersize=7, zorder=3)
        ax.plot(x, y2, "o", color=color, markersize=7, zorder=3)

    # 1. Initial Rotation Layer (x=1.3 and x=2.1)
    for idx, y in enumerate(qubits):
        draw_gate(1.2, y, r"$R_y(\theta_{0," + str(idx) + r"})$", color="#bae6fd")
        draw_gate(2.1, y, r"$R_z(\phi_{0," + str(idx) + r"})$", color="#bae6fd")

    # Layer 0 separator
    ax.axvline(2.7, color="#cbd5e1", linestyle="--", lw=1.2)
    ax.text(1.65, 3.5, "Initial Rotations", ha="center", fontsize=9.5, fontweight="bold", color="#0369a1")

    # 2. Entangling Block 1 (x=3.3 to x=4.5)
    draw_cz(3.3, 3, 2, color="#6366f1")
    draw_cz(3.7, 2, 1, color="#6366f1")
    draw_cz(4.1, 1, 0, color="#6366f1")
    # Circular wrap (q3 to q0)
    ax.plot([4.5, 4.5], [3, 0], color="#6366f1", lw=2, linestyle=":", zorder=2)
    ax.plot(4.5, 3, "o", color="#6366f1", markersize=7, zorder=3)
    ax.plot(4.5, 0, "o", color="#6366f1", markersize=7, zorder=3)
    ax.text(3.9, 3.5, "Circular CZ Entanglement", ha="center", fontsize=9.5, fontweight="bold", color="#4338ca")

    # 3. Layer 1 Rotation Block (x=5.3, 6.1)
    for idx, y in enumerate(qubits):
        draw_gate(5.3, y, r"$R_y(\theta_{1," + str(idx) + r"})$", color="#ddd6fe")
        draw_gate(6.1, y, r"$R_z(\phi_{1," + str(idx) + r"})$", color="#ddd6fe")
    ax.axvline(6.7, color="#cbd5e1", linestyle="--", lw=1.2)
    ax.text(5.7, 3.5, "Layer 1 Rotations", ha="center", fontsize=9.5, fontweight="bold", color="#6d28d9")

    # 4. Repeating Layer 2 indicator (x=7.5 to 8.5)
    bracket = FancyBboxPatch((7.0, -0.4), 1.8, 3.8, boxstyle="round,pad=0.1",
                             facecolor="#f8fafc", edgecolor="#94a3b8", lw=1.5, linestyle="--", zorder=0)
    ax.add_patch(bracket)
    ax.text(7.9, 1.5, "Repeating\nLayers\n($L = 2, 3, \\dots$)", ha="center", va="center", fontsize=9.5, color="#475569")

    ax.set_title("Chained $R_y(\\theta) - R_z(\\phi) - \\mathrm{CZ}$ Quantum Circuit Architecture", pad=15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_pde_stencil_diagram(out_path: Path):
    """Illustration 2: 1D Finite-Difference Grid and Stencil Diagram."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=220)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axis("off")

    # Grid nodes: x0, x1, x2, x3, x4
    node_x = [0.8, 2.0, 3.2, 4.4, 5.6]
    labels = [r"$x_{j-2}$", r"$x_{j-1}$", r"$x_j$", r"$x_{j+1}$", r"$x_{j+2}$"]

    # Base line
    ax.plot([0.3, 6.1], [0, 0], color="#94a3b8", lw=3, zorder=1)

    for i, x in enumerate(node_x):
        # Center node highlight
        color = "#ef4444" if i == 2 else ("#3b82f6" if i in (1, 3) else "#64748b")
        size = 14 if i == 2 else (11 if i in (1, 3) else 8)
        ax.plot(x, 0, "o", color=color, markersize=size, zorder=3)
        ax.text(x, -0.45, labels[i], ha="center", va="top", fontsize=11, fontweight="bold", color="#0f172a")

    # Grid spacing h
    ax.annotate("", xy=(3.2, -0.9), xytext=(2.0, -0.9),
                arrowprops=dict(arrowstyle="<->", color="#0f172a", lw=1.5))
    ax.text(2.6, -1.2, r"$\Delta x = h$", ha="center", fontsize=10, fontweight="bold")

    # Stencil Arcs
    # Diffusion coupling (-nu / h^2)
    style = "arc3,rad=0.35"
    ax.annotate(r"$\mathrm{Diffusion}: -\frac{\nu}{h^2}$", xy=(2.0, 0.15), xytext=(3.2, 0.15),
                arrowprops=dict(arrowstyle="<->", connectionstyle=style, color="#2563eb", lw=2),
                ha="center", fontsize=9.5, color="#1e40af", fontweight="bold")
    ax.annotate(r"$\mathrm{Diffusion}: -\frac{\nu}{h^2}$", xy=(3.2, 0.15), xytext=(4.4, 0.15),
                arrowprops=dict(arrowstyle="<->", connectionstyle=style, color="#2563eb", lw=2),
                ha="center", fontsize=9.5, color="#1e40af", fontweight="bold")

    # Advection coupling (c / 2h)
    style_adv = "arc3,rad=-0.4"
    ax.annotate(r"$\mathrm{Advection}: \frac{c}{2h}$", xy=(4.4, -0.15), xytext=(2.0, -0.15),
                arrowprops=dict(arrowstyle="->", connectionstyle=style_adv, color="#059669", lw=2),
                ha="center", fontsize=9.5, color="#065f46", fontweight="bold")

    # Diagonal Reaction & Diffusion term
    ax.annotate(r"$\mathrm{Diagonal}: \frac{2\nu}{h^2} + \sigma$", xy=(3.2, 0.35), xytext=(3.2, 1.4),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2),
                ha="center", fontsize=10, color="#991b1b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#f87171", lw=1.2))

    ax.set_title(r"1D Advection-Diffusion-Reaction Finite-Difference Spatial Stencil", pad=15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_fwi_workflow_diagram(out_path: Path):
    """Illustration 3: Complete Full Waveform Inversion Adjoint Workflow."""
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=220)
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle="", color="#e0f2fe", border="#0284c7"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             fc=color, ec=border, lw=1.6, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a", zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h*0.3, subtitle, ha="center", va="center", fontsize=8.5, color="#334155", zorder=3)

    def draw_arrow(x1, y1, x2, y2, label="", color="#475569"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0), zorder=4)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.18, label, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color, zorder=5)

    # 1. Subsurface Velocity Model
    draw_box(0.2, 2.6, 2.2, 1.2, "Velocity Model v(x)", "m(x) = 1 / v(x)^2", color="#fef3c7", border="#d97706")

    # 2. Forward Modeling
    draw_box(3.2, 2.6, 2.4, 1.2, "Forward Helmholtz", "A(m) u_s = b_s\n(HHL / VQLS / LU)", color="#dbeafe", border="#2563eb")
    draw_arrow(2.4, 3.2, 3.2, 3.2, "Assemble A(m)")

    # 3. Synthetic Data & Residual
    draw_box(6.4, 2.6, 2.4, 1.2, "Receiver Data & Residual", r"$\delta d = R u_s - d^{\mathrm{obs}}$", color="#e0e7ff", border="#4f46e5")
    draw_arrow(5.6, 3.2, 6.4, 3.2, "Project R u")

    # 4. Adjoint Helmholtz Equation
    draw_box(6.4, 0.4, 2.4, 1.2, "Adjoint Helmholtz", r"$A(m) \lambda_s = R^T \delta d$", color="#fce7f3", border="#db2777")
    draw_arrow(7.6, 2.6, 7.6, 1.6, r"Backproject $R^T \delta d$")

    # 5. Adjoint Gradient Assembly
    draw_box(3.2, 0.4, 2.4, 1.2, "Gradient Calculation", r"$\nabla_m J = \sum \omega^2 \mathrm{Re}(\lambda^* u)$", color="#fee2e2", border="#dc2626")
    draw_arrow(6.4, 1.0, 5.6, 1.0, "Correlate with u_s")

    # 6. Optimization Update
    draw_box(0.2, 0.4, 2.2, 1.2, "Model Optimization", "L-BFGS-B / Adam\nv_{k+1} = v_k - \\alpha g", color="#dcfce7", border="#16a34a")
    draw_arrow(3.2, 1.0, 2.4, 1.0, r"$\nabla_v J$ via Chain Rule")

    # Closing loop back to model
    draw_arrow(1.3, 1.6, 1.3, 2.6, "Iterative Update", color="#16a34a")

    ax.set_title("Full Waveform Inversion (FWI) Adjoint-State Optimization Cycle", pad=15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_hhl_vs_vqls_schematic(out_path: Path):
    """Illustration 4: Comparison schematic between HHL and VQLS."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=220)

    # Subplot 1: HHL (Fault-Tolerant)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis("off")
    ax1.set_title("Harrow-Hassidim-Lloyd (HHL)\nFault-Tolerant Unitary Algorithm", fontsize=11, fontweight="bold", color="#1e3a8a")

    def card1(y, label, desc, bg="#dbeafe", bd="#2563eb"):
        b = FancyBboxPatch((0.8, y), 8.4, 1.1, boxstyle="round,pad=0.08", fc=bg, ec=bd, lw=1.5)
        ax1.add_patch(b)
        ax1.text(5.0, y + 0.65, label, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")
        ax1.text(5.0, y + 0.28, desc, ha="center", va="center", fontsize=8.2, color="#334155")

    card1(6.2, "1. State Preparation", r"Initialize target register $|b\rangle = \sum \beta_j |u_j\rangle$", "#f1f5f9", "#64748b")
    card1(4.8, "2. Quantum Phase Estimation (QPE)", r"Simulate $e^{i A t}$ on $n_{\mathrm{clock}}$ qubits $\rightarrow \sum \beta_j |u_j\rangle |\tilde{\lambda}_j\rangle$", "#e0f2fe", "#0284c7")
    card1(3.4, "3. Eigenvalue Inversion", r"Rotate flag ancilla: $R_y(2 \arcsin(C / \tilde{\lambda}_j))$", "#fef3c7", "#d97706")
    card1(2.0, "4. Inverse QPE (Uncomputation)", r"Disentangle clock register: $\mathrm{QPE}^\dagger$", "#e0f2fe", "#0284c7")
    card1(0.6, "5. Ancilla Post-Selection", r"Measure flag $|1\rangle \Rightarrow |x\rangle \propto A^{-1}|b\rangle$", "#dcfce7", "#16a34a")

    for y in [6.2, 4.8, 3.4, 2.0]:
        ax1.annotate("", xy=(5.0, y - 0.25), xytext=(5.0, y),
                     arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5))

    # Subplot 2: VQLS (NISQ Variational)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis("off")
    ax2.set_title("Variational Quantum Linear Solver (VQLS)\nNISQ Parameterized Optimization", fontsize=11, fontweight="bold", color="#1e3a8a")

    def card2(x, y, w, h, label, desc, bg="#f3e8ff", bd="#9333ea"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=bg, ec=bd, lw=1.5)
        ax2.add_patch(b)
        ax2.text(x + w/2, y + h*0.65, label, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")
        ax2.text(x + w/2, y + h*0.3, desc, ha="center", va="center", fontsize=8.2, color="#334155")

    card2(1.0, 5.5, 8.0, 1.4, "Quantum Circuit Ansatz U(θ)", r"Generates candidate state $|\psi(\theta)\rangle = U(\theta)|0\rangle$", "#f3e8ff", "#9333ea")
    card2(1.0, 3.2, 8.0, 1.4, "Quantum Cost Evaluation C(θ)", "Global Cost C_G or Local Cost C_L\n(Overlap with |b> via Hadamard tests)", "#e0e7ff", "#4f46e5")
    card2(1.0, 0.8, 8.0, 1.4, "Classical / Riemannian Optimizer", "Adam, L-BFGS-B, or Quantum Natural Gradient\nUpdates θ_{k+1} = θ_k - η g^{-1} ∇C", "#dcfce7", "#16a34a")

    ax2.annotate("", xy=(5.0, 4.6), xytext=(5.0, 5.5),
                 arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8))
    ax2.annotate("", xy=(5.0, 2.2), xytext=(5.0, 3.2),
                 arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8))
    # Feedback loop arrow
    ax2.annotate("", xy=(1.0, 6.2), xytext=(1.0, 1.5),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", color="#16a34a", lw=2.0))
    ax2.text(0.1, 3.85, "Update\nParameters\n$\\theta$", ha="center", va="center", fontsize=9, fontweight="bold", color="#16a34a")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_quantum_geometry_diagram(out_path: Path):
    """Illustration 5: Quantum Geometry, Riemannian Manifold, and DLA Concept."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=220)

    # Panel 1: Euclidean vs Riemannian Optimization on State Manifold
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.3, 1.3)
    ax1.axis("off")
    ax1.set_title("Riemannian Optimization on State Manifold\n(Quantum Natural Gradient vs Euclidean)", fontsize=11, fontweight="bold")

    # Draw curved manifold surface
    u = np.linspace(-1.0, 1.0, 100)
    v = 0.5 * (1.0 - u**2) + 0.3
    ax1.plot(u, v, color="#64748b", lw=3.0, zorder=1)
    ax1.fill_between(u, -0.2, v, color="#f1f5f9", alpha=0.6, zorder=0)
    ax1.text(0.7, 0.1, r"Manifold $\mathcal{M} = \{ |\psi(\theta)\rangle \}$", fontsize=10, fontstyle="italic", color="#475569")

    # Current point theta_t
    pt_x, pt_y = -0.5, 0.5 * (1.0 - (-0.5)**2) + 0.3
    ax1.plot(pt_x, pt_y, "o", color="#0f172a", markersize=9, zorder=4)
    ax1.text(pt_x - 0.08, pt_y + 0.12, r"$\theta_t$", fontsize=11, fontweight="bold")

    # Euclidean gradient step (overshoots off the manifold)
    ax1.annotate("", xy=(0.0, pt_y + 0.35), xytext=(pt_x, pt_y),
                 arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.2, linestyle="--"), zorder=3)
    ax1.text(-0.05, pt_y + 0.42, r"Euclidean step $-\eta \nabla C$", color="#dc2626", fontsize=9.5, fontweight="bold")

    # Natural gradient step (follows the Riemannian curvature g_ij)
    ax1.annotate("", xy=(0.1, 0.5 * (1.0 - 0.1**2) + 0.3), xytext=(pt_x, pt_y),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#16a34a", lw=2.5), zorder=3)
    ax1.text(0.15, 0.55, r"Natural Gradient $-\eta g^{-1} \nabla C$", color="#15803d", fontsize=9.5, fontweight="bold")

    # Panel 2: QFIM Rank Saturation & Overparameterization Threshold
    layers = np.array([1, 2, 3, 4, 5, 6, 7])
    rank_curve = np.array([3, 5, 6, 6, 6, 6, 6])
    params = np.array([4, 8, 12, 16, 20, 24, 28])

    ax2.plot(layers, rank_curve, "o-", color="#2563eb", lw=2.5, markersize=7, label=r"$\mathrm{rank}(g(\theta))$ (QFIM)")
    ax2.axhline(6, color="#dc2626", linestyle="--", lw=1.8, label=r"Max Rank Bound $2(2^n - 1)$")
    ax2.axvline(3, color="#16a34a", linestyle=":", lw=2.0, label="Overparameterization $L^*$")

    ax2.annotate("Critical Transition $L^*$\nNo Spurious Local Minima", xy=(3, 4.5), xytext=(4.2, 3.5),
                 arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5),
                 fontsize=9, fontweight="bold", color="#15803d",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#86efac"))

    ax2.set_xlabel("Circuit Layers ($L$)", fontsize=10)
    ax2.set_ylabel("Metric Tensor Rank", fontsize=10)
    ax2.set_title("QFIM Rank Saturation & Overparameterization", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=8.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    setup_style()
    img_dir = Path(__file__).resolve().parent / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating illustrations in {img_dir}...")
    make_ansatz_circuit_diagram(img_dir / "custom_ansatz_circuit.png")
    print(" - Generated custom_ansatz_circuit.png")

    make_pde_stencil_diagram(img_dir / "pde_discretization_stencil.png")
    print(" - Generated pde_discretization_stencil.png")

    make_fwi_workflow_diagram(img_dir / "fwi_adjoint_state_workflow.png")
    print(" - Generated fwi_adjoint_state_workflow.png")

    make_hhl_vs_vqls_schematic(img_dir / "vqls_vs_hhl_schematic.png")
    print(" - Generated vqls_vs_hhl_schematic.png")

    make_quantum_geometry_diagram(img_dir / "quantum_geometry_dla_qfim.png")
    print(" - Generated quantum_geometry_dla_qfim.png")

    print("All didactic illustrations generated successfully!")

if __name__ == "__main__":
    main()
