"""
Trainability subpackage: barren plateau detection, DLA, geometry (QFIM), expressibility, mitigation, and studies.
"""
from qpdesolver.trainability.gradients import (
    GradientStats,
    BPDiagnosis,
    gradient_statistics,
    scan_gradient_variance,
    diagnose_barren_plateau,
    make_global_projector_cost,
    make_local_projector_cost,
)
from qpdesolver.trainability.dla import (
    DLAResult,
    dynamical_lie_algebra,
)
from qpdesolver.trainability.geometry import (
    QFIMStats,
    fubini_study_metric,
    qfim,
    qfim_statistics,
    overparameterization_scan,
)
from qpdesolver.trainability.expressibility import (
    expressibility,
    meyer_wallach_measure,
    entangling_capability,
)
from qpdesolver.trainability.mitigation import (
    LayerwiseTrainer,
    compare_initializations,
)
from qpdesolver.trainability.studies import (
    StudyConfig,
    StudyResult,
    TrainabilityStudy,
    AnsatzComparison,
)
from qpdesolver.trainability.plots import (
    plot_barren_plateau_scaling,
    plot_ansatz_comparison,
)

__all__ = [
    "GradientStats",
    "BPDiagnosis",
    "gradient_statistics",
    "scan_gradient_variance",
    "diagnose_barren_plateau",
    "make_global_projector_cost",
    "make_local_projector_cost",
    "DLAResult",
    "dynamical_lie_algebra",
    "QFIMStats",
    "fubini_study_metric",
    "qfim",
    "qfim_statistics",
    "overparameterization_scan",
    "expressibility",
    "meyer_wallach_measure",
    "entangling_capability",
    "LayerwiseTrainer",
    "compare_initializations",
    "StudyConfig",
    "StudyResult",
    "TrainabilityStudy",
    "AnsatzComparison",
    "plot_barren_plateau_scaling",
    "plot_ansatz_comparison",
]
