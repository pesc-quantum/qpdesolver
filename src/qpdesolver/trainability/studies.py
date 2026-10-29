"""
Study orchestration and benchmark pipelines for trainability and ansatz comparison.
All results serialize seamlessly to HDF5.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Sequence, Optional, Union
from pathlib import Path
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.ansatze.base import Ansatz
from qpdesolver.ansatze.library import RealAmplitudes, HardwareEfficient
from qpdesolver.solvers.vqls.solver import VQLSSolver
from qpdesolver.trainability.gradients import (
    gradient_statistics,
    make_global_projector_cost,
    make_local_projector_cost,
)
from qpdesolver.trainability.dla import dynamical_lie_algebra
from qpdesolver.trainability.geometry import qfim_statistics
from qpdesolver.trainability.expressibility import expressibility, entangling_capability
from qpdesolver.io import save_results, load_results

@dataclass
class StudyConfig:
    """Configuration for a trainability study."""
    qubits: List[int]
    layers: List[int]
    ansatze: List[str]
    n_samples: int = 50
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qubits": self.qubits,
            "layers": self.layers,
            "ansatze": self.ansatze,
            "n_samples": self.n_samples,
            "seed": self.seed,
        }

@dataclass
class StudyResult:
    """Holds structured trainability or ansatz comparison experiment results."""
    records: List[Dict[str, Any]]
    config: Dict[str, Any]

    def save(self, path: Union[str, Path], group: str, overwrite: bool = False) -> None:
        """Saves study results into HDF5 file."""
        data = {"records_count": len(self.records)}
        for idx, rec in enumerate(self.records):
            data[f"record_{idx:03d}"] = rec
        save_results(path, group, data, config=self.config, overwrite=overwrite)

    @classmethod
    def load(cls, path: Union[str, Path], group: str) -> "StudyResult":
        """Loads study results from HDF5 file."""
        data, config = load_results(path, group)
        records = []
        count = data.get("records_count", 0)
        for idx in range(count):
            rec = data.get(f"record_{idx:03d}")
            if rec is not None:
                records.append(rec)
        return cls(records=records, config=config)

class TrainabilityStudy:
    """
    Orchestrates systematic evaluation of gradient variance, DLA, QFIM, and expressibility
    across ansatz families and qubit counts.
    """
    def __init__(self, config: StudyConfig):
        self.config = config

    def run(self) -> StudyResult:
        records = []
        for ans_name in self.config.ansatze:
            for n_q in self.config.qubits:
                for l in self.config.layers:
                    if ans_name.lower() == "realamplitudes":
                        ansatz = RealAmplitudes(layers=l)
                    elif ans_name.lower() == "hardwareefficient":
                        ansatz = HardwareEfficient(layers=l)
                    else:
                        ansatz = RealAmplitudes(layers=l)

                    # 1. Gradient variance (global vs local cost)
                    stats_global = gradient_statistics(
                        ansatz, n_q, make_global_projector_cost,
                        n_samples=self.config.n_samples, seed=self.config.seed
                    )
                    stats_local = gradient_statistics(
                        ansatz, n_q, make_local_projector_cost,
                        n_samples=self.config.n_samples, seed=self.config.seed
                    )

                    # 2. DLA dimension
                    dla_res = dynamical_lie_algebra(ansatz, n_q)

                    # 3. QFIM rank
                    qfim_res = qfim_statistics(ansatz, n_q, n_samples=min(10, self.config.n_samples), seed=self.config.seed)

                    # 4. Expressibility & entanglement
                    expr = expressibility(ansatz, n_q, n_pairs=min(200, self.config.n_samples * 4), seed=self.config.seed)
                    ent = entangling_capability(ansatz, n_q, n_samples=min(50, self.config.n_samples), seed=self.config.seed)

                    record = {
                        "ansatz": ans_name,
                        "n_qubits": n_q,
                        "layers": l,
                        "var_grad_global": stats_global.total_variance,
                        "var_grad_local": stats_local.total_variance,
                        "dla_dim": dla_res.dimension,
                        "dla_is_full_su": dla_res.is_full_su,
                        "qfim_rank": qfim_res.rank,
                        "qfim_bound": qfim_res.max_rank_bound,
                        "expressibility": expr,
                        "entangling_capability": ent,
                    }
                    records.append(record)

        return StudyResult(records=records, config=self.config.to_dict())

class AnsatzComparison:
    """
    Directly compares multiple ansatz architectures solving a given PDE linear system.
    """
    def __init__(
        self,
        system: LinearSystem,
        ansatze: Dict[str, Ansatz],
        max_iter: int = 80,
        seeds: Sequence[int] = (42,),
    ):
        self.system = system
        self.ansatze = ansatze
        self.max_iter = max_iter
        self.seeds = seeds

    def run(self) -> StudyResult:
        records = []
        for name, ans in self.ansatze.items():
            for s in self.seeds:
                solver = VQLSSolver(ansatz=ans, max_iter=self.max_iter, seed=s)
                res = solver.solve(self.system)
                dla_res = dynamical_lie_algebra(ans, self.system.n_qubits)

                records.append({
                    "ansatz": name,
                    "seed": s,
                    "fidelity": float(res.fidelity),
                    "residual": float(res.residual),
                    "final_cost": float(res.metadata["final_cost"]),
                    "wall_time": float(res.wall_time),
                    "circuit_evals": int(res.n_circuit_evals),
                    "num_params": ans.num_params(self.system.n_qubits),
                    "dla_dim": dla_res.dimension,
                })

        cfg = {
            "system_name": self.system.name,
            "n_qubits": self.system.n_qubits,
            "max_iter": self.max_iter,
            "ansatze": list(self.ansatze.keys()),
        }
        return StudyResult(records=records, config=cfg)
