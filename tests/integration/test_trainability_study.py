"""Integration tests for TrainabilityStudy, HDF5 persistence, and LayerwiseTrainer."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from qpdesolver.ansatze import RealAmplitudes, HardwareEfficient
from qpdesolver.trainability.studies import TrainabilityStudy, StudyConfig, StudyResult
from qpdesolver.trainability.mitigation import LayerwiseTrainer, compare_initializations
from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.optimizers.first_order import Adam


def test_trainability_study_and_hdf5():
    """Test full TrainabilityStudy on small systems with HDF5 roundtrip."""
    cfg = StudyConfig(
        qubits=[2],
        layers=[1],
        ansatze=["RealAmplitudes"],
        n_samples=5,
        seed=42,
    )
    study = TrainabilityStudy(cfg)
    result = study.run()

    assert len(result.records) == 1
    rec = result.records[0]
    assert rec["ansatz"] == "RealAmplitudes"
    assert rec["n_qubits"] == 2
    assert "var_grad_global" in rec
    assert "var_grad_local" in rec
    assert "dla_dim" in rec
    assert "qfim_rank" in rec
    assert "expressibility" in rec
    assert "entangling_capability" in rec

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "study_results.h5"
        result.save(h5_path, group="study_01")
        assert h5_path.is_file()

        loaded = StudyResult.load(h5_path, group="study_01")
        assert len(loaded.records) == len(result.records)
        assert loaded.records[0]["ansatz"] == rec["ansatz"]
        assert loaded.records[0]["n_qubits"] == rec["n_qubits"]
        assert np.isclose(loaded.records[0]["var_grad_global"], rec["var_grad_global"])


def test_layerwise_trainer():
    """Test LayerwiseTrainer on a 2-qubit Helmholtz problem."""
    helmholtz = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    system = helmholtz.system(helmholtz.point_source(1), omega=1.0)

    trainer = LayerwiseTrainer(
        base_ansatz_factory=lambda l: RealAmplitudes(layers=l),
        max_layers=2,
        optimizer=Adam(lr=0.05),
        max_iter_per_layer=15,
    )

    result = trainer.train(system, seed=42)
    assert result.fidelity > 0.5
    assert len(result.metadata["optimal_params"]) == RealAmplitudes(layers=2).num_params(2)


def test_compare_initializations():
    """Test initialization comparison utility."""
    helmholtz = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    system = helmholtz.system(helmholtz.point_source(1), omega=1.0)
    ansatz = RealAmplitudes(layers=1)

    results = compare_initializations(
        system=system,
        ansatz=ansatz,
        strategies=("small_angle", "uniform"),
        seeds=(42,),
        max_iter=10,
    )
    assert "small_angle" in results
    assert "uniform" in results
    assert results["small_angle"]["mean_fidelity"] > 0.0
