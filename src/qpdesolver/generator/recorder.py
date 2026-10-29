"""
Snapshot recording and field manipulation utilities for seismic simulations.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def flatten_fields(fields: Any) -> Dict[str, Any]:
    """
    Normalise *fields* into an ordered ``{name: Function}`` dict.

    Accepts a single Function, a list/tuple of them, a dict, or Devito's
    composite ``VectorTimeFunction`` / ``TensorTimeFunction`` objects, which are
    expanded into their scalar components. Symmetric tensors yield each unique
    component once (``t_xy`` is not duplicated).
    """
    out: Dict[str, Any] = {}

    def _add(obj: Any, hint: Optional[str] = None) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _add(v, hint=k)
            return

        if hasattr(obj, "values") and not hasattr(obj, "data"):
            for comp in obj.values():
                _add(comp)
            return

        if isinstance(obj, (list, tuple)):
            for v in obj:
                _add(v)
            return

        name = hint or getattr(obj, "name", None)
        if name is None:
            raise TypeError(f"cannot determine a name for field {obj!r}")
        if name in out and out[name] is not obj:
            raise ValueError(f"duplicate field name {name!r}")
        out[name] = obj

    _add(fields)
    if not out:
        raise ValueError("no fields given")
    return out


def strip_absorbing_layer(array: Any, nbl: int, ndim: Optional[int] = None) -> np.ndarray:
    """
    Drop *nbl* points from every spatial edge.

    ``array`` is assumed to have its spatial axes last, so a snapshot stack of
    shape ``(nsnaps, nx, nz)`` and a plain field of shape ``(nx, nz)`` are both
    handled. ``ndim`` is the number of trailing spatial axes (default: all but
    a leading snapshot axis is guessed from *nbl*).
    """
    if not nbl:
        return np.asanyarray(array)
    array = np.asanyarray(array)
    if ndim is None:
        ndim = array.ndim if array.ndim <= 3 else 3
        ndim = min(ndim, array.ndim)
    sl = [slice(None)] * array.ndim
    for ax in range(array.ndim - ndim, array.ndim):
        sl[ax] = slice(nbl, -nbl if nbl else None)
    return array[tuple(sl)]


class SnapshotRecorder:
    """
    Record ``nsnaps`` evenly spaced wavefield snapshots inside the time loop.

    Builds a ``ConditionalDimension`` that fires every ``factor`` time steps and
    one shadow ``TimeFunction`` per field, sized exactly to the number of
    snapshots. Append :attr:`equations` to the ``Operator``'s equation list and
    Devito does the rest; nothing needs to change in the stencil.

    Parameters
    ----------
    fields : Function | list | dict | VectorTimeFunction | TensorTimeFunction
        What to snapshot. Composite fields are expanded component-wise.
    nt : int
        Number of time iterations the operator will run, i.e.
        ``time_range.num``. Required unless ``factor`` is given.
    nsnaps : int, optional
        How many snapshots you want. Must satisfy ``1 <= nsnaps <= nt``.
        The achievable count is ``ceil(nt / ceil(nt / nsnaps))``, which is
        ``<= nsnaps``; read :attr:`nsnaps` afterwards for the real number.
    factor : int, optional
        Alternative to *nsnaps*: record every ``factor``-th step.
    record : {'current', 'forward'}
        Snapshot ``f`` (the wavefield at time *t*, the default) or ``f.forward``.
    name_fmt : str
        Format string for the shadow functions' names.

    Attributes
    ----------
    equations : list of Eq      append these to your Operator
    nsnaps    : int             snapshots actually recorded
    factor    : int             recording stride, in time steps
    indices   : ndarray[int]    time-step index of each snapshot
    """

    def __init__(
        self,
        fields: Any,
        nt: Optional[int] = None,
        nsnaps: Optional[int] = None,
        factor: Optional[int] = None,
        record: str = "current",
        name_fmt: str = "%ssave",
        time_dim: Any = None,
    ):
        try:
            from devito import ConditionalDimension, TimeFunction, Eq
        except ImportError as exc:
            raise ImportError(
                "SnapshotRecorder requires devito to be installed. "
                "Install it with: pip install devito"
            ) from exc

        self.fields = flatten_fields(fields)
        first = next(iter(self.fields.values()))
        grid = first.grid

        # ---- work out the stride -------------------------------------------
        if factor is None:
            if nsnaps is None:
                raise ValueError("give either nsnaps or factor")
            if nt is None:
                raise ValueError("nt is required when nsnaps is given")
            nt = int(nt)
            nsnaps = int(nsnaps)
            if nsnaps < 1:
                raise ValueError(f"nsnaps must be >= 1, got {nsnaps}")
            if nsnaps > nt:
                raise ValueError(
                    f"nsnaps={nsnaps} exceeds the number of time steps nt={nt} -- you "
                    "cannot save more snapshots than the solution has steps."
                )
            factor = max(1, int(math.ceil(nt / nsnaps)))
        else:
            factor = int(factor)
            if factor < 1:
                raise ValueError("factor must be >= 1")
            if nt is None:
                raise ValueError("nt is required")
            nt = int(nt)

        self.nt = nt
        self.factor = factor
        self.nsnaps = int(math.ceil(nt / factor))
        self.requested_nsnaps = nsnaps
        self.record = record

        if nsnaps is not None and self.nsnaps != nsnaps:
            import warnings
            warnings.warn(
                f"nsnaps={nsnaps} is not reachable with an integer stride over {nt} "
                f"time steps; recording {self.nsnaps} snapshots every {self.factor} steps instead. "
                f"Use a divisor of {nt} (or pass factor=...) for an exact count.",
                stacklevel=2,
            )

        # ---- the conditional time dimension ---------------------------------
        parent = time_dim if time_dim is not None else grid.time_dim
        self.time_dim = ConditionalDimension(
            name="t_snap", parent=parent, factor=self.factor
        )

        # ---- one shadow TimeFunction per field ------------------------------
        self.buffers: Dict[str, Any] = {}
        self.equations: List[Any] = []
        for name, f in self.fields.items():
            kwargs: Dict[str, Any] = dict(
                name=name_fmt % name,
                grid=f.grid,
                time_order=0,
                space_order=getattr(f, "space_order", 2),
                save=self.nsnaps,
                time_dim=self.time_dim,
            )
            stag = getattr(f, "staggered", None)
            if stag:
                kwargs["staggered"] = stag
            buf = TimeFunction(**kwargs)
            src = f.forward if record == "forward" else f
            self.buffers[name] = buf
            self.equations.append(Eq(buf, src))

        self.indices = np.arange(self.nsnaps, dtype=np.int64) * self.factor

    def times(self, dt: float, t0: float = 0.0) -> np.ndarray:
        """Physical time (same units as *dt*) of each snapshot."""
        return t0 + self.indices * float(dt)

    def arrays(self, nbl: int = 0, dtype: Any = None) -> Dict[str, np.ndarray]:
        """``{name: ndarray}`` of the recorded snapshots, shape ``(nsnaps, *space)``."""
        out: Dict[str, np.ndarray] = {}
        for name, buf in self.buffers.items():
            a = np.asarray(buf.data)
            if nbl:
                a = strip_absorbing_layer(a, nbl, ndim=a.ndim - 1)
            if dtype is not None:
                a = a.astype(dtype, copy=False)
            out[name] = a
        return out

    def __repr__(self) -> str:
        return (
            f"<SnapshotRecorder {list(self.fields)}  nsnaps={self.nsnaps} "
            f"(requested {self.requested_nsnaps})  factor={self.factor}  nt={self.nt}>"
        )

