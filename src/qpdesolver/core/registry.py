"""
Generic registry pattern for qpdesolver components.
"""
from typing import Dict, Any, Callable, TypeVar, Optional, List

T = TypeVar("T")

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Any] = {}

    def register(self, key: str) -> Callable[[T], T]:
        key = key.lower()
        def decorator(cls_or_fn: T) -> T:
            if key in self._entries:
                raise ValueError(f"'{key}' is already registered in registry '{self.name}'.")
            self._entries[key] = cls_or_fn
            return cls_or_fn
        return decorator

    def get(self, key: str) -> Any:
        key = key.lower()
        if key not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise KeyError(f"'{key}' not found in registry '{self.name}'. Available: [{available}]")
        return self._entries[key]

    def create(self, key: str, *args, **kwargs) -> Any:
        entry = self.get(key)
        if callable(entry):
            return entry(*args, **kwargs)
        return entry

    def available(self) -> List[str]:
        return sorted(list(self._entries.keys()))

    def keys(self):
        return self._entries.keys()

    def items(self):
        return self._entries.items()

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._entries

    def __repr__(self) -> str:
        return f"Registry('{self.name}', entries={self.available()})"

# Global registry instances
REGISTRY_ANSATZE = Registry("ansatze")
REGISTRY_OPTIMIZERS = Registry("optimizers")
REGISTRY_SOLVERS = Registry("solvers")
REGISTRY_BACKENDS = Registry("backends")
REGISTRY_DISCRETIZATIONS = Registry("discretizations")
REGISTRY_INITIALIZERS = Registry("initializers")
REGISTRY_COSTS = Registry("costs")
