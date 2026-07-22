"""spectral-telemetry: streaming spectral-capacity diagnostics for neural nets.

Public API lives in :mod:`spectral_telemetry.telemetry`; the numpy-only names
import eagerly, torch-backed names load lazily on first access.
"""

from spectral_telemetry import telemetry as _facade

__version__ = "0.1.0"
__all__ = list(_facade.__all__)


def __getattr__(name):
    return getattr(_facade, name)


def __dir__():
    return sorted(set(__all__) | {"__version__"})
