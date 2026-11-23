"""driftgate: catch LLM quality drift before it merges."""

from driftgate.config import Suite, load_suite
from driftgate.results import RunResult

__all__ = ["RunResult", "Suite", "__version__", "load_suite"]
__version__ = "0.1.0"
