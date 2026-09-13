"""A minimal, dependency-light virtual-fly toolkit built on the FlyWire connectome.

Logging goes through loguru. The library itself never configures a sink; the
command-line scripts do (see `flybrain.logging_setup.configure`).
"""
from . import circuits
from .anatomy import Anatomy
from .connectome import Connectome
from .lif import DEFAULT_PARAMS, LIFNetwork, Result
from .names import Names

__all__ = ["Anatomy", "Connectome", "LIFNetwork", "Result", "Names", "DEFAULT_PARAMS", "circuits"]
