"""
Public package interface for sonetsim.
"""

from importlib.metadata import PackageNotFoundError, version

from .sonetsim import GraphEvaluator, GraphSimulator

__all__ = ["GraphEvaluator", "GraphSimulator", "__version__"]

try:
    __version__ = version("sonetsim")
except PackageNotFoundError:
    __version__ = "0+unknown"
