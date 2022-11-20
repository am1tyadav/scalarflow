"""scalarflow

A Machine Learning library written in pure Python for educational purpose.
"""

from scalarflow.activation import relu
from scalarflow.core.operator import add, divide, multiply, power, subtract
from scalarflow.core.scalar import Scalar
from scalarflow.loss import mean_squared_error, squared_error
from scalarflow.node import Node
from scalarflow.training import optimisation_step

__version__ = "0.1"
__all__ = [
    "Scalar",
    "add",
    "divide",
    "multiply",
    "power",
    "subtract",
    "Node",
    "relu",
    "mean_squared_error",
    "squared_error",
    "optimisation_step",
]
