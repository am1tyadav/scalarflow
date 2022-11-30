import math

from scalarflow.core.operator import Operator
from scalarflow.core.scalar import Scalar


class ReLU(Operator):
    def __init__(self) -> None:
        super().__init__(name="relu", num_arguments=1)

    def forward(self) -> float:
        return max(self.arguments[0].data, 0.0)

    def backward(self) -> None:
        self.arguments[0].gradient += float(self.result.data > 0) * self.result.gradient


class Sigmoid(Operator):
    def __init__(self) -> None:
        super().__init__(name="sigmoid", num_arguments=1)

        self._epsilon = 1e-7

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.e ** (-x))

    def forward(self) -> float:
        return self.sigmoid(self.arguments[0].data)

    def backward(self) -> None:
        x = self.arguments[0].data
        self.arguments[0].gradient += (
            self.result.gradient
            * self.sigmoid(x)
            / (1 - self.sigmoid(x) + self._epsilon)
        )


def relu(a: Scalar) -> Scalar:
    """Convenience function for ReLU activation."""
    return ReLU()(arguments=(a,))


def sigmoid(a: Scalar) -> Scalar:
    """Convenience function for Sigmoid activation"""
    return Sigmoid()(arguments=(a,))
