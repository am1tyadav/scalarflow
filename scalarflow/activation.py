from scalarflow.core.operator import Operator
from scalarflow.core.scalar import Scalar


class ReLU(Operator):
    def __init__(self) -> None:
        super().__init__(name="relu", num_arguments=1)

    def forward(self) -> None:
        data = max(self.arguments[0].data, 0.0)
        self._result = Scalar(data=data, operator=self)

    def backward(self) -> None:
        self.arguments[0].gradient += float(self.result.data > 0) * self.result.gradient


def relu(a: Scalar) -> Scalar:
    """Convenience function for ReLU activation."""
    return ReLU()(arguments=(a,))
