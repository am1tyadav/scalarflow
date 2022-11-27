from typing import Callable, Tuple, Union

from scalarflow.core.scalar import Scalar
from scalarflow.node import Node


class Dense:
    def __init__(
        self, output_dim: int, input_dim: int, activation: Callable = None
    ) -> None:
        self._output_dim = output_dim
        self._input_dim = input_dim

        self._nodes = [
            Node(num_inputs=input_dim, activation=activation)
            for _ in range(0, output_dim)
        ]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def __call__(self, inputs: Tuple[Scalar]) -> Union[Tuple[Scalar], Scalar]:
        if self._output_dim == 1:
            return self._nodes[0](inputs=inputs)
        return [node(inputs=inputs) for node in self._nodes]

    def summary(self, index: int = 0) -> None:
        print("=" * 10, f"Layer {index}", "=" * 10)

        for index, node in enumerate(self._nodes):
            node.summary(index)
