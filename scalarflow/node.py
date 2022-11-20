from typing import Callable, Optional, Tuple

from scalarflow.core.operator import add, multiply
from scalarflow.core.scalar import Scalar


class Node:
    def __init__(self, num_inputs: int, activation: Optional[Callable] = None) -> None:
        self._num_inputs = num_inputs
        self._activation = activation

        self._weights = [Scalar(trainable=True) for _ in range(0, num_inputs)]
        self._bias = Scalar(trainable=True)

    def __repr__(self) -> str:
        return f"Node(num_inputs={self._num_inputs}, activation={self._activation})"

    def __call__(self, inputs: Tuple[Scalar]) -> Scalar:
        assert (
            len(inputs) == self._num_inputs
        ), "Node can only accept the number of inputs specified during initialisation"

        total = None

        for index, _input in enumerate(inputs):
            weighted = multiply(_input, self._weights[index])
            total = weighted if total is None else add(weighted, total)

        result = add(total, self._bias)

        if self._activation is None:
            return result
        return self._activation(result)

    def summary(self) -> None:
        print("=" * 10, "Weights", "=" * 10)

        for weight in self._weights:
            print(weight)

        print("=" * 10, "Bias", "=" * 10)
        print(self._bias)
