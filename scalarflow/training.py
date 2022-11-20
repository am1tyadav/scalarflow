from typing import Tuple, Union

from scalarflow.core.operator import Operator
from scalarflow.core.scalar import Scalar


def create_scalar_graph(root: Scalar) -> Tuple[Scalar]:
    visited = []
    op_or_scalar = []

    def _build(_root: Union[Scalar, Operator]):
        if _root not in visited and _root is not None:
            visited.append(_root)

            if isinstance(_root, Operator):
                children = _root.arguments
            elif isinstance(_root, Scalar):
                children = [_root.operator]
            else:
                raise TypeError(
                    f"_root can only be either Scalar or Operator, not of type {type(_root)}"
                )

            for child in children:
                _build(child)

            op_or_scalar.append(_root)

    _build(root)
    return tuple(reversed(op_or_scalar))


def backward(graph: Tuple[Scalar]) -> None:
    for op_or_scalar in graph:
        if isinstance(op_or_scalar, Operator):
            op_or_scalar.backward()


def optimisation_step(root: Scalar, lr: float) -> None:
    graph = create_scalar_graph(root=root)

    for op_or_scalar in graph:
        if isinstance(op_or_scalar, Scalar):
            op_or_scalar.gradient = 0.0

    root.gradient = 1.0

    backward(graph=graph)

    for op_or_scalar in graph:
        if isinstance(op_or_scalar, Scalar):
            if op_or_scalar.trainable:
                op_or_scalar.data = op_or_scalar.data - lr * op_or_scalar.gradient
