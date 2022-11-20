from typing import Tuple

from scalarflow.core.operator import add, divide, power, subtract
from scalarflow.core.scalar import Scalar


def squared_error(y_true: Scalar, y_pred: Scalar) -> Scalar:
    difference = subtract(y_true, y_pred)
    return power(difference, 2)


def mean_squared_error(y_true: Tuple[Scalar], y_pred: Tuple[Scalar]) -> Scalar:
    num_examples = len(y_true)

    if num_examples == 1:
        return squared_error(y_true[0], y_pred[0])

    error = None

    for true, pred in zip(y_true, y_pred):
        current_error = squared_error(true, pred)
        error = current_error if error is None else add(error, current_error)

    return divide(error, num_examples)
