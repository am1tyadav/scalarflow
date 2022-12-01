from typing import List, Tuple

from scalarflow.core.scalar import Scalar
from scalarflow.types import ScalarLike


class Metric:
    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, y_true: Tuple[ScalarLike], y_pred: Tuple[ScalarLike]) -> float:
        y_true: List[float] = [Scalar.make_float(label) for label in y_true]
        y_pred: List[float] = [Scalar.make_float(label) for label in y_pred]
        return self.compute(y_true, y_pred)

    def compute(self, y_true: Tuple[ScalarLike], y_pred: Tuple[ScalarLike]) -> float:
        raise NotImplementedError("Method 'compute' not implemented")


class BinaryAccuracy(Metric):
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(name="binary_accuracy")

        self._threshold = threshold

    def compute(self, y_true: Tuple[ScalarLike], y_pred: Tuple[ScalarLike]) -> float:
        assert len(y_true) > 0
        return sum(
            [
                int(float(pred > self._threshold) == gt)
                for gt, pred in zip(y_true, y_pred)
            ]
        ) / len(y_true)
