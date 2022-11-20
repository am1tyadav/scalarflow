from __future__ import annotations

import random
from typing import Optional, TypeVar, Union

from scalarflow.core.common import Identifiable, SetPropertyNotAllowedError

_Operator = TypeVar("_Operator")


class Scalar(Identifiable):
    def __init__(
        self,
        data: Optional[Union[int, float]] = None,
        operator: Optional[_Operator] = None,
        trainable: Optional[bool] = False,
    ) -> None:
        """Represents a scalar value and associated properties.

        Args:
            data: Value of the scalar, could be int or float but is
                always cast to a float
            operator: Reference to the operator, if any, that resulted in
                the scalar instance.
            trainable: A flag to set if the scalar is updated during
                training.
        """

        super().__init__()

        self._data = float(data) if data is not None else random.random()
        self._operator = operator
        self._trainable = trainable
        self._gradient = 0.0

    def __repr__(self) -> str:
        return f"Scalar(data={self._data:.4f}, operator={self._operator}, trainable={self._trainable}, gradient={self._gradient:.4f})"

    @property
    def data(self) -> Optional[float]:
        return self._data

    @data.setter
    def data(self, updated_value: float) -> None:
        if not self.trainable:
            raise SetPropertyNotAllowedError("data")

        self._data = updated_value

    @property
    def operator(self) -> Optional[_Operator]:
        return self._operator

    @operator.setter
    def operator(self, *_) -> None:
        raise SetPropertyNotAllowedError("operator")

    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, updated_value: bool) -> None:
        self._trainable = updated_value

    @property
    def gradient(self) -> float:
        return self._gradient

    @gradient.setter
    def gradient(self, updated_value: float) -> None:
        self._gradient = updated_value


def make_scalar(scalar_like: Union[Scalar, int, float]) -> Scalar:
    if isinstance(scalar_like, Scalar):
        return scalar_like
    if isinstance(scalar_like, (int, float)):
        return Scalar(data=scalar_like)

    raise TypeError(
        "Argument 'scalar_like' can only be one of three types: Scalar, int or float"
    )
