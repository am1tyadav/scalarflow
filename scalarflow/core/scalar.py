import random
from typing import Optional, Union

from scalarflow.core.common import Identifiable, SetPropertyNotAllowedError
from scalarflow.core.operator import Operator


class Scalar(Identifiable):
    def __init__(
        self,
        data: Optional[Union[int, float]] = None,
        operator: Optional[Operator] = None,
        trainable: Optional[bool] = False,
    ) -> None:
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
    def data(self, *_) -> None:
        raise SetPropertyNotAllowedError("data")

    @property
    def operator(self) -> Optional[Operator]:
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
