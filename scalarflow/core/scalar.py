from typing import Union, Optional
from scalarflow.core.common import Identifiable, SetPropertyNotAllowedError
from scalarflow.core.operator import Operator


class Scalar(Identifiable):
    def __init__(self, data: Optional[Union[int, float]] = None, operator: Optional[Operator] = None, trainable: Optional[bool] = False) -> None:
        super().__init__()

        self._data = float(data)
        self._operator = operator
        self._trainable = trainable
        self._gradient = 0.0

    @property
    def data(self) -> float:
        return float(self._data)

    @data.setter
    def data(self, *_) -> None:
        raise SetPropertyNotAllowedError()
