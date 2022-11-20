from typing import Optional, Tuple, Union

from scalarflow.core.common import Identifiable, SetPropertyNotAllowedError, make_scalar
from scalarflow.core.scalar import Scalar


class Operator(Identifiable):
    def __init__(self, name: str, num_arguments: int) -> None:
        """Operator does some sort of computation on given Scalar arguments.

        This is a base class, and any concrete implementations should
        implement the forward and backward methods.

        Args:
            name: Name of the Operator, used just for plotting and logging
            num_arguments: Number of arguments that this operator's __call__
                method should expect.
        """

        super().__init__()

        self._name = name
        self._num_arguments = num_arguments

        self._arguments: Optional[Tuple[Scalar]] = None
        self._result: Optional[Scalar] = None

    def __repr__(self) -> str:
        return f"Operator(name={self._name}, num_arguments={self._num_arguments})"

    def __call__(self, arguments: Union[Scalar, int, float]) -> Scalar:
        self._arguments = [make_scalar(argument) for argument in arguments]
        self.forward()
        return self._result

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, *_) -> None:
        raise SetPropertyNotAllowedError("name")

    @property
    def num_arguments(self) -> int:
        return self._num_arguments

    @num_arguments.setter
    def num_arguments(self, *_) -> None:
        raise SetPropertyNotAllowedError("num_arguments")

    @property
    def arguments(self) -> Optional[Tuple[Scalar]]:
        return self._arguments

    @arguments.setter
    def arguments(self, *_) -> None:
        raise SetPropertyNotAllowedError("arguments")

    @property
    def result(self) -> Optional[Scalar]:
        return self._result

    @result.setter
    def result(self, *_) -> None:
        raise SetPropertyNotAllowedError("result")

    def forward(self) -> None:
        raise NotImplementedError("The method 'forward' is not yet implemented")

    def backward(self) -> None:
        raise NotImplementedError("The method 'backward' is not yet implemented")
