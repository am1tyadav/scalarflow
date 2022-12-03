import math
from typing import Optional, Tuple

from scalarflow.core.common import Identifiable, SetPropertyNotAllowedError
from scalarflow.core.scalar import Scalar


class Operator(Identifiable):
    def __init__(self, name: str, num_arguments: int) -> None:
        """Operator does some sort of computation on given Scalar arguments.

        This is a base class, and any concrete implementations should
        implement the forward and backward methods.

        Args:
            name: Name of the Operator, used just for plotting and logging
            num_arguments: Number of arguments that this operator should expect.
        """

        super().__init__()

        self._name = name
        self._num_arguments = num_arguments

        self._arguments: Optional[Tuple[Scalar]] = None
        self._result: Optional[Scalar] = None

    def __repr__(self) -> str:
        return f"Operator(name={self._name}, num_arguments={self._num_arguments})"

    def __call__(self, arguments: Tuple[Scalar | int | float]) -> Scalar:
        assert (
            len(arguments) == self.num_arguments
        ), f"Expected number of arguments: {self.num_arguments}, but given {len(arguments)}"

        self._arguments = tuple(
            [Scalar.make_scalar(argument) for argument in arguments]
        )
        self._result = Scalar(data=self.forward(), operator=self)
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

    def forward(self) -> float:
        raise NotImplementedError("The method 'forward' is not yet implemented")

    def backward(self) -> None:
        raise NotImplementedError("The method 'backward' is not yet implemented")


class Add(Operator):
    def __init__(self) -> None:
        super().__init__(name="add", num_arguments=2)

    def forward(self) -> float:
        return self.arguments[0].data + self.arguments[1].data

    def backward(self) -> None:
        self.arguments[0].gradient += self.result.gradient
        self.arguments[1].gradient += self.result.gradient


class Subtract(Operator):
    def __init__(self) -> None:
        super().__init__(name="subtract", num_arguments=2)

    def forward(self) -> float:
        return self.arguments[0].data - self.arguments[1].data

    def backward(self) -> None:
        self.arguments[0].gradient += self.result.gradient
        self.arguments[1].gradient += self.result.gradient * -1


class Multiply(Operator):
    def __init__(self) -> None:
        super().__init__(name="multiply", num_arguments=2)

    def forward(self) -> float:
        return self.arguments[0].data * self.arguments[1].data

    def backward(self) -> None:
        self.arguments[0].gradient += self.result.gradient * self.arguments[1].data
        self.arguments[1].gradient += self.result.gradient * self.arguments[0].data


class Divide(Operator):
    def __init__(self) -> None:
        super().__init__(name="divide", num_arguments=2)

    def forward(self) -> float:
        assert self.arguments[1].data != 0, "Can not divide by zero"
        return self.arguments[0].data / self.arguments[1].data

    def backward(self) -> None:
        self.arguments[0].gradient += self.result.gradient * (
            self.arguments[1].data ** -1
        )
        self.arguments[1].gradient += self.result.gradient * (
            -1 * self.arguments[0].data * (self.arguments[1].data ** -2)
        )


class Power(Operator):
    def __init__(self, power: int) -> None:
        super().__init__(name=f"power_{power}", num_arguments=1)

        self._power = power

    def forward(self) -> float:
        return self.arguments[0].data ** self._power

    def backward(self) -> None:
        self.arguments[0].gradient += (
            self.result.gradient
            * self._power
            * self.arguments[0].data ** (self._power - 1)
        )


class ReLU(Operator):
    def __init__(self) -> None:
        super().__init__(name="relu", num_arguments=1)

    def forward(self) -> float:
        return max(self.arguments[0].data, 0.0)

    def backward(self) -> None:
        self.arguments[0].gradient += float(self.result.data > 0) * self.result.gradient


class Sigmoid(Operator):
    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__(name="sigmoid", num_arguments=1)

        self._epsilon = epsilon

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.e ** (-x))

    def forward(self) -> float:
        return self.sigmoid(self.arguments[0].data)

    def backward(self) -> None:
        x = self.arguments[0].data
        self.arguments[0].gradient += (
            self.result.gradient
            * self.sigmoid(x)
            / (1 - self.sigmoid(x) + self._epsilon)
        )
