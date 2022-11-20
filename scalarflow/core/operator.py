from typing import Optional, Tuple, Union

from scalarflow.core.common import Identifiable, SetPropertyNotAllowedError
from scalarflow.core.scalar import Scalar, make_scalar


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


class Add(Operator):
    def __init__(self) -> None:
        super().__init__(name="add", num_arguments=2)

    def forward(self) -> None:
        data = self.arguments[0].data + self.arguments[1].data
        self._result = Scalar(data=data, operator=self)

    def backward(self) -> None:
        self.arguments[0].gradient += self.result.gradient
        self.arguments[1].gradient += self.result.gradient


class Subtract(Operator):
    def __init__(self) -> None:
        super().__init__(name="subtract", num_arguments=2)

    def forward(self) -> None:
        data = self.arguments[0].data - self.arguments[1].data
        self._result = Scalar(data=data, operator=self)

    def backward(self) -> None:
        self.arguments[0].gradient += self.result.gradient
        self.arguments[1].gradient += self.result.gradient * -1


class Multiply(Operator):
    def __init__(self) -> None:
        super().__init__(name="multiply", num_arguments=2)

    def forward(self) -> None:
        data = self.arguments[0].data * self.arguments[1].data
        self._result = Scalar(data=data, operator=self)

    def backward(self) -> None:
        self.arguments[0].gradient += self.result.gradient * self.arguments[1].data
        self.arguments[1].gradient += self.result.gradient * self.arguments[0].data


class Divide(Operator):
    def __init__(self) -> None:
        super().__init__(name="divide", num_arguments=2)

    def forward(self) -> None:
        a = self.arguments[0].data
        b = self.arguments[1].data

        assert b != 0, "Can not divide by zero"

        data = a / b
        self._result = Scalar(data=data, operator=self)

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

    def forward(self) -> None:
        data = self.arguments[0].data ** self._power
        self._result = Scalar(data=data, operator=self)

    def backward(self) -> None:
        self.arguments[0].gradient += (
            self.result.gradient
            * self._power
            * self.arguments[0].data ** (self._power - 1)
        )


def add(a: Scalar, b: Scalar) -> Scalar:
    """Convenience function to add two Scalars.
    Args:
        a: Scalar
        b: Scalar

    Returns:
        a + b
    """
    return Add()(arguments=(a, b))


def subtract(a: Scalar, b: Scalar) -> Scalar:
    """Convenience function to subtract two Scalars.

    Args:
        a: Scalar
        b: Scalar

    Returns:
        a - b
    """
    return Subtract()(arguments=(a, b))


def multiply(a: Scalar, b: Scalar) -> Scalar:
    """Convenience function to multiply two scalars.

    Args:
        a: Scalar
        b: Scalar

    Returns:
        a * b
    """
    return Multiply()(arguments=(a, b))


def divide(a: Scalar, b: Scalar) -> Scalar:
    """Convenience function to divide two scalars.

    Args:
        a: Scalar
        b: Scalar

    Returns:
        a / b
    """
    return Divide()(arguments=(a, b))


def power(a: Scalar, b: int) -> Scalar:
    """Convenience function to compute power of a Scalar.

    Args:
        a: Scalar
        b: int value to be used as the power of the Scalar a

    Returns:
        a ** b
    """
    return Power(power=b)(arguments=(a,))
