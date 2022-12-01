from scalarflow.core.operator import (
    Add,
    Divide,
    Multiply,
    Power,
    ReLU,
    Sigmoid,
    Subtract,
)
from scalarflow.core.scalar import Scalar


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


def relu(a: Scalar) -> Scalar:
    """Convenience function for ReLU activation."""
    return ReLU()(arguments=(a,))


def sigmoid(a: Scalar) -> Scalar:
    """Convenience function for Sigmoid activation"""
    return Sigmoid()(arguments=(a,))
