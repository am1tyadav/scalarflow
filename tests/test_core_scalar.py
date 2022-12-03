import pytest

from scalarflow.core.common import SetPropertyNotAllowedError
from scalarflow.core.operator import Operator
from scalarflow.core.scalar import Scalar


def test_random_data_exists():
    # Test if data exists even if no initial value is given

    scalar = Scalar()

    assert type(scalar.data) == float


def test_repr():
    scalar = Scalar(data=0.1, operator=None, trainable=True)

    assert "Scalar(data=0.1000, operator=None, trainable=True, gradient=0.0000)" == str(
        scalar
    )


def test_data_property():
    # Test data property setter and getter
    # Setter should work only if trainable is True

    scalar = Scalar(trainable=True)
    scalar.data = 0.1

    assert scalar.data == 0.1

    scalar = Scalar()

    with pytest.raises(SetPropertyNotAllowedError):
        scalar.data = 0.0


def test_operator_property():
    # Test if operator property getter and setter

    op = Operator(name="", num_arguments=1)
    scalar = Scalar(operator=op)

    assert scalar.operator == op

    with pytest.raises(SetPropertyNotAllowedError):
        scalar.operator = None


def test_trainable_property():
    # Test trainable property getter and setter

    scalar = Scalar(trainable=True)

    assert scalar.trainable

    scalar.trainable = False

    assert not scalar.trainable


def test_gradient_property():
    # Test graident property setter and getter

    scalar = Scalar()

    assert scalar.gradient == 0.0

    scalar.gradient = 1.0

    assert scalar.gradient == 1.0


def test_make_scalar_for_scalar():
    # Test if make_scalar returns same instance for scalar input

    scalar = Scalar()

    assert scalar == Scalar.make_scalar(scalar_like=scalar)


@pytest.mark.parametrize(argnames=("data"), argvalues=(0, 0.1, 1))
def test_make_scalar_for_int_and_float(data: int | float):
    # Test if make_scalar returns new instance with given data for int or float values

    scalar = Scalar(data=data)

    assert Scalar.make_scalar(scalar_like=scalar).data == data


@pytest.mark.parametrize(argnames=("scalar_like"), argvalues=("1.2", {1: 1}))
def test_make_scalar_raises_error(scalar_like):
    # make_scalar should raise TypeError if argument is not of acceptable type

    with pytest.raises(TypeError):
        _ = Scalar.make_scalar(scalar_like=scalar_like)


def test_make_float_for_scalar():
    # Test if make_float returns data as float for a Scalar input

    scalar = Scalar()

    assert type(Scalar.make_float(scalar_like=scalar)) == float
    assert Scalar.make_float(scalar_like=scalar) == scalar.data


@pytest.mark.parametrize(argnames=("data"), argvalues=(0, 0.1, 1))
def test_make_float_for_int_and_float(data: int | float):
    # Test if make_float works for int or float values

    assert Scalar.make_float(data) == data


@pytest.mark.parametrize(argnames=("scalar_like"), argvalues=("1.2", {1: 1}))
def test_make_float_raises_error(scalar_like):
    # Raises TypeError for the wrong type

    with pytest.raises(TypeError):
        Scalar.make_float(scalar_like=scalar_like)
