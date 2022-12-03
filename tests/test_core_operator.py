import pytest

from scalarflow.core.common import SetPropertyNotAllowedError
from scalarflow.core.operator import Add, Operator, Subtract
from scalarflow.core.scalar import Scalar


def test_repr():
    op = Operator(name="op123", num_arguments=123)
    expected = "Operator(name=op123, num_arguments=123)"

    assert expected == str(op)


def test_name_property():
    # Test name property setter and getter

    op = Operator(name="123", num_arguments=123)

    assert op.name == "123"

    with pytest.raises(SetPropertyNotAllowedError):
        op.name = "1234"


def test_num_arguments_property():
    # Test num_arguments property setter and getter

    op = Operator(name="123", num_arguments=123)

    assert op.num_arguments == 123

    with pytest.raises(SetPropertyNotAllowedError):
        op.num_arguments = 1


def test_arguments_property():
    # Test arguments property getter and setter

    arguments = (
        1,
        2,
    )

    op = Operator(name="", num_arguments=2)

    assert op.arguments is None

    with pytest.raises(NotImplementedError):
        _ = op(arguments=arguments)

    for original, transformed in zip(arguments, op.arguments):
        assert transformed.data == original

    with pytest.raises(SetPropertyNotAllowedError):
        op.arguments = ()


def test_result_property():
    # Test result property setter and getter

    op = Operator(name="", num_arguments=1)

    assert op.result is None

    with pytest.raises(SetPropertyNotAllowedError):
        op.result = None


def test_call_raises_assertion_error():
    # Operator __call__ should raise assertion error if number
    # of arguments mis-match

    op = Operator(name="", num_arguments=1)

    with pytest.raises(AssertionError):
        op(
            arguments=(
                1,
                2,
            )
        )


def test_call_raises_not_implemented_error():
    # Operator forward (via __call__) and backward raise NotImplementedError

    op = Operator(name="", num_arguments=1)

    with pytest.raises(NotImplementedError):
        op(arguments=(1,))

    with pytest.raises(NotImplementedError):
        op.backward()


# Test Concrete Implementations


@pytest.mark.parametrize(
    argnames="arguments, forward_result, backward_result",
    argvalues=(((1, 2), 3.0, 1.0), ((0, -1), -1.0, -1.0)),
)
def test_add_forward_and_backward(
    arguments: tuple[float], forward_result: float, backward_result: float
):
    # Test the forward and backward implementations

    add = Add()
    output = add(arguments=arguments)

    assert output.data == forward_result

    add._result = Scalar(trainable=True)
    add._result.gradient = backward_result

    add.backward()

    assert add.arguments[0].gradient == backward_result
    assert add.arguments[1].gradient == backward_result


@pytest.mark.parametrize(
    argnames="arguments, forward_result, backward_result",
    argvalues=(((1, 2), -1.0, 1.0), ((0, -1), 1.0, -1.0)),
)
def test_subtract_forward_and_backward(
    arguments: tuple[float], forward_result: float, backward_result: float
):
    # Test the forward and backward implementations

    subtract = Subtract()
    output = subtract(arguments=arguments)

    assert output.data == forward_result

    subtract._result = Scalar(trainable=True)
    subtract._result.gradient = backward_result

    subtract.backward()

    assert subtract.arguments[0].gradient == backward_result
    assert subtract.arguments[1].gradient == backward_result * -1
