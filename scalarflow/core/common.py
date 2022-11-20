import uuid
from typing import Union

from scalarflow.core.scalar import Scalar


class SetPropertyNotAllowedError(Exception):
    def __init__(self, name: str, *args: object) -> None:
        """SetPropertyNotAllowedError

        Raised when there's an attempt to set a property value in the
        instances where it's not allowed.
        """
        super().__init__(f"Setting a value for property '{name}' is not allowed", *args)


class Identifiable:
    def __init__(self) -> None:
        self._uuid = str(uuid.uuid4())

    @property
    def uuid(self) -> str:
        return self._uuid

    @uuid.setter
    def uuid(self, *_):
        raise SetPropertyNotAllowedError("uuid")


def make_scalar(scalar_like: Union[Scalar, int, float]) -> Scalar:
    if isinstance(scalar_like, Scalar):
        return scalar_like
    if isinstance(scalar_like, (int, float)):
        return Scalar(data=scalar_like)

    raise TypeError("scalar_like can only be one of three types: Scalar, int or float")
