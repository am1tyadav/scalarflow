import uuid


class SetPropertyNotAllowedError(Exception):
    def __init__(self, name: str, *args: object) -> None:
        """SetPropertyNotAllowedError

        Raised when there's an attempt to set a property value in the
        instances where it's not allowed.
        """
        super().__init__(f"Setting a value for property '{name}' is not allowed", *args)


class Identifiable:
    def __init__(self) -> None:
        """Base class to be extended for any classes that need identifiable instances.
        That is, every instance of this and derived classes will have a uuid
        property which can be used as a unique id
        """

        self._uuid = str(uuid.uuid4())

    @property
    def uuid(self) -> str:
        "Returns the unique id of this Identifiable object as a string"

        return self._uuid

    @uuid.setter
    def uuid(self, *_):
        raise SetPropertyNotAllowedError("uuid")
