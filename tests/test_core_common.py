import pytest

from scalarflow.core.common import Identifiable, SetPropertyNotAllowedError


def test_identifiable():
    identifiable = Identifiable()

    assert type(identifiable.uuid) == str

    with pytest.raises(SetPropertyNotAllowedError):
        identifiable.uuid = "new_id"
