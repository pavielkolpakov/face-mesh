import pytest

from face_mesh.domain.errors import InputError, TransientError


def test_input_error_prefixes_message():
    err = InputError("no face detected")
    assert str(err) == "no face detected"
    assert err.public_message == "INPUT: no face detected"


def test_transient_error_prefixes_message():
    err = TransientError("supabase 503")
    assert err.public_message == "INTERNAL: supabase 503"


def test_input_error_is_not_transient():
    with pytest.raises(InputError):
        raise InputError("x")
    with pytest.raises(TransientError):
        raise TransientError("y")
    assert not issubclass(InputError, TransientError)
