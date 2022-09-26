from aironsuit.callbacks import get_basic_callbacks


def test_get_basic_callbacks():
    assert isinstance(get_basic_callbacks(), dict)
