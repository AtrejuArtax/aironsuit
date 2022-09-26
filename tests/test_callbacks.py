from aironsuit.callbacks import basic_callbacks


def test_basic_callbacks():
    assert isinstance(basic_callbacks(), dict)
