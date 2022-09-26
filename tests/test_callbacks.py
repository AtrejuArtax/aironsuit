from aironsuit.callbacks import get_basic_callbacks


def test_get_basic_callbacks():
    basic_callbacks = get_basic_callbacks()
    assert isinstance(basic_callbacks, list)
    for callback_dict in basic_callbacks:
        assert isinstance(callback_dict, dict)
        for callback_name, callback_sub_dict in callback_dict.items():
            assert isinstance(callback_name, str)
            assert isinstance(callback_dict[callback_name]["kwargs"], dict)
