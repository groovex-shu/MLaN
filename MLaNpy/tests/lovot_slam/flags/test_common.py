import importlib
import re

import pytest

import lovot_slam.flags.debug_params
from lovot_slam.flags.common import get_param_from_ltm
from lovot_slam.redis import create_ltm_client


@pytest.fixture
def prepare_redis():
    ltm_client = create_ltm_client()
    for key in ltm_client.keys("slam:debug:*"):
        ltm_client.delete(key)

    yield ltm_client

    for key in ltm_client.keys("slam:debug:*"):
        ltm_client.delete(key)


# reason why param_str is specified by string:
# they has to be evaluated after importlib.reload
@pytest.mark.parametrize("param_str,key,value,correct_param", [
    ("lovot_slam.flags.debug_params.PARAM_DISABLE_REMOVING_FILES", "slam:debug:disable_removing_files", None, False),
    ("lovot_slam.flags.debug_params.PARAM_DISABLE_REMOVING_FILES", "slam:debug:disable_removing_files", '1', True),
    ("lovot_slam.flags.debug_params.PARAM_BUILD_MAP_RATE", "slam:debug:build_map_rate", None, 0.5),
    ("lovot_slam.flags.debug_params.PARAM_BUILD_MAP_RATE", "slam:debug:build_map_rate", '1.0', 1.0),
])
def test_params(prepare_redis, param_str, key, value, correct_param):
    if value:
        prepare_redis.set(key, value)
    importlib.reload(lovot_slam.flags.debug_params)
    assert eval(param_str) == correct_param


def test_bool_default_value(prepare_redis):
    """To test if the default value is correctly set."""
    key = "slam:debug:test_bool"
    # make sure the key does not exist
    prepare_redis.delete(key)
    assert get_param_from_ltm(key, True)
    assert not get_param_from_ltm(key, False)


def test_warning_messages(prepare_redis, caplog):
    """
    To test if warning messages are correctly logged,
    because the log message is significant to confirm the value is not set during the QA test.
    """
    key = "slam:debug:test_log"

    # make sure the key does not exist
    prepare_redis.delete(key)
    assert get_param_from_ltm(key, True)
    assert not get_param_from_ltm(key, False)
    # no warning message should be logged
    assert len(caplog.text) == 0

    # set something
    prepare_redis.set(key, '1')
    assert get_param_from_ltm(key, True)
    # warning message should be logged
    assert re.match(r'^WARNING.*slam:debug:test_log.*', caplog.text)


def test_invalid_type(prepare_redis):
    key = "slam:debug:test_invalid_type"
    # expect int, but set string
    prepare_redis.set(key, 'string')
    with pytest.raises(ValueError):
        get_param_from_ltm(key, 1)
    # expect float, but set string
    prepare_redis.set(key, 'string')
    with pytest.raises(ValueError):
        get_param_from_ltm(key, 1.0)
    # expect bool, but set string -> this is passed
    prepare_redis.set(key, 'string')
    assert get_param_from_ltm(key, True)
    # expect str, but set int -> this is OK
    prepare_redis.set(key, 1)
    assert get_param_from_ltm(key, 'string') == '1'
