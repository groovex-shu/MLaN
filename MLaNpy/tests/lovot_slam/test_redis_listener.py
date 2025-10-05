import time

import pytest

from lovot_slam.redis.clients import create_stm_client
from lovot_slam.redis_listener import RedisListener


@pytest.fixture
def redis_cli():
    return create_stm_client()


def test_empty(redis_cli):
    channel = 'test:command'
    redis_listener = RedisListener(redis_cli, channel)
    redis_listener.start()
    time.sleep(0.1)
    _ = redis_listener.get()

    assert redis_listener.empty()

    redis_cli.publish(channel, 'test')
    time.sleep(0.1)
    assert not redis_listener.empty()
    _ = redis_listener.get()
    assert redis_listener.empty()

    redis_listener.stop()


def test_get(redis_cli):
    channel = 'test:command'
    redis_listener = RedisListener(redis_cli, channel)
    redis_listener.start()
    time.sleep(0.1)
    res = redis_listener.get()
    print(res)

    assert redis_listener.empty()

    redis_cli.publish(channel, 'test')
    time.sleep(0.1)
    assert not redis_listener.empty()

    res = redis_listener.get()
    assert res['data'] == 'test'

    redis_listener.stop()
