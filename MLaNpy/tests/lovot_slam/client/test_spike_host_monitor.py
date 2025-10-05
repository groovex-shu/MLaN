import socket

import pytest
import trio

from lovot_slam.client.spike_host_monitor import SpikeHostMonitor
from lovot_slam.redis import create_ltm_client

DUMMY_MDNS = {
    'DN00000XXXXXXXXXXXXX.local': [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('172.23.0.1', 0))],
    'DN00000YYYYYYYYYYYYY.local': [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('172.23.0.2', 0))],
    'DN00000ZZZZZZZZZZZZZ.local': [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 0))],
}


@pytest.fixture
def redis_fixture():
    ltm_client = create_ltm_client()

    ltm_client.delete(SpikeHostMonitor._COLONY_NEST_DEVICE_ID_KEY)

    yield ltm_client

    ltm_client.delete(SpikeHostMonitor._COLONY_NEST_DEVICE_ID_KEY)


@pytest.fixture
def mock_socket_getaddreinfo(monkeypatch):
    async def mock_getaddrinfo(host_name):
        try:
            return DUMMY_MDNS[host_name]
        except KeyError:
            raise socket.gaierror()

    from lovot_slam.client import spike_host_monitor
    monkeypatch.setattr(spike_host_monitor, '_getaddrinfo', mock_getaddrinfo)


async def test_get_addr_fallback(redis_fixture, autojump_clock):
    monitor = SpikeHostMonitor()
    assert await monitor.get_addr() == SpikeHostMonitor._FALLBACK_ADDRESS


async def test_update_host_from_device_id(redis_fixture, nursery, mock_socket_getaddreinfo, autojump_clock):
    monitor = SpikeHostMonitor()
    nursery.start_soon(monitor.run)

    # device_id is not set
    with pytest.raises(trio.TooSlowError):
        with trio.fail_after(10):
            addr = await monitor.spike_host_event.wait_value(lambda v: v)
    assert await monitor.get_addr() == SpikeHostMonitor._FALLBACK_ADDRESS

    # device_id is set
    redis_fixture.set(SpikeHostMonitor._COLONY_NEST_DEVICE_ID_KEY, 'DN00000XXXXXXXXXXXXX')
    with trio.fail_after(SpikeHostMonitor._RETRY_INTERVAL + 10):
        addr = await monitor.spike_host_event.wait_value(lambda v: v)
    assert addr == '172.23.0.1'
    assert await monitor.get_addr() == '172.23.0.1'

    # device_id is changed
    redis_fixture.set(SpikeHostMonitor._COLONY_NEST_DEVICE_ID_KEY, 'DN00000YYYYYYYYYYYYY')
    with trio.fail_after(SpikeHostMonitor._UPDATE_INTERVAL + 10):
        addr, _ = await monitor.spike_host_event.wait_transition(lambda v, _: v)
    assert addr == '172.23.0.2'
    assert await monitor.get_addr() == '172.23.0.2'
