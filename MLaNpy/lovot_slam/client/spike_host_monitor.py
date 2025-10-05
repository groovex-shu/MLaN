import socket
from logging import getLogger
from typing import Optional

import redis
import trio
from trio_util import AsyncValue

from lovot_slam.redis import create_ltm_client

_logger = getLogger(__name__)


async def _getaddrinfo(host_name: str) -> Optional[str]:
    return await trio.socket.getaddrinfo(
        host_name, None, family=socket.AF_INET, type=socket.SOCK_STREAM)


class SpikeHostMonitor:
    """Periodically check the spike host and update the value if it is changed.

    Host address resolution is done by the following order:
    1. Check the colony:nest:device_id in redis, and resolve its address
    2. If it fails (no key in redis or name resolution fails), use the fallback address

    e.g.
    monitor = SpikeHostMonitor()
    async with trio.open_nursery() as nursery:
        nursery.start_soon(monitor.run)
        async for addr, _ in self._spike_host_event.transitions():
            print(addr)
    """
    _COLONY_NEST_DEVICE_ID_KEY = 'colony:nest:device_id'
    _FALLBACK_ADDRESS = '192.168.88.1'
    _UPDATE_INTERVAL = 30 * 60
    _RETRY_INTERVAL = 10

    def __init__(self):
        self._spike_host_event = AsyncValue(None)
        self._nest_device_id = None

        self._ltm_client = create_ltm_client()

    @property
    def spike_host_event(self) -> AsyncValue[str]:
        return self._spike_host_event

    async def get_addr(self, timeout: float = 5) -> str:
        with trio.move_on_after(timeout):
            return await self._spike_host_event.wait_value(lambda v: v)
        return self._FALLBACK_ADDRESS

    async def _get_host_by_name(self, host_name: str) -> Optional[str]:
        try:
            _logger.debug(f'resolving host name: {host_name}')
            addrinfo = await _getaddrinfo(host_name)
            if not addrinfo or len(addrinfo[0]) != 5:
                return
            resolved_ip = addrinfo[0][4][0]
            _logger.debug(f'host found: {resolved_ip}')
            return resolved_ip
        except socket.gaierror:
            _logger.debug(f'unable to resolve IP from {host_name}')

    async def _get_spike_host_by_device_id(self) -> Optional[str]:
        try:
            nest_device_id = self._ltm_client.get(self._COLONY_NEST_DEVICE_ID_KEY)
            if not nest_device_id:
                _logger.debug("colony's spike not found in redis.")
                return
            self._nest_device_id = nest_device_id
            return await self._get_host_by_name(f'{nest_device_id}.local')
        except redis.RedisError as exc:
            _logger.debug(f"redis connection failed: {exc}")

    async def run(self):
        async with trio.open_nursery() as nursery:
            @nursery.start_soon
            async def log_on_update():
                async for value, _ in self._spike_host_event.transitions():
                    _logger.info(f'Spike address has been updated: {value}'
                                 f'{" (" + self._nest_device_id + ")" if self._nest_device_id else ""}')

            while True:
                spike_host = await self._get_spike_host_by_device_id()
                if not spike_host:
                    # retry soon if failed to get spike host
                    await trio.sleep(self._RETRY_INTERVAL)
                    continue
                self._spike_host_event.value = spike_host
                await trio.sleep(self._UPDATE_INTERVAL)
