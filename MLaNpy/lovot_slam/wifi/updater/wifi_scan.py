"""
WiFi scan module.
Periodically polls lovot-wifi service and emits the latest fingerprint.

e.g.
async def async_main():
    async with trio.open_nursery() as nursery:
        scan = WiFiScan()
        nursery.start_soon(scan.run)

        async for value, _ in scan.fingerprint_event.transitions():
            print(f'{value.monotonic_stamp:9.3f}: {len(value.access_points)}')

if __name__ == '__main__':
    trio.run(async_main)
"""
from logging import getLogger
from typing import Optional

import prometheus_client
import purerpc
from trio_util import AsyncValue, periodic

from lovot_slam import ContextMixin
from lovot_slam.wifi.type import StampedAccessPoints

_logger = getLogger(__name__)

_access_points_metric = prometheus_client.Gauge(
    'localization_wifi_fingerrpinting_access_points',
    'number of access points in available APs')


class WiFiScan(ContextMixin):
    """
    Periodically polls lovot-wifi service and emits the latest fingerprint.

    e.g.
    async with trio.open_nursery() as nursery:
        scan = WiFiScan()
        nursery.start_soon(scan.run)

        async for value, _ in scan.fingerprint_event.transitions():
            print(f'{value.monotonic_stamp:9.3f}: {len(value.access_points)}')
    """
    _PERIOD = 5

    def __init__(self) -> None:
        self._access_points_event = AsyncValue(None)

    @property
    def access_points_event(self) -> AsyncValue[Optional[StampedAccessPoints]]:
        return self._access_points_event

    @property
    def access_points(self) -> StampedAccessPoints:
        return self._access_points_event.value

    async def _get_fingerprint(self) -> Optional[StampedAccessPoints]:
        wifi_client = self.context.wifi_client
        try:
            res = await wifi_client.get_available_ap()
            if not res:
                return
        except purerpc.GRPCError as e:
            _logger.debug(f'Failed to get available AP: {e}')
            return

        try:
            return StampedAccessPoints.from_response(res)
        except ValueError as e:
            _logger.debug(f'Failed to parse access points: {e}')

    async def run(self):
        async for _ in periodic(self._PERIOD):
            access_points = await self._get_fingerprint()
            # NOTE: The equality is checked only by the stamps.
            # This is because some properties of the access points MAY CHANGE even with the same last_seen stamps.
            # In this case, only the first data is used.
            if not access_points \
                    or (self._access_points_event.value
                        and access_points.stamp == self._access_points_event.value.stamp):
                continue
            self._access_points_event.value = access_points
            _access_points_metric.set(len(access_points))
            _logger.debug(f'Available access points has been updated: {len(access_points)} aps'
                          f' and stamp of {access_points.stamp:.9f}')
