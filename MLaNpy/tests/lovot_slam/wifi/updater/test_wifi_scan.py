from contextlib import AsyncExitStack
from unittest.mock import Mock

import pytest
import trio

from lovot_apis.lovot_minid.wifi.wifi_pb2 import AP, GetAvailableAPResponse

from lovot_slam import Context, context
from lovot_slam.client import open_wifi_service_client
from lovot_slam.wifi.type import Ssid
from lovot_slam.wifi.updater.wifi_scan import WiFiScan

from ...client.test_wifi_service_client import MockWifiServicer
from ...client.util import open_servicer_and_client


@pytest.fixture
async def mock_context(nursery):
    async with AsyncExitStack() as stack:
        wifi_scan = WiFiScan()

        mock_wifi_service, wifi_client = \
            await stack.enter_async_context(open_servicer_and_client(MockWifiServicer, open_wifi_service_client))

        context.set(Context(
            slam_servicer_client=None,
            wifi_client=wifi_client,
            lovot_tf_client=Mock(),
            localization_client=Mock(),
            fingerprint_sync=Mock(),
            wifi_scan=wifi_scan,
            radio_map=Mock(),
        ))

        nursery.start_soon(wifi_scan.run)

        yield mock_wifi_service

        nursery.cancel_scope.cancel()


async def test_wifi_scan_event(mock_context, autojump_clock):
    mock_wifi_service = mock_context
    wifi_scan = context.get().wifi_scan

    mock_wifi_service.response = GetAvailableAPResponse(
        ap=[
            AP(ssid='test1',
               hw_address='00:00:00:00:00:00',
               strength=99,
               last_seen=1),
        ])

    # initial scan
    with trio.fail_after(WiFiScan._PERIOD * 2):
        await wifi_scan.access_points_event.wait_transition()
    ap = wifi_scan.access_points_event.value[Ssid(bytes.fromhex('000000000000'), 'test1')]
    assert ap.last_seen == 1
    assert ap.strength == 99

    # no change
    with pytest.raises(trio.TooSlowError):
        with trio.fail_after(WiFiScan._PERIOD * 2):
            await wifi_scan.access_points_event.wait_transition()

    # last_seenもstrengthも変わった
    mock_wifi_service.response = GetAvailableAPResponse(
        ap=[
            AP(ssid='test1',
               hw_address='00:00:00:00:00:00',
               strength=98,
               last_seen=2),
        ])

    # changed
    with trio.fail_after(WiFiScan._PERIOD * 2):
        await wifi_scan.access_points_event.wait_transition()
    ap = wifi_scan.access_points_event.value[Ssid(bytes.fromhex('000000000000'), 'test1')]
    assert ap.last_seen == 2
    assert ap.strength == 98

    # strengthは変わったけど、last_seenは同じ (実際にあり得るパターン)
    mock_wifi_service.response = GetAvailableAPResponse(
        ap=[
            AP(ssid='test1',
               hw_address='00:00:00:00:00:00',
               strength=0,
               last_seen=2),
        ])

    # no change (same last_seen with different strength)
    with trio.move_on_after(WiFiScan._PERIOD * 2):
        await wifi_scan.access_points_event.wait_transition()
    ap = wifi_scan.access_points_event.value[Ssid(bytes.fromhex('000000000000'), 'test1')]
    assert ap.last_seen == 2
    assert ap.strength == 98
