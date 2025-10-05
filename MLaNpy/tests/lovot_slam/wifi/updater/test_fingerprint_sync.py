from contextlib import AsyncExitStack
from unittest.mock import Mock

import pytest
import trio

from lovot_apis.lovot_minid.wifi.wifi_pb2 import AP, GetAvailableAPResponse
from lovot_apis.lovot_tf.tf.tf_pb2 import GetTransformResponse, Header, Quaternion, Transform, TransformStamped, Vector3

from lovot_slam import Context, context
from lovot_slam.client import open_localization_client, open_lovot_tf_client, open_wifi_service_client
from lovot_slam.wifi.type import Ssid, StampedAccessPoints
from lovot_slam.wifi.updater.fingerprint_sync import FingerprintSync
from lovot_slam.wifi.updater.wifi_scan import WiFiScan

from ...client.test_lovot_tf_client import MockTfServicer
from ...client.test_wifi_service_client import MockWifiServicer
from ...client.util import open_servicer_and_client


@pytest.fixture
async def mock_context(nursery):
    async with AsyncExitStack() as stack:
        wifi_scan = WiFiScan()
        fingerprint_sync = FingerprintSync()

        mock_wifi_service, wifi_client = \
            await stack.enter_async_context(open_servicer_and_client(MockWifiServicer, open_wifi_service_client))
        mock_tf_service, lovot_tf_client = \
            await stack.enter_async_context(open_servicer_and_client(MockTfServicer, open_lovot_tf_client))
        
        localization_client = open_localization_client()
        
        context.set(Context(
            slam_servicer_client=None,
            wifi_client=wifi_client,
            lovot_tf_client=lovot_tf_client,
            localization_client=localization_client,
            fingerprint_sync=fingerprint_sync,
            wifi_scan=wifi_scan,
            radio_map=Mock(),
        ))

        nursery.start_soon(wifi_scan.run)
        nursery.start_soon(fingerprint_sync.run)

        yield mock_wifi_service, mock_tf_service

        nursery.cancel_scope.cancel()


async def test_fingerprint_sync(mock_context, autojump_clock):
    mock_wifi_service, mock_tf_service = mock_context

    mock_wifi_service.response = GetAvailableAPResponse(
        ap=[
            AP(ssid='test1',
               hw_address='00:00:00:00:00:00',
               strength=99,
               last_seen=1),
        ])

    fingerprint_event = context.get().fingerprint_sync.fingerprint_event

    await trio.sleep(10)
    assert fingerprint_event.value is None

    mock_tf_service.response = GetTransformResponse(
        transform_stamped=TransformStamped(
            header=Header(frame_id='map'),
            child_frame_id='base',
            transform=Transform(
                translation=Vector3(x=1, y=2, z=3),
                rotation=Quaternion(x=0, y=0, z=0, w=1),
            )))
    mock_wifi_service.response = GetAvailableAPResponse(
        ap=[
            AP(ssid='test1',
               hw_address='00:00:00:00:00:00',
               strength=99,
               last_seen=2),
        ])

    with trio.fail_after(20):
        fingerprint, _ = await fingerprint_event.wait_transition(lambda v, _: v is not None)
    assert fingerprint.transform.transform == Transform(
        translation=Vector3(x=1, y=2, z=3),
        rotation=Quaternion(x=0, y=0, z=0, w=1),
    )
    expected_aps_dict = {Ssid(bssid=b'\x00\x00\x00\x00\x00\x00', essid='test1'):
                         AP(ssid='test1',
                            hw_address='00:00:00:00:00:00',
                            strength=99,
                            last_seen=2)}
    assert fingerprint.access_points == StampedAccessPoints(expected_aps_dict, fingerprint.access_points.stamp)
