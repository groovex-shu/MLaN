import pytest

from lovot_apis.lovot_minid.wifi.wifi_grpc import WifiServiceServicer
from lovot_apis.lovot_minid.wifi.wifi_pb2 import AP, GetAvailableAPResponse

from lovot_slam.client import wifi_service_client
from lovot_slam.client.wifi_service_client import open_wifi_service_client

from .util import open_servicer_and_client


class MockWifiServicer(WifiServiceServicer):
    def __init__(self) -> None:
        super().__init__()

        self.response = None
        self.request = None

    async def GetAvailableAP(self, request):
        assert self.response
        self.request = request
        return self.response


@pytest.fixture
async def open_mock_servicer_and_client(monkeypatch, nursery):
    # NOTE: サーバの起動に時間がかかることがあるので、タイムアウトを長めに設定する
    monkeypatch.setattr(wifi_service_client, 'GRPC_TIMEOUT', 10.0)

    async with open_servicer_and_client(MockWifiServicer, open_wifi_service_client) as (mock_service, client):
        yield mock_service, client


@pytest.mark.parametrize(
    "expected_response",
    [
        (GetAvailableAPResponse(
            ap=[
                AP(ssid='test1',
                   hw_address='00:00:00:00:00:00',
                   strength=99,
                   last_seen=1),
            ])),
        (GetAvailableAPResponse(
            ap=[
                AP(ssid='test1',
                   hw_address='00:00:00:00:00:00',
                   strength=99,
                   last_seen=1),
                AP(ssid='test2',
                   hw_address='01:02:00:00:00:00',
                   strength=95,
                   last_seen=0),
            ])),
    ]
)
async def test_wifi_service_client(open_mock_servicer_and_client, expected_response: GetAvailableAPResponse):
    mock_servicer, client = open_mock_servicer_and_client
    mock_servicer.response = expected_response

    res = await client.get_available_ap()

    assert res == expected_response
