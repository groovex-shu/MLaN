import socket
from unittest.mock import AsyncMock

import anyio
import purerpc
import pytest
import trio
from google.protobuf import empty_pb2

from lovot_apis.lovot.navigation.domain_event_pb2 import HomeMapEvent, SpotEvent, UnwelcomedAreaEvent
from lovot_apis.lovot.navigation.navigation_pb2 import Coordinate, Spot, UnwelcomedArea
from lovot_apis.lovot.navigation.rpc_pb2 import (DeleteDestinationRequest, ResetMapRequest, SetDestinationRequest,
                                                 SetSpotCoordinateRequest, SetUnwelcomedAreaRequest)
from lovot_apis.lovot_app_api.navigation_grpc import NavigationServiceStub

from lovot_slam.redis.clients import create_ltm_client, create_stm_client
from lovot_slam.service.navigation_service import DESTINATION_CHANNEL, serve_navigation_service

from ..client.util import _get_free_port


@pytest.fixture(name="stub")
async def navigation_service_with_client(nursery):
    host = "localhost"
    port = await _get_free_port()

    mock_func = AsyncMock()
    nursery.start_soon(serve_navigation_service, mock_func, port)

    async with purerpc.insecure_channel(host, port) as channel:
        stub = NavigationServiceStub(channel)
        yield stub


@pytest.fixture(name="stub_localizer")
async def navigation_service_with_client_localizer(nursery):
    host = "localhost"
    port = await _get_free_port()

    nursery.start_soon(serve_navigation_service, None, port)

    async with purerpc.insecure_channel(host, port) as channel:
        stub_localizer = NavigationServiceStub(channel)
        yield stub_localizer


@pytest.fixture
def ltm_client_with_cleanup():
    keys = []
    ltm = create_ltm_client()

    def _func(*_keys):
        nonlocal keys
        keys = _keys
        return ltm

    yield _func

    for key in keys:
        ltm.delete(key)


async def test_reset_map(stub, ltm_client_with_cleanup, mock_httpx):
    # normal case
    ltm = ltm_client_with_cleanup("slam:map",
                                  "slam:unwelcomed_area",
                                  "slam:unwelcomed_area_hash",
                                  "slam:spot:test")
    ltm.hset("slam:map", "test_hash_key", "test_hash_value")
    ltm.set("slam:unwelcomed_area", "some_area")
    ltm.hset("slam:unwelcomed_area_hash", "test_hash_key", "test_hash_value")
    ltm.set("slam:spot:test", "spot_value")

    req = ResetMapRequest()
    reply = await stub.ResetMap(req)

    assert isinstance(reply, empty_pb2.Empty)
    unwelcomed_area = ltm.get("slam:unwelcomed_area")
    assert unwelcomed_area is None
    unwelcomed_area_hash = ltm.hgetall("slam:unwelcomed_area_hash")
    assert unwelcomed_area_hash == {}
    spot_val = ltm.get("slam:spot:test")
    assert spot_val is None

    # check event to cloud
    assert mock_httpx.call_count == 1
    assert mock_httpx.call_args[0][0] == 'http://localhost:48480/navigation/home-map-event'
    event = HomeMapEvent()
    event.ParseFromString(mock_httpx.call_args_list[0][1]["data"])
    assert event.event == HomeMapEvent.Event.home_map_reset


async def test_reset_map_localizer(stub_localizer, ltm_client_with_cleanup):
    # normal case
    ltm = ltm_client_with_cleanup("slack:map",
                                  "slam:unwelcomed_area",
                                  "slam:unwelcomed_area_hash",
                                  "slam:spot:test")
    ltm.hset("slam:map", "test_hash_key", "test_hash_value")
    ltm.set("slam:unwelcomed_area", "some_area")
    ltm.hset("slam:unwelcomed_area_hash", "test_hash_key", "test_hash_value")
    ltm.set("slam:spot:test", "spot_value")

    with pytest.raises(purerpc.CancelledError):
        req = ResetMapRequest()
        await stub_localizer.ResetMap(req)


async def test_set_unwelcomed_area(stub, ltm_client_with_cleanup, mock_httpx):
    # normal case
    ltm = ltm_client_with_cleanup("slam:map", "slam:unwelcomed_area")
    ltm.hset("slam:map", "test_hash_key", "test_hash_value")

    test_data = "[{\"shape\":\"polygon\",\"vertices\":[[-0.7597594,-0.07001495],[-3.105589,3.25611544],[-6.327943,0.983476639],[-3.98211312,-2.34265375]]}]"
    req = SetUnwelcomedAreaRequest(
        colony_id="test_colony_id",
        map_id="1",
        area_id="123",
        data=test_data
    )
    reply = await stub.SetUnwelcomedArea(req)

    assert isinstance(reply, UnwelcomedArea)
    re_data = ltm.get("slam:unwelcomed_area")
    assert re_data == test_data

    # check event to cloud
    assert mock_httpx.call_count == 1
    assert mock_httpx.call_args[0][0] == 'http://localhost:48480/navigation/unwelcomed-area-event'
    event = UnwelcomedAreaEvent()
    event.ParseFromString(mock_httpx.call_args_list[0][1]["data"])
    assert event.colony_id == "test_colony_id"
    assert event.map_id == "1"
    assert event.area_id == "123"
    assert event.event == UnwelcomedAreaEvent.Event.unwelcomed_area_updated
    assert event.area.data == test_data


async def test_set_unwelcomed_area_no_map(stub):
    # no map case
    test_data = "[{\"shape\":\"polygon\",\"vertices\":[[-0.7597594,-0.07001495],[-3.105589,3.25611544],[-6.327943,0.983476639],[-3.98211312,-2.34265375]]}]"
    req = SetUnwelcomedAreaRequest(
        colony_id="test_colony_id",
        map_id="1",
        area_id="123",
        data=test_data
    )
    with pytest.raises(purerpc.NotFoundError, match="No map exists"):
        await stub.SetUnwelcomedArea(req)


async def test_set_unwelcomed_area_wrong_map_id(stub, ltm_client_with_cleanup):
    # wrong map id
    ltm = ltm_client_with_cleanup("slam:map")
    ltm.hset("slam:map", "test_hash_key", "test_hash_value")

    test_data = "[{\"shape\":\"polygon\",\"vertices\":[[-0.7597594,-0.07001495],[-3.105589,3.25611544],[-6.327943,0.983476639],[-3.98211312,-2.34265375]]}]"
    req = SetUnwelcomedAreaRequest(
        colony_id="test_colony_id",
        map_id="2",
        area_id="123",
        data=test_data
    )
    with pytest.raises(purerpc.NotFoundError, match="map not found: 2"):
        await stub.SetUnwelcomedArea(req)


async def test_set_spot_coordinate(stub, ltm_client_with_cleanup, mock_httpx):
    # normal case
    ltm = ltm_client_with_cleanup("slam:map",
                                  "slam:spot:entrance")
    ltm.hset("slam:map", "test_hash_key", "test_hash_value")

    test_data = Coordinate(px=1.0, py=2.0, pz=3.0, ox=0.0, oy=0.0, oz=0.0, ow=1.0)
    req = SetSpotCoordinateRequest(
        colony_id="test_colony_id",
        map_id="1",
        spot_name="entrance",
        coordinate=test_data
    )
    reply = await stub.SetSpotCoordinate(req)

    assert isinstance(reply, Spot)
    re_data = ltm.hget("slam:spot:entrance", "coordinate")
    coordinate_format = "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}"
    coordinate_test = coordinate_format.format(
        test_data.px, test_data.py, test_data.pz, test_data.ox, test_data.oy, test_data.oz, test_data.ow)
    assert re_data.split(",") == coordinate_test.split(",")

    # check event to cloud
    assert mock_httpx.call_count == 1
    assert mock_httpx.call_args[0][0] == 'http://localhost:48480/navigation/spot-event'
    event = SpotEvent()
    event.ParseFromString(mock_httpx.call_args_list[0][1]["data"])
    assert event.colony_id == "test_colony_id"
    assert event.map_id == "1"
    assert event.spot_name == "entrance"
    assert event.event == SpotEvent.Event.spot_registered
    assert event.spot.coordinate == test_data


async def test_set_spot_coordinate_no_map(stub):
    # no map case
    test_data = Coordinate(px=1.0, py=2.0, pz=3.0, ox=0.0, oy=0.0, oz=0.0, ow=1.0)
    req = SetSpotCoordinateRequest(
        colony_id="test_colony_id",
        map_id="1",
        spot_name="entrance",
        coordinate=test_data
    )
    with pytest.raises(purerpc.NotFoundError, match="No map exists"):
        await stub.SetSpotCoordinate(req)


async def test_set_spot_coordinate_wrong_map_id(stub, ltm_client_with_cleanup):
    # wrong map id
    ltm = ltm_client_with_cleanup("slam:map")
    ltm.hset("slam:map", "test_hash_key", "test_hash_value")
    test_data = Coordinate(px=1.0, py=2.0, pz=3.0, ox=0.0, oy=0.0, oz=0.0, ow=1.0)
    req = SetSpotCoordinateRequest(
        colony_id="test_colony_id",
        map_id="2",
        spot_name="entrance",
        coordinate=test_data
    )
    with pytest.raises(purerpc.NotFoundError, match="map not found: 2"):
        await stub.SetSpotCoordinate(req)


async def test_set_spot_coordinate_wrong_spot_name(stub, ltm_client_with_cleanup):
    # wrong spot name
    ltm = ltm_client_with_cleanup("slam:map")
    ltm.hset("slam:map", "test_hash_key", "test_hash_value")
    test_data = Coordinate(px=1.0, py=2.0, pz=3.0, ox=0.0, oy=0.0, oz=0.0, ow=1.0)
    req = SetSpotCoordinateRequest(
        colony_id="test_colony_id",
        map_id="1",
        spot_name="wrong_name",
        coordinate=test_data
    )
    with pytest.raises(purerpc.InvalidArgumentError, match="spot name not supported: wrong_name"):
        await stub.SetSpotCoordinate(req)


async def test_set_destination(stub, nursery):
    stm = create_stm_client()
    pubsub = stm.pubsub()
    pubsub.subscribe(DESTINATION_CHANNEL)

    async def _check_msg():
        i = 2.0     # wait for 2 seconds
        while i > 0:
            message = pubsub.get_message()
            if message and message['type'] == 'message':
                data = message['data']
                assert data is not None
                assert data.split(",") == ["1.0000", "2.0000", "3.0000", "0.0000", "0.0000", "0.0000", "1.0000"]
                return

            await trio.sleep(0.1)
            i -= 0.1

        assert False, "message is not received within timeout"

    async def _check_response():
        # Set up the request and send it
        req = SetDestinationRequest(
            ghost_id="test_ghost_id",
            destination=Coordinate(px=1.0, py=2.0, pz=3.0, ox=0.0, oy=0.0, oz=0.0, ow=1.0)
        )
        reply = await stub.SetDestination(req)
        assert isinstance(reply, empty_pb2.Empty)

    nursery.start_soon(_check_response)
    nursery.start_soon(_check_msg)


async def test_delete_destination(stub, nursery):
    stm = create_stm_client()
    pubsub = stm.pubsub()
    pubsub.subscribe(DESTINATION_CHANNEL)

    async def _check_msg():
        i = 2.0     # wait for 2 second
        while i > 0:
            message = pubsub.get_message()
            if message and message['type'] == 'message':
                data = message['data']
                assert data is not None
                assert data == ""
                return

            await trio.sleep(0.1)
            i -= 0.1

        assert False, "message is not received within timeout"

    async def _check_response():
        # Set up the request and send it
        req = DeleteDestinationRequest(
            ghost_id="test_ghost_id"
        )
        reply = await stub.DeleteDestination(req)
        assert isinstance(reply, empty_pb2.Empty)

    nursery.start_soon(_check_response)
    nursery.start_soon(_check_msg)


async def test_server_fails_to_start(monkeypatch, autojump_clock):
    """Test that the server fails to start when getaddrinfo fails."""
    port = await _get_free_port()

    def getaddrinfo_raises(*args, **kwargs):
        raise socket.gaierror

    # monkeypatch.setattr(trio.socket, "getaddrinfo", getaddrinfo_raises)
    monkeypatch.setattr(anyio._core._sockets, "getaddrinfo", getaddrinfo_raises)

    with pytest.raises(RuntimeError, match="Failed to start server"):
        await serve_navigation_service(None, port)


async def test_server_starts_with_retry(monkeypatch, autojump_clock, nursery, mock_httpx):
    """Test that the server starts after failing to resolve the host address."""
    host = "localhost"
    port = await _get_free_port()

    def getaddrinfo_raises(*args, **kwargs):
        raise socket.gaierror

    async def reset_func():
        pass

    # backup original getaddrinfo
    org_getaddrinfo = getattr(anyio._core._sockets, "getaddrinfo")
    monkeypatch.setattr(anyio._core._sockets, "getaddrinfo", getaddrinfo_raises)

    nursery.start_soon(serve_navigation_service, reset_func, port)

    # it's supposed to fail and retry
    await trio.sleep(3)

    # reset monkeypatch only for getaddrinfo
    monkeypatch.setattr(anyio._core._sockets, "getaddrinfo", org_getaddrinfo)
    await trio.sleep(3)

    # check if the server is running
    async with purerpc.insecure_channel(host, port) as channel:
        stub = NavigationServiceStub(channel)
        req = ResetMapRequest()
        await stub.ResetMap(req)
