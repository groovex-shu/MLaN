import socket
from logging import getLogger

import purerpc
import redis
import trio
from google.protobuf import empty_pb2

from lovot_apis.lovot.navigation.domain_event_pb2 import HomeMapEvent, SpotEvent, UnwelcomedAreaEvent
from lovot_apis.lovot.navigation.navigation_pb2 import Spot, UnwelcomedArea
from lovot_apis.lovot.navigation.rpc_pb2 import (DeleteDestinationRequest, ResetMapRequest, SetDestinationRequest,
                                                 SetSpotCoordinateRequest, SetUnwelcomedAreaRequest)
from lovot_apis.lovot_app_api.navigation_grpc import NavigationServiceServicer

from lovot_slam.client.agent_client import upload_data_to_cloud
from lovot_slam.redis.clients import create_ltm_client, create_stm_client
from lovot_slam.redis.keys import redis_keys
from lovot_slam.service.purerpc_server import Server

# Constants
_NAVIGATION_SERVICE_PORT = 39050

DESTINATION_CHANNEL = "app_operation:destination:pose_tf"
FIXED_MAPID = "1"
FIXED_SPOTNAME = "entrance"

logger = getLogger(__name__)


class NavigationServiceImpl(NavigationServiceServicer):
    """Implementation of the NavigationService service.
    The service is defined in the following proto file:
    https://github.com/groove-x/lovot-apis/blob/master/lovot-app-api/navigation.proto

    The client of this service is the mobile-app and
    only some of the methods are used from it, so only those are implemented here.
    Not implemented methods will raise NotImplementedError.
    """

    def __init__(self, reset_func, nursery) -> None:
        if reset_func:
            self._reset_func = reset_func
        self._nursery = nursery
        self._stm = create_stm_client()
        self._ltm = create_ltm_client()

        logger.info("Navigation Service initialized")


    async def ResetMap(self, request: ResetMapRequest) -> empty_pb2.Empty:
        assert hasattr(self, "_reset_func"), "RestMap should not be called on localizer"
        
        # parse request
        colony_id = request.colony_id
        map_id = request.map_id

        # Cloud
        evt = HomeMapEvent(
            colony_id=colony_id,
            map_id=map_id,
            event=HomeMapEvent.Event.home_map_reset,
            home_map=None
        )
        
        res = await upload_data_to_cloud("navigation/home-map-event", evt.SerializePartialToString())
        if not res:
            raise purerpc.UnavailableError("Failed to upload data to cloud")

        # Local
        # Reset the map by event listener at main thread (SLAM の地図をリセットする)
        # nest_slam_manager._reset will do all the reset work
        logger.info("Navigation service call reset map from grpc client")
        await self._reset_func()
        logger.info("Reset map done")

        # Delete unwelcomed area & spots (玄関を削除)
        unwelcomed_area_key = redis_keys.unwelcomed_area
        unwelcomed_area_hash_key = redis_keys.unwelcomed_area_hash
        spot_keys = self._ltm.keys(redis_keys.spot("*"))
        self._ltm.delete(unwelcomed_area_key, unwelcomed_area_hash_key, *spot_keys)

        return empty_pb2.Empty()

    async def SetUnwelcomedArea(self, request: SetUnwelcomedAreaRequest) -> UnwelcomedArea:
        # parse request
        colony_id = request.colony_id
        map_id = request.map_id
        area_id = request.area_id
        # data: json string
        data = request.data

        # check map existence & map_id (現状では1つしか地図を扱わない)
        if self._ltm.exists(redis_keys.map) == 0:
            raise purerpc.NotFoundError("No map exists")

        if map_id != FIXED_MAPID:
            raise purerpc.NotFoundError(f"map not found: {map_id}")

        # create unwelcomed area
        unwelcomed_area = UnwelcomedArea(
            colony_id=colony_id,
            map_id=map_id,
            area_id=area_id,
            data=data
        )

        # Cloud
        evt = UnwelcomedAreaEvent(
            colony_id=colony_id,
            map_id=map_id,
            area_id=area_id,
            event=UnwelcomedAreaEvent.Event.unwelcomed_area_updated,
            area=unwelcomed_area
        )
        
        res = await upload_data_to_cloud("navigation/unwelcomed-area-event", evt.SerializePartialToString())
        if not res:
            raise purerpc.UnavailableError("Failed to upload data to cloud")

        # Local
        unwelcomed_area_key = redis_keys.unwelcomed_area
        self._ltm.set(unwelcomed_area_key, data)
        logger.info(f"Set unwelcomed area: {data}")

        return unwelcomed_area

    async def SetSpotCoordinate(self, request: SetSpotCoordinateRequest) -> Spot:
        # parse request
        colony_id = request.colony_id
        map_id = request.map_id
        spot_name = request.spot_name
        # coordinate: {px, py, pz, ox, oy, oz, ow}
        coordinate = request.coordinate

        # check map existence, map_id & spot name (現状では1つしか地図を扱わない)
        if self._ltm.exists(redis_keys.map) == 0:
            raise purerpc.NotFoundError("No map exists")

        if map_id != FIXED_MAPID:
            raise purerpc.NotFoundError(f"map not found: {map_id}")

        if spot_name != FIXED_SPOTNAME:
            raise purerpc.InvalidArgumentError(f"spot name not supported: {spot_name}")

        spot = Spot(
            colony_id=colony_id,
            map_id=map_id,
            name=spot_name,
            coordinate=coordinate
        )

        # Cloud
        evt = SpotEvent(
            colony_id=colony_id,
            map_id=map_id,
            spot_name=spot_name,
            event=SpotEvent.Event.spot_registered,
            spot=spot
        )
        
        res = await upload_data_to_cloud("navigation/spot-event", evt.SerializePartialToString())
        if not res:
            raise purerpc.UnavailableError("Failed to upload data to cloud")

        # Local
        spot_key = redis_keys.spot(spot_name)
        coordinate_format = "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}"
        coordinate_str = coordinate_format.format(coordinate.px, coordinate.py, coordinate.pz,
                                                  coordinate.ox, coordinate.oy, coordinate.oz, coordinate.ow)
        mapping = {
            "id": spot_name,
            "name": spot_name,
            "coordinate": coordinate_str
        }
        self._ltm.hset(spot_key, mapping=mapping)
        logger.info(f"Set spot coordinate: {coordinate_str}")

        return spot

    async def SetDestination(self, request: SetDestinationRequest) -> empty_pb2.Empty:
        coordinate = request.destination
        if coordinate is None:
            raise purerpc.StatusCode.INVALID_ARGUMENT("destination is required")

        coordinate_format = "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}"
        coordinate_str = coordinate_format.format(coordinate.px, coordinate.py, coordinate.pz,
                                                  coordinate.ox, coordinate.oy, coordinate.oz, coordinate.ow)

        try:
            self._stm.publish(DESTINATION_CHANNEL, coordinate_str)
            logger.info(f"Set destination: {coordinate_str}")

        except (redis.exceptions.RedisError) as redis_err:
            raise purerpc.InternalError(f"failed on redis: {redis_err}")

        return empty_pb2.Empty()

    async def DeleteDestination(self, request: DeleteDestinationRequest) -> empty_pb2.Empty:
        try:
            self._stm.publish(DESTINATION_CHANNEL, "")

        except (redis.exceptions.RedisError) as redis_err:
            raise purerpc.InternalError(f"failed on redis: {redis_err}")

        return empty_pb2.Empty()


async def serve_navigation_service(reset_func = None, port: int = _NAVIGATION_SERVICE_PORT):
    """Serve the navigation service on the given port.
    It doesn't return until the server is stopped.

    example:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(serve_navigation_service, 39050)
    """

    async with trio.open_nursery() as nursery:
        # specify ipv4 host to make serve_async fail intentionally if the host cannot be resolved
        # otherwise, it serves only on ipv6 and the client (lovot-agent) cannot connect to it
        server = Server('0.0.0.0', port)
        servicer = NavigationServiceImpl(reset_func, nursery)
        server.add_service(servicer.service)

        retry_count = 0
        while True:
            try:
                await server.serve_async()
            except socket.gaierror as e:
                logger.warning(f"Failed to start server: {e}")

            await trio.sleep(1)

            retry_count += 1
            if retry_count > 10:
                raise RuntimeError("Failed to start server")
