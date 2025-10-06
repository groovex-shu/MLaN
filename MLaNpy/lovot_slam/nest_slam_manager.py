from concurrent import futures
from logging import getLogger
from typing import Optional

import anyio
import grpc
from lovot_slam.protobufs.slam_pb2_grpc import add_SlamServicer_to_server

from lovot_slam.env import (GRPC_MAX_RECEIVE_MESSAGE_LENGTH, 
                            GRPC_PORT, 
                            MAPSET_ROOT_DIR, 
                            MAX_MAP_BUILD_FAILED_COUNT,
                            NestSlamState)
from lovot_slam.exploration.exploration_status import ExplorationStatusMonitor
from lovot_slam.exploration.exploration_token import ExplorationTokenManager
from lovot_slam.flags.debug_params import PARAM_ALWAYS_BUILD_MAP_ON_BUILDER
from lovot_slam.map_build.map_build import MapBuilder
from lovot_slam.map_build.request_queue import (MergeMapsOption,
                                                BuildOption, BuildSingleMapOption, 
                                                RequestQueue, RequestTypes)
from lovot_slam.model import LovotModel, NestModel
from lovot_slam.redis.keys import INTENTION_KEY, PHYSICAL_STATE_KEY
from lovot_slam.service.navigation_service import serve_navigation_service
from lovot_slam.slam_manager import SlamManager
from lovot_slam.slam_servicer import SlamServicer
from lovot_map.utils.map_utils import MAXIMUM_MERGE_NUMBER, MapSetUtils
from lovot_slam.utils.unwelcomed_area import set_unwelcomed_area_metric
from lovot_slam.utils.segmentation_monitor import SegmentationUpdater

logger = getLogger(__name__)

MAP_UPLOAD_RETRY_INTERVAL = 60  # seconds

# Builder
class NestSlamManager(SlamManager):
    def __init__(self, debug=False, journal=False):
        super().__init__(debug=debug, journal=journal)
        logger.info('initialize NestSlamManager')

        # FIXME when shaun's model will be fixed: read shaun's model
        try:
            self._model = NestModel.get()
            logger.info(f'model: {self._model}')
        except RuntimeError:
            self._model = LovotModel.get()
            logger.warning(f'falling back to lovot model: {self._model}')

        self.segmentation_updater = SegmentationUpdater(self.map_utils, self.spot_utils)

        self.requests = RequestQueue(self.redis_ltm)
        self.requests.load_from_redis()

        # map builder
        self._map_builder = MapBuilder(self.requests, self._is_running_on_nest(), self._model,
                                       debug, journal)
        # cancel scope for the current map building
        self._map_build_cancel_scope = anyio.CancelScope()
        # map build scheduler
        self._can_build_map = False

        # exploration status
        self._exploration_status_monitor = ExplorationStatusMonitor()

        # if this event is set, keep retrying to upload until succeeded
        self._map_update_event = anyio.Event()

        # coro2 -> use Lovot replace Nest
        is_ttl_from_waketime = True if isinstance(self._model, LovotModel) else False
        self._exploration_token_manager = ExplorationTokenManager(
            self.redis_ltm,
            self._exploration_status_monitor,
            self._map_builder.metrics,
            self._map_builder.is_processing_map,
            is_ttl_from_waketime,
            self.requests,
            self.map_utils)

        # multiple maps
        self._mapset_utils = MapSetUtils(MAPSET_ROOT_DIR)
        self._next_mapset: Optional[str] = None
        self._nursery_cancel_scope: Optional[anyio.CancelScope] = None


    def _is_running_on_nest(self) -> bool:
        """ Returns True if I'm running on nest (in case of Coro1), else False (in case of Coro2) """
        return isinstance(self._model, NestModel)

    def _start_grpc_server(self):
        # gRPC server
        # the worker thread is limited to 1 to prohibit concurrent process,
        # because some of the calls treat files and are not thread-safe.
        # process each request one by one.
        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[
            ('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
        add_SlamServicer_to_server(
            SlamServicer(self.requests, self._exploration_status_monitor, self._exploration_token_manager),
            self._grpc_server
        )
        self._grpc_server.add_insecure_port(f'[::]:{GRPC_PORT}')
        self._grpc_server.start()

    def _stop_grpc_server(self):
        self._grpc_server.stop(None)
        self._grpc_server.wait_for_termination()
        self._grpc_server = None

    async def _reset(self) -> None:
        logger.info('reset')
        # clear all requests before canceling the current map build
        self.requests.clear()
        await self._cancel_map_build_and_wait()

        self._map_builder.reset()
        self.segmentation_updater.inactivate()
        self.map_utils.clean_map_directory()
        self.map_utils.delete_map_from_redis_ltm()
        self.bag_utils.remove_all_bags()
        self.map_utils.reset_map_accuracy()
        self._exploration_status_monitor.reset()
        self.segmentation_updater.remove()

    def _trigger_mapset_change(self, mapset: str) -> None:
        """Trigger mapset change.
        This will stop coroutines in the current nursery,
        change the mapset by changing the symlink,
        and restart the nursery.
        """
        logger.info(f'trigger mapset change to {mapset}')
        self._next_mapset = mapset
        # NOTE: this assertion should be always true,
        # since this function is called only after nursery is started
        assert(self._nursery_cancel_scope)
        self._nursery_cancel_scope.cancel()

    def _change_mapset(self) -> None:
        """Change mapset by changing the symlink."""
        assert(self._next_mapset)

        logger.info(f'change mapset to {self._next_mapset}')
        self._mapset_utils.change_mapset(self._next_mapset)
        # create directories if not exists
        self.map_utils.create_directory()
        self.bag_utils.create_directory()

    async def process_command(self, req):
        req_cmd, req_id, req_args = self.parse_request(req)
        if req_cmd == '':
            self.publish_response('invalid_request', '', 'error')
            return

        elif req_cmd == 'build_map':
            logger.info('push build_map to queue')
            self.segmentation_updater.inactivate()
            if len(req_args) == 0:
                self.publish_response(req_cmd, req_id, 'error', err='bag_not_specified')
                return
            self.requests.push(RequestTypes.BuildMap, BuildSingleMapOption(req_args[0]))
            self.segmentation_updater.activate()
            self.publish_response(req_cmd, req_id, 'success')

        elif req_cmd == 'merge_maps':
            logger.info('push merge_maps to queue')
            self.segmentation_updater.inactivate()
            single_maps = self.map_utils.get_recent_single_mission_maps(MAXIMUM_MERGE_NUMBER)
            options = MergeMapsOption(single_maps[0])
            if len(single_maps) > 1:
                options.maps_to_append = single_maps[1:]
            self.requests.push(RequestTypes.MergeMaps, options)
            self.segmentation_updater.activate()
            self.publish_response(req_cmd, req_id, 'success')

        elif req_cmd == 'reset':
            await self._reset()
            self.publish_response(req_cmd, req_id, 'success')

        elif req_cmd == '_rebuild_segmentation':
            logger.info('rebuild segmentation (testing)')
            self.segmentation_updater.rebuild()
            self.publish_response(req_cmd, req_id, 'success')

        elif req_cmd == 'change_mapset':
            logger.info('change mapset')
            if len(req_args) == 0:
                self.publish_response(req_cmd, req_id, 'error', err='target_mapset_not_specified')
                return
            self._trigger_mapset_change(req_args[0])
            self.publish_response(req_cmd, req_id, 'success')

        else:
            logger.info('command not found. ' + req)

    def trigger_map_sync(self) -> None:
        """ Start map sync. If sync failed, sync is automatically retried until succeeded."""
        # Register the latest map to redis if it exists
        latest_map_name = self.map_utils.get_latest_merged_map()
        if latest_map_name:
            rosmap = self.map_utils.get_ros_map(latest_map_name)
            self.map_utils.update_redis(rosmap)

        self._map_update_event.set()

    async def _cancel_map_build_and_wait(self, timeout: float = 5.0) -> None:
        if not self._map_builder.is_processing_event.value:
            return

        logger.info('cancelling map build and wait for idle.')
        self._map_build_cancel_scope.cancel()
        # timeout is for fail-safe to prevent stuck
        with anyio.move_on_after(timeout):
            await self._map_builder.is_processing_event.wait_value(False)

    async def _process_build_map_request(self, req_type: RequestTypes, option: BuildOption) -> None:
        async def on_map_merged():
            # update exploration status (merged missions count and frontiers)
            self._exploration_status_monitor.update()
            self._exploration_status_monitor.transform_area_histories()

            # update redis and upload to cloud
            logger.info('registering the map.')
            self.trigger_map_sync()

        async def on_merge_failed():
            # if initial map update (1st merge) continuously failed, remove map
            continuous_merge_fail_count = \
                self._map_builder.metrics.get_continuous_fail_count(
                    False, NestSlamState.BUILD_FEATURE_MAP)
            if self.map_utils.get_maps_number_in_latest_merged_map() < 2 \
                    and continuous_merge_fail_count >= MAX_MAP_BUILD_FAILED_COUNT:
                logger.warning(f'initial map update has been failed over {MAX_MAP_BUILD_FAILED_COUNT} times, '
                               'reset current map')
                await self._reset()

        merge_success = None
        self._map_build_cancel_scope = anyio.CancelScope()
        with self._map_build_cancel_scope:
            if req_type == RequestTypes.BuildMap:
                assert isinstance(option, BuildSingleMapOption)
                merge_option = await self._map_builder.build_single_mission_map(map_name=option.map_name)
                if not merge_option:
                    return
                self.segmentation_updater.inactivate()
                merge_success = await self._map_builder.build_merged_map(option=merge_option)
                self.segmentation_updater.activate()

            elif req_type == RequestTypes.BuildSingleMissionMap:
                assert isinstance(option, BuildSingleMapOption)
                _ = await self._map_builder.build_single_mission_map(map_name=option.map_name)
                return

            elif req_type == RequestTypes.MergeMaps:
                assert isinstance(option, MergeMapsOption)
                self.segmentation_updater.inactivate()
                merge_success = await self._map_builder.build_merged_map(option=option)
                self.segmentation_updater.activate()

        if merge_success is None:
            pass
        elif merge_success:
            await on_map_merged()
        else:
            await on_merge_failed()

        if self._map_build_cancel_scope.cancelled_caught:
            logger.warning('map build was cancelled')

    async def _poll_build_map_request(self):
        while True:
            if self.requests.empty():
                await anyio.sleep(1)
                continue
            if not self._can_build_map:
                await anyio.sleep(1)
                continue

            req_type, request = self.requests.pop()
            await self._process_build_map_request(req_type, request)
            await anyio.sleep(1)

    async def _poll_unwelcomed_area(self):
        while True:
            unwelcomed_area = self._map_builder._spot_utils.get_unwelcomed_area_from_redis()
            _sum = sum([poly.area for poly in unwelcomed_area])
            set_unwelcomed_area_metric(_sum)
            await anyio.sleep(600)

    async def _monitor_build_map_availability(self) -> None:
        """Monitor physical state and intention to determine if we can build map.
        This is available only for shaun, else we always allow map building.
        - When lovot is ON_NEST and SLEEPY for more than 10 minuites,
          we can build map.
        - When lovot is not ON_NEST nor SLEEPY,
          we prohibit map building and cancel current map build if exists.
        """
        if self._is_running_on_nest():
            logger.info('always allow to build map on nest')
            self._can_build_map = True
            await anyio.sleep(float('inf'))
        if PARAM_ALWAYS_BUILD_MAP_ON_BUILDER:
            logger.warning('always allow to build map by debug param')
            self._can_build_map = True
            await anyio.sleep(float('inf'))

        while True:
            def can_build_map() -> bool:
                physical_state = self.redis_stm.get(PHYSICAL_STATE_KEY)
                intention = self.redis_stm.get(INTENTION_KEY)
                return (physical_state == 'ON_NEST' and
                        (intention == 'SLEEPY' or intention == 'INFINITE_EXPLORE'))

            can_build_map_now = can_build_map()
            if self._can_build_map and not can_build_map_now:
                logger.info('change to not allow build map')
                self._can_build_map = False
                if self._map_builder.is_processing_map():
                    logger.info('cancel current map build')
                    await self._cancel_map_build_and_wait()
            elif not self._can_build_map and can_build_map_now:
                # ネストから出されるかもしれないので、しばらく待ってフラグが安定しているならマップ作成
                intention = self.redis_stm.get(INTENTION_KEY)
                with anyio.move_on_after(10 * 60) as cancel_scope:
                    # INFINITE_EXPLORE モードの場合は、ネストから出されるようなことは起きにくい
                    # と思うので、すぐマップ作成する
                    if intention == 'INFINITE_EXPLORE':
                        cancel_scope.cancel()
                    while True:
                        if not can_build_map():
                            break
                        await anyio.sleep(5)
                if cancel_scope.cancelled_caught:
                    logger.info('change to allow build map')
                    self._can_build_map = True
            await anyio.sleep(1)

    async def _is_under_repair(self, timeout) -> bool:
        """ Check if the robot is under repair. If failed to get the status until timeout, return the last repair flag."""
        UNDER_REPAIR = "1"
        repair_flag = UNDER_REPAIR
        with anyio.move_on_after(timeout) as cancel_scope:
            while True:
                repair_flag = self.redis_stm.get("under_repair")
                ghost_status = self.redis_stm.get("cloud:ghost:status")

                if ghost_status == "identified":
                    logger.info(f"repair flag: {repair_flag}")
                    break
                await anyio.sleep(1)

        if cancel_scope.cancelled_caught:
            logger.warning(f"failed to get repair flag, use the last flag: {repair_flag}")

        return repair_flag == UNDER_REPAIR

    async def _ensure_map_consistency(self):
        """ Ensure the latest map is registered in redis and uploaded to cloud.
        If map is missing, remove map from cloud and remove spots and unwelcomed area."""
        CHECK_REPAIR_TIMEOUT = 100  # esconds
        under_repair = await self._is_under_repair(CHECK_REPAIR_TIMEOUT)
        if under_repair:
            return  # Do not sync map during repair

        map_update_flag = False
        while True:
            # Retry to sync the map with interval of MAP_UPLOAD_RETRY_INTERVAL
            # if map_update_flag is True.
            with anyio.move_on_after(MAP_UPLOAD_RETRY_INTERVAL):
                await self._map_update_event.wait()   # sync is triggered immediately when event is set
                map_update_flag = True

            self._map_update_event = anyio.Event()  # reset the event

            if map_update_flag:
                latest_map_name = self.map_utils.get_latest_merged_map()
                if latest_map_name:
                    rosmap = self.map_utils.get_ros_map(latest_map_name)
                    ready = self._exploration_status_monitor.status.is_ready()
                    if await self.map_utils.upload_map_to_cloud(rosmap, ready):
                        map_update_flag = False
                else:
                    # If map is missing, remove entrance and unwelcomed area from redis
                    self.spot_utils.remove_spot_from_redis("entrance")
                    self.spot_utils.set_unwelcomed_area_to_redis(None)

                    if await self.map_utils.reset_cloud_map():
                        map_update_flag = False

    async def _setup_context(self, stack) -> None:
        await anyio.sleep(0)

    async def _run_main(self):
        while True:
            self._start_grpc_server()  # grpc server

            self.trigger_map_sync()
            self._map_builder.remove_unused_resources()
            logger.info(f'map list: {self.map_utils.get_map_list(use_cache=False)}')

            async with anyio.create_task_group() as tg:
                tg.start_soon(self._exploration_status_monitor.run)
                tg.start_soon(self._poll_build_map_request)
                tg.start_soon(self._poll_unwelcomed_area)
                tg.start_soon(self._monitor_build_map_availability)
                tg.start_soon(self._ensure_map_consistency)
                tg.start_soon(self.segmentation_updater.run)
                tg.start_soon(serve_navigation_service, self._reset)
                self._nursery_cancel_scope = tg.cancel_scope
                logger.info('builder is started')

                # NOTE: this will exit only when changing the mapset

            self._nursery_cancel_scope = None

            self._stop_grpc_server()

            # change the target directory of the symlink
            if self._next_mapset:
                self._change_mapset()
            logger.info('restarting builder...')

    async def _stop(self):
        self._stop_grpc_server()
