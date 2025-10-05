import datetime
import os
from contextlib import AsyncExitStack
from functools import partial
from logging import getLogger
from typing import List, Optional

import numpy as np
import prometheus_client
import rosgraph
import trio
from trio_util import AsyncBool, AsyncValue, RepeatedEvent, periodic, wait_any

from lovot_slam import Context, ContextMixin, context
from lovot_slam.client import open_localization_client, open_lovot_tf_client, open_wifi_service_client
from lovot_slam.env import MAP_FEATUREMAP, MAP_SUMMARYMAP, OMNI_CAMERA_YAML, PUSHBAG_RETRY_INTERVAL
from lovot_slam.model import HardwareVariants, LovotModel
from lovot_slam.monitor.reliability_monitor import LocalizationReliabilityMonitor
from lovot_slam.monitor.ros_node_monitor import RosNodeMarkerLocalizationMonitor
from lovot_slam.redis import redis_keys
from lovot_slam.redis.clients import create_stm_client
from lovot_slam.service.navigation_service import serve_navigation_service
from lovot_slam.slam_manager import SlamManager
from lovot_slam.spike_client import open_slam_servicer_client
from lovot_slam.subprocess import BaseSubprocess, LocalizationSubprocess, RecordSubprocess
from lovot_slam.utils.exceptions import (SlamMapError, SlamProcedureCallError, SlamProcessError, SlamSensorError,
                                         SlamTransferError)
from lovot_slam.utils.map_utils import BagUtils
from lovot_slam.utils.omni_camera_mode import CameraOperationMode, OmniCameraMode, OmniCameraMode2
from lovot_slam.utils.segmentation_monitor import SegmentationDownloader
from lovot_slam.utils.unwelcomed_area import calc_unwelcomed_area_hash_from_str
from lovot_slam.wifi.evaluator.evaluator import InferenceEvaluator
from lovot_slam.wifi.mapping import Mapping, RadioMap
from lovot_slam.wifi.updater import FingerprintSync, WiFiScan

logger = getLogger(__name__)

SERVICER_HOST = os.getenv('LOCALIZATION_SERVICER_HOST')
MARKERS_POSITION_KEY = 'slam:markers_position'

_ros_node_shutdown_metric = prometheus_client.Counter(
    'localization_ros_node_shutdown_count', 'ros node shutdown count',
    labelnames=['node']
)


async def _raise_if_rosmaster_is_online(timeout: float = 5):
    with trio.move_on_after(timeout) as cancel_scope:
        async for _ in periodic(1):
            if not rosgraph.is_master_online():
                break
    if cancel_scope.cancelled_caught:
        logger.warning("rosmaster is still alive after all launches are terminated")
        raise RuntimeError("inconsistency between rosmaster and ros nodes")


class _RecordProcessController:
    MAX_RECORD_DURATION = 5 * 60
    CLEANUP_TIMEOUT_SEC = 20.0

    def __init__(self, base_process: BaseSubprocess, bag_utils: BagUtils, variants: HardwareVariants,
                 output_to_console: bool = False) -> None:
        self._variants = variants

        # Switch OmniCameraMode based on the model
        self._omni_mode = OmniCameraMode() if variants.model < LovotModel.LV110 else OmniCameraMode2()
        logger.info(f'initialize _omni_mode as {type(self._omni_mode)}')

        self._stm_client = create_stm_client()

        self._record_process = RecordSubprocess(variants, output_to_console=output_to_console)

        self._base_process = base_process
        self._bag_utils = bag_utils

        self._start_event = RepeatedEvent()
        self._stop_event = AsyncBool(False)
        self._is_recording = AsyncBool(False)
        self._record_name = None

    async def start(self):
        """Trigger starting record subprocess.
        :returns map name if it's successfully started, else None in 10 seconds
        it always takes 10 seconds to detect any failures.
        """
        if self._is_recording.value:
            logger.warning("failed to start record process, because it's already started")
            return None
        self._stop_event.value = False
        self._start_event.set()
        with trio.move_on_after(10):
            await self._is_recording.wait_value(True)
            return self._record_name
        return None

    async def stop(self):
        """Trigger stopping record subprocess.
        :returns True if the subprocess is successfully stopped.
        """
        if not self._is_recording.value:
            logger.warning("failed to stop record process, because it's not started")
            return False
        self._stop_event.value = True
        with trio.move_on_after(10):
            await self._is_recording.wait_value(False)
            return True
        return False

    @property
    def is_recording(self):
        return self._is_recording.value

    async def _start_record_process(self, map_name) -> bool:
        if self._omni_mode and \
                not await self._omni_mode.change_mode_and_wait(CameraOperationMode.RECORD_MODE, 2):
            logger.info('failed to change omni camera mode and does not start recording')
            return False
        bag_file = self._bag_utils.get_full_path(map_name)
        logger.info('starting record ros nodes')
        self._record_process.start(bag_file)
        return True

    async def _stop_record_process(self):
        logger.info('stopping record ros nodes')
        try:
            await self._record_process.stop_process_and_wait()
        finally:
            if self._omni_mode:
                await self._omni_mode.change_mode_and_wait(CameraOperationMode.NORMAL_MODE, 2)

    def enable_tof_stream(self):
        self._stm_client.hset('tracking_module:settings', 'enable_tof_stream', 1)

    async def _control_record(self):
        async def monitor_record_process():
            nonlocal remove_bag
            await self._record_process.wait_for_termination()
            logger.error('record stopped unexpectedly.')
            self._record_process.dump_stderr()
            self._record_process.clear()
            remove_bag = True

        async for _ in self._start_event.unqueued_events():
            logger.info("starting record process")
            if not self._base_process.is_running():
                logger.warning("failed to start record process, because base nodes are not running")
                continue

            self._bag_utils.remove_all_bags()

            # enable TOF stream just in case that it hasn't been enabled.
            # usually record is triggered by neodm who should enable it before recording
            # so this is just a safety measure only during development.
            # NOTE: it's not disabled after recording
            self.enable_tof_stream()

            # start recording
            remove_bag = False
            self._record_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not await self._start_record_process(self._record_name):
                logger.warning('failed to start record process due to an error')
                self._record_name = None
                continue
            self._is_recording.value = True

            # wait for the stop command or unexpected shutdown of the process
            with trio.move_on_after(self.MAX_RECORD_DURATION) as cancel_scope:
                await wait_any(
                    partial(self._stop_event.wait_value, True),
                    monitor_record_process
                )
            if cancel_scope.cancelled_caught:
                logger.warning('record timeout, stopping recording and remove all rosbags')
                remove_bag = True

            # stop recording
            try:
                await self._stop_record_process()
            except SlamProcessError:
                logger.warning('failed to stop record process due to an error')
            finally:
                if remove_bag:
                    self._bag_utils.remove_all_bags()
                self._record_name = None
                self._stop_event.value = False
                self._is_recording.value = False

    async def initialize(self) -> bool:
        """Initialize camera mode.
        :return: True if success, else False
        """
        if self._omni_mode:
            result = await self._omni_mode.change_mode_and_wait(CameraOperationMode.NORMAL_MODE, 3)
            logger.info(f'omni camera mode is set to default: {result}')
            return result
        await trio.sleep(0)
        return True

    async def run(self):
        try:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(self._control_record)
        finally:
            with trio.move_on_after(self.CLEANUP_TIMEOUT_SEC) as cleanup_scope:
                cleanup_scope.shield = True
                if self._is_recording.value:
                    await self._stop_record_process()
                if self._omni_mode:
                    result = await self._omni_mode.change_mode_and_wait(CameraOperationMode.NORMAL_MODE, 3)
                    logger.info(f'omni camera mode is set to default: {result}')


class LovotSlamManager(SlamManager, ContextMixin):
    RESTART_INTERVAL_AFTER_CRASH_SEC = 20

    def __init__(self, default_map='', debug=False, journal=False):
        super().__init__(debug=debug, journal=journal)
        logger.info('initialize LovotSlamManager')

        self._variants = HardwareVariants.get()

        # load a map
        self.map_name = None
        self._map_updated = False
        map_name = self.map_utils.get_latest_merged_map()
        if default_map != '':
            map_name = default_map
        if map_name and not self._update_map(map_name):
            self.map_utils.remove(map_name)

        # running subprocess info
        self.base_process = BaseSubprocess(
            self._variants,
            output_to_console=self._subprocess_output_to_console)
        self.localization_process = LocalizationSubprocess(
            self._variants,
            output_to_console=self._subprocess_output_to_console)

        self._record_controller = _RecordProcessController(
            self.base_process, self.bag_utils, self._variants,
            output_to_console=self._subprocess_output_to_console)

        # Filename of rosbag if failed to transfer, used to re-transfer
        self._failed_push_bag_file: AsyncValue[Optional[str]] = AsyncValue(self.get_failed_bag_file())

        self.segmentation_downloader = SegmentationDownloader(self.map_utils, self.spot_utils)

        self._reliability_monitor = LocalizationReliabilityMonitor()
        self._marker_localization_monitor = RosNodeMarkerLocalizationMonitor()

        # WiFi fingerprinting
        self._wifi_scan = WiFiScan()
        self._fingerprint_sync = FingerprintSync()
        self._wifi_mapping = Mapping()
        self._wifi_evaluator = InferenceEvaluator()

        # model type
        self._model = LovotModel.get()

    def _update_map(self, map_name: str):
        if self.map_utils.check_map(map_name):
            self.map_name = map_name
            self.map_utils.set_map_to_redis_stm(map_name)
            self._map_updated = True
            return True
        return False

    def _reset_marker_info(self):
        removed_count = self.redis_ltm.delete(MARKERS_POSITION_KEY)
        logger.info(f"Cleared marker data. Keys removed: {removed_count}")

    async def _undeploy_map(self) -> None:
        logger.info('undeploying map...')
        self.map_utils.reset_map_accuracy()
        self.map_utils.remove_all_maps()
        self.map_utils.delete_map_from_redis_stm()
        self.map_name = None
        self.segmentation_downloader.remove()
        await self._wifi_mapping.remove_map()
        self._reset_marker_info()

    def get_failed_bag_file(self) -> Optional[str]:
        # TODO: move this to BagUtils and add unit test
        # check if there is transfer-failed rosbag file,
        # if exist return its filename, otherwise None.
        # Assuming the number of bags should be only one at a maximum.
        bag_files = list(self.bag_utils.root.glob('*.bag'))
        if not bag_files:
            return None

        return bag_files[0].stem

    async def restart_base(self) -> bool:
        try:
            # stop other launch files before restarting base,
            # because rosmaster, which is launched by base launch, is required by other launch files
            if self._record_controller.is_recording:
                await self._record_controller.stop()
            await self.localization_process.stop_process_and_wait()
            # restart base
            await self.base_process.stop_process_and_wait()
            self.base_process.start(OMNI_CAMERA_YAML)
            # wait for rosmaster is online
            with trio.fail_after(10):
                while not rosgraph.is_master_online():
                    await trio.sleep(1)
        except SlamProcessError:
            logger.error("starting base roslaunch error")
            return False
        except trio.TooSlowError:
            logger.error("starting base roslaunch timeout")
            return False
        logger.info('base ros nodes started')
        return True

    async def retry_push_bag(self, map_name):
        try:
            await self.context.slam_servicer_client.upload_rosbag(map_name, self.bag_utils.get_full_path(map_name))
            logger.info('retry_push_bag: start build_map')
            await self.context.slam_servicer_client.build_single_mission_map(map_name)
            self.bag_utils.remove(map_name)
            return True
        except SlamTransferError:
            logger.error("retry_push_bag: SlamTransferError")
        except OSError:
            logger.error("retry_push_bag: file remove error")
        except SlamProcedureCallError:
            logger.error("retry_push_bag: SlamProcedureCallError")

        return False

    async def stop_localization(self) -> bool:
        try:
            await self.localization_process.stop_process_and_wait()
        except SlamProcessError:
            logger.error("Failed to stop localization nodes.")
            return False
        return True

    async def start_localization(self, map_name: str) -> bool:
        if not self.map_utils.check_map(map_name):
            logger.error("Failed to (re)start localization due to map error.")
            await self._undeploy_map()
            return False

        try:
            vi_map_folder = MAP_SUMMARYMAP if self.map_utils.has_summary_map(map_name) \
                else MAP_FEATUREMAP
            self.localization_process.start(map_name, vi_map_folder)
        except SlamProcessError:
            logger.error("Failed to (re)start localization nodes.")
            return False
        logger.info('localization ros nodes started')
        return True

    def cancel_retry_push_bag(self):
        # give up retrying pushing bag
        if self._failed_push_bag_file.value is not None:
            logger.info(
                f"pushbag retry {self._failed_push_bag_file.value} is cancelled")
            self._failed_push_bag_file.value = None

    async def _monitor_retry_push_bag(self):
        while True:
            await self._failed_push_bag_file.wait_value(lambda val: val is not None)
            with trio.move_on_after(PUSHBAG_RETRY_INTERVAL) as cancel_scope:
                await self._failed_push_bag_file.wait_value(lambda val: val is None)
            if cancel_scope.cancelled_caught:
                logger.info('detected a un-transferred bag file, retrying push_bag')
                if await self.retry_push_bag(self._failed_push_bag_file.value):
                    logger.info('retrying push_bag succeeded')
                    self._failed_push_bag_file.value = None
                else:
                    logger.warning('retrying push_bag failed')

    async def process_command(self, req):
        req_cmd, req_id, req_args = self.parse_request(req)
        if req_cmd == '':
            return

        elif req_cmd == 'record_start':
            logger.info('command: record_start')
            if self._record_controller.is_recording:
                await self._record_controller.stop()

            self.cancel_retry_push_bag()

            map_name = await self._record_controller.start()
            if map_name:
                self.publish_response(req_cmd, req_id, 'success', map_name)
                return
            logger.warning("failed to start record process")
            self.publish_response(req_cmd, req_id, 'error', 'process_error')

        elif req_cmd == 'record_stop':
            logger.info('command: record_stop')
            if not self._record_controller.is_recording:
                logger.warning('record is already stopped')
                self.publish_response(req_cmd, req_id, 'error', 'state_error')
                return
            if await self._record_controller.stop():
                self.publish_response(req_cmd, req_id, 'success')
                return
            self.publish_response(req_cmd, req_id, 'error', 'process_error')

        elif req_cmd == 'transfer_start':
            logger.info('command: transfer_start')

            self.cancel_retry_push_bag()

            if len(req_args) < 1:
                logger.error('transfer_start: data name required')
                self.publish_response(req_cmd, req_id, 'error', 'data_name_required')
                return
            if self._record_controller.is_recording:
                logger.error('failed to transfer bag: still recording')
                self.publish_response(req_cmd, req_id, 'error', 'state_error')
                return

            map_name = req_args[0]
            try:
                await self.context.slam_servicer_client.upload_rosbag(map_name, self.bag_utils.get_full_path(map_name))
                self.bag_utils.remove(map_name)
                await self.context.slam_servicer_client.build_single_mission_map(map_name)
                self.publish_response(req_cmd, req_id, 'success')
            except SlamTransferError:
                logger.error("transfer_start: push_bag SlamTransferError")
                self.publish_response(req_cmd, req_id, 'error', 'transfer_error')
                self._failed_push_bag_file.value = map_name  # scheduling retry
            except OSError:
                logger.error("transfer_start: file remove error")
                self._failed_push_bag_file.value = map_name  # scheduling retry
            except SlamProcedureCallError:
                logger.error("transfer_start: SlamProcedureCallError")
                self.publish_response(req_cmd, req_id, 'error')
                self._failed_push_bag_file.value = map_name  # scheduling retry

        elif req_cmd == 'change_map':
            logger.info('command: change_map')
            if self._record_controller.is_recording:
                logger.error('failed to change map: currently recording')
                self.publish_response(req_cmd, req_id, 'error', 'state_error')
                return
            if len(req_args) == 0:
                logger.error('change_map: data name required')
                self.publish_response(req_cmd, req_id, 'error', 'data_name_required')
                return
            try:
                map_name = req_args[0]
                if map_name == 'latest' or not map_name:
                    map_name = await self.context.slam_servicer_client.get_latest_map()
                map_path = self.map_utils.get_full_path(map_name)
                logger.info(f'downloading {map_name} to {map_path}')
                await self.context.slam_servicer_client.download_map(map_name, map_path)
                if not self._update_map(map_name):
                    self.map_utils.remove(map_name)
                    raise SlamMapError
                self.map_utils.remove_all_maps_except(map_name)
                self.publish_response(req_cmd, req_id, 'success', map_name)
            except SlamTransferError:
                logger.error('change_map: SlamTransferError')
                self.publish_response(req_cmd, req_id, 'error', 'transfer_error')
            except SlamMapError:
                logger.error('change_map: SlamMapError')
                self.publish_response(req_cmd, req_id, 'error', 'map_data_error')
            except SlamProcessError:
                logger.error('change_map: SlamProcessError')
                self.publish_response(req_cmd, req_id, 'error', 'process_error')
            except SlamProcedureCallError:
                self.publish_response(req_cmd, req_id, 'error')

        elif req_cmd == 'request_map_list':
            logger.info('command: request_map_list')
            try:
                map_list = await self.context.slam_servicer_client.map_list()
                logger.debug(f'map_list: {map_list}')
                str_map_list = ' '.join(map_list)
                self.publish_response(req_cmd, req_id, 'success', str_map_list)
            except SlamProcedureCallError:
                logger.error('request_map_list: SlamProcedureCallError')
                self.publish_response(req_cmd, req_id, 'error', 'rpc_error')
            except TypeError:
                logger.error('request_map_list: type error')
                self.publish_response(req_cmd, req_id, 'error', 'type_error')

        elif req_cmd == 'get_frontier':
            logger.info('command: get_frontier')
            if len(req_args) < 2:
                logger.error('get_frontier: not enough arguments (map_name px,py,pz,ox,oy,oz,ow')
                self.publish_response(req_cmd, req_id, 'error', 'argument_error')
                return
            map_name = req_args[0]
            pose = np.asarray(req_args[1].split(',')).astype(float)
            try:
                pose_list = await self.context.slam_servicer_client.get_frontier(map_name, pose)
                logger.info(f'frontier list: {pose_list}')
                pose_list_str = ' '.join(map(str, [','.join(map(str, np.around(pose, decimals=2))) for pose in pose_list]))
                self.publish_response(req_cmd, req_id, 'success', pose_list_str)
            except RuntimeError:
                logger.error('get_frontier: PlannerError')
                self.publish_response(req_cmd, req_id, 'error', 'planner_error')
            except SlamProcedureCallError:
                logger.error('get_frontier: SlamProcedureCallError')
                self.publish_response(req_cmd, req_id, 'error', 'rpc_error')

        elif req_cmd == 'get_latest_map':
            logger.debug('command: get_latest_map')
            try:
                map_name = await self.context.slam_servicer_client.get_latest_map()
                logger.debug(f'map_name: {map_name}')
                if map_name == '':
                    # return empty map name with success, when no maps exist on spike.
                    self.publish_response(req_cmd, req_id, 'success')
                else:
                    self.publish_response(req_cmd, req_id, 'success', map_name)
            except SlamProcedureCallError:
                logger.error('get_latest_map: SlamProcedureCallError')
                self.publish_response(req_cmd, req_id, 'error', 'rpc_error')
            except TypeError:
                logger.error('get_latest_map: type error')
                self.publish_response(req_cmd, req_id, 'error', 'type_error')

        elif req_cmd == 'check_explore_rights':
            logger.debug('command: check_explore_rights')
            try:
                success, token = await self.context.slam_servicer_client.check_explore_rights(self.ghost_id)
                if success:
                    logger.info(f'explore rights token issued: {token}')
                    self.publish_response(req_cmd, req_id, 'success', token)
                else:
                    self.publish_response(req_cmd, req_id, 'error', 'failed_to_get_token')
            except SlamProcedureCallError:
                logger.error('check_explore_rights: SlamProcedureCallError')
                self.publish_response(req_cmd, req_id, 'error', 'rpc_error')

        elif req_cmd == 'undeploy_map':
            logger.info('command: undeploy_map')
            try:
                # _control_ros_nodes coroutine stops localization process, when map_name is changed to None
                self.map_name = None
                await self.localization_process.wait_for_termination()
                await self._undeploy_map()
                self.publish_response(req_cmd, req_id, 'success')
            except SlamProcessError:
                logger.error('undeploy_map: SlamProcessError')
                self.publish_response(req_cmd, req_id, 'error', 'process_error')

        elif req_cmd == 'reload_slam_spots_from_nest':
            """
            Reload slam:spot:** from nest and write them to redis LTM.
            Delete the corresponding key from LTM, if it is not found in spike.
            """
            # logger.info('command: reload_slam_spots_from_nest')
            if len(req_args) < 1:
                logger.error('reload_slam_spots_from_nest: not enough arguments ("spot_name1,spot_name2,...")')
                self.publish_response(req_cmd, req_id, 'error', 'argument_error')
                return

            try:
                spot_names = req_args[0].split(',')
                spots_dict = await self.context.slam_servicer_client.get_spots(spot_names)
            except SlamProcedureCallError:
                logger.error('get_spots: SlamProcedureCallError')
                self.publish_response(req_cmd, req_id, 'error', 'rpc_error')
                return

            for name in spot_names:
                key = redis_keys.spot(name)
                if name in spots_dict:
                    self.redis_ltm.hset(key, mapping=spots_dict[name])
                elif self.redis_ltm.delete(key):
                    logger.info(f'Spot {key} was removed because it is not found in spike.')
            self.publish_response(req_cmd, req_id, 'success')

        elif req_cmd == 'reload_unwelcomed_area_from_nest':
            """
            Reload slam:unwelcomed_area from nest and write it to redis LTM.
            """
            # logger.info('command: reload_unwelcomed_area_from_nest')

            try:
                unwelcomed_area = await self.context.slam_servicer_client.get_unwelcomed_area()
                unwelcomed_area_hash = calc_unwelcomed_area_hash_from_str(unwelcomed_area)

                current_hash = self.redis_ltm.get(redis_keys.unwelcomed_area_hash)
                if current_hash != unwelcomed_area_hash:
                    self.redis_ltm.set(redis_keys.unwelcomed_area, unwelcomed_area)
                    self.redis_ltm.set(redis_keys.unwelcomed_area_hash, unwelcomed_area_hash)
                self.publish_response(req_cmd, req_id, 'success')
            except SlamProcedureCallError:
                logger.error('reload_unwelcomed_area_from_nest: SlamProcedureCallError')
                self.publish_response(req_cmd, req_id, 'error', 'rpc_error')

        elif req_cmd == 'reload_segmentation_from_nest':
            """
            Reload /data/localization/segmentation from nest
            """
            self.segmentation_downloader.force()
            # force 自体は常に成功するが、download自体は別スレッドで行われるのでログを参照。
            self.publish_response(req_cmd, req_id, 'success')

        else:
            logger.error('command not found. ' + req)

    def should_start_localization(self, lc_stopped):
        """Check if localization ros nodes should starting up now:

        1. map is ready
        2. rosmaster is alive, and
        3. localization ros nodes are not running, and
        """
        # rosgraph.is_master_online should be tested only if needed,
        # because it takes some time (2 ~ 5 msec when ros is down, 1 ~ 2 msec when ros is up)
        return all((self.map_name, lc_stopped)) and rosgraph.is_master_online()

    def should_stop_localization(self, lc_running):
        """Check if it is time to shutdown localization ros nodes:

        1. localization ros nodes are running, and
        2. map doesn't exist
        """
        return lc_running and not self.map_name

    def should_restart_localization(self, lc_running, map_updated):
        """Check if it is time to restart localization ros nodes to reload map:

        1. map exists, and
        2. localization ros nodes are running, and
        3. map has been updated and
        4. rosmaster is alive.
        """
        return all((self.map_name, lc_running, map_updated)) and rosgraph.is_master_online()

    def _dump_subprocess_log(self):
        if self.base_process.is_terminated():
            self.base_process.dump_stderr()
        if self.localization_process.is_terminated():
            self.localization_process.dump_stderr()

    async def _handle_ros_crash_and_raise(self, crashed_processes: List[str]) -> None:
        if not crashed_processes:
            self._dump_subprocess_log()
            raise RuntimeError("reached max retry count.")

        most_common = sorted(crashed_processes, key=crashed_processes.count)[-1]
        try:
            if most_common == "localizer":
                logger.warning("maplab localizer crashed, map data might be broken.")
                await self._undeploy_map()
            elif most_common == "map_server":
                logger.warning("map_server crashed, map data might be broken.")
                await self._undeploy_map()
            elif most_common == "accuracy_monitor":
                logger.warning("accuracy_monitor crashed, map data might be broken.")
                await self._undeploy_map()
            elif most_common == "omni_streamer":
                logger.warning("omni_streamer crashed, omni camera might be unstable.")
                # we raise sensor error here, because this is mostly like a hardware error.
                raise SlamSensorError
            elif most_common == "shm_to_depth":
                logger.warning("shm_to_depth crashed, tracking module might be unstable.")
                # we raise sensor error here, because this is mostly like a hardware error.
                raise SlamSensorError
            raise RuntimeError("reached max retry count. "
                               f"last crashed reason: {crashed_processes[-1]}")
        finally:
            logger.warning(f"Most common crashed process is {most_common}.")
            logger.warning(f"Crashed process history: {crashed_processes}")
            self._dump_subprocess_log()

    async def _push_localization_accuracy(self):
        ACCURACY_MAP_UPLOAD_PERIOD_SEC = 60 * 60
        last_time_accuracy_map_uploaded = 0

        async for _ in periodic(30):
            # upload accuracy map files
            try:
                accuracy_map_dir = self._monitor_root / 'accuracy_map'
                if (accuracy_map_dir.exists()
                        and trio.current_time() - last_time_accuracy_map_uploaded > ACCURACY_MAP_UPLOAD_PERIOD_SEC):
                    await self.context.slam_servicer_client.upload_accuracy_map(self.ghost_id)
                    last_time_accuracy_map_uploaded = trio.current_time()
            except RuntimeError as e:
                logger.warning(e)
            except SlamProcedureCallError:
                pass

    async def _control_ros_nodes(self):
        MAX_RETRY = 10
        ros_crash_count = 0
        crashed_processes = []

        def get_crashed_process():
            nonlocal crashed_processes

            pid = None
            if self.base_process.is_terminated():
                pid = self.base_process.pid
            elif self.localization_process.is_terminated():
                pid = self.localization_process.pid

            crashed_time, crashed_proc = self._ros_log.get_launch_shutdown_process(pid)
            if crashed_time and crashed_proc:
                crashed_dt_str = crashed_time.strftime('%Y-%m-%d %H:%M:%S')
                logger.warning(f'{crashed_proc} was crashed at [{crashed_dt_str}]')
                crashed_processes.append(crashed_proc)
                _ros_node_shutdown_metric.labels(node=crashed_proc).inc()
                return crashed_proc
            return None

        async def handle_ros_crash():
            nonlocal ros_crash_count
            get_crashed_process()
            ros_crash_count += 1
            if MAX_RETRY <= ros_crash_count:
                await self._handle_ros_crash_and_raise(crashed_processes)
                # NOTE: never reach here
            # clear process state
            if self.base_process.is_terminated():
                logger.warning("base ros nodes unexpectedly stopped")
                self.base_process.clear()
                # if localization process is also running, stop it.
                if self.localization_process.is_running():
                    await self.localization_process.stop_process_and_wait()
                # check whether rosmaster is still alive (this should not happen)
                await _raise_if_rosmaster_is_online()
            if self.localization_process.is_terminated():
                logger.warning("localization ros nodes unexpectedly stopped")
                self.localization_process.clear()
            # cool-off time before restarting
            await trio.sleep(self.RESTART_INTERVAL_AFTER_CRASH_SEC)

        async for _ in periodic(0.2):
            # check whether base or localization process is terminated (crashed)
            if self.base_process.is_terminated() or self.localization_process.is_terminated():
                await handle_ros_crash()

            # check base ros nodes should startup
            # NOTE: test with is_stopped, because handle_ros_crash should be executed before this
            if self.base_process.is_stopped():
                logger.info('(re)starting base ros nodes')
                await self.restart_base()
                await trio.sleep(2.0)

            # check localization ros nodes should startup or shutdown
            lc_stopped = self.localization_process.is_stopped()
            lc_running = self.localization_process.is_running()
            map_updated = self.map_name and (self.map_name != self.localization_process.map_name)
            # NOTE: test with is_stopped, because handle_ros_crash should be executed before this
            if self.should_start_localization(lc_stopped):
                logger.info('starting localization ros nodes')
                await self.start_localization(self.map_name)
            elif self.should_restart_localization(lc_running, map_updated):
                logger.info('restarting localization ros nodes')
                await self.stop_localization()
                # NOTE: start nodes whenever failed to stop it
                await self.start_localization(self.map_name)
            elif self.should_stop_localization(lc_running):
                logger.info('shutting down localization ros nodes')
                await self.localization_process.stop_process_and_wait()

    async def _setup_context(self, stack: AsyncExitStack) -> None:
        # client tries to connect to <spike_device_id>.local when host is None
        slam_servicer_client = await stack.enter_async_context(open_slam_servicer_client(host=SERVICER_HOST))
        wifi_client = await stack.enter_async_context(open_wifi_service_client())
        lovot_tf_client = await stack.enter_async_context(open_lovot_tf_client())
        localization_client = open_localization_client()

        context.set(Context(
            slam_servicer_client=slam_servicer_client,
            wifi_client=wifi_client,
            lovot_tf_client=lovot_tf_client,
            localization_client=localization_client,
            fingerprint_sync=self._fingerprint_sync,
            wifi_scan=self._wifi_scan,
            radio_map=RadioMap(),
        ))

    async def _run_main(self):
        if not await self._record_controller.initialize():
            logger.error('Failed to initialize sensosrs.')
            raise SlamSensorError

        async with trio.open_nursery() as nursery:
            # TODO
            # - wifi fingerprintingとそれ以外のコードが混在しているのでリファクタする
            #     - 元々のLovotSlamManager.__init__でのインスタンス化などを1レイヤー上に移動させるなど
            # - redisのasync化

            nursery.start_soon(self._record_controller.run)
            nursery.start_soon(self._push_localization_accuracy)
            nursery.start_soon(self._control_ros_nodes)
            nursery.start_soon(self._monitor_retry_push_bag)
            nursery.start_soon(self.segmentation_downloader.run)
            nursery.start_soon(self._reliability_monitor.run)
            nursery.start_soon(self._marker_localization_monitor.run)

            # WiFi fingerprinting
            nursery.start_soon(self._wifi_scan.run)
            nursery.start_soon(self._fingerprint_sync.run)
            nursery.start_soon(self._wifi_mapping.run)
            nursery.start_soon(self._wifi_evaluator.run)

            # gRPC server (only coro1 needs to execute this)
            if self._model < LovotModel.LV110:
                nursery.start_soon(serve_navigation_service)

            logger.info('lovot slam manager started')

    async def _stop(self):
        self.map_utils.delete_map_from_redis_stm()
        await self.localization_process.stop_process_and_wait()
        await self.base_process.stop_process_and_wait()
