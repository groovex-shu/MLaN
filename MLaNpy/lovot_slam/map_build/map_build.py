import json
import os
import time
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable, Optional

import anyio
import prometheus_client

from lovot_slam.env import (MAP_FEATUREMAP, MAP_STATISTICS_YAML, 
                            NestSlamState, 
                            SlamState, data_directories, redis_keys)
from lovot_slam.map_build.map_build_metrics import MapBuildAttemptResultsMetric
from lovot_slam.map_build.maplab_map_stats import MaplabMapStatsParser
from lovot_slam.map_build.request_queue import MergeMapsOption, RequestQueue
from lovot_slam.map_build.rosbag_info import is_camera_info_recorded
from lovot_slam.model import Model
from lovot_slam.redis.clients import create_stm_client
from lovot_slam.subprocess.subprocess import BuildMapSubprocess
from lovot_slam.utils.exceptions import SlamBuildMapError, SlamError
from lovot_slam.utils.file_util import sync_to_disk
from lovot_map.utils.map_utils import BagUtils, MapUtils, SpotUtils

_logger = getLogger(__name__)


_processing_duration_metric = prometheus_client.Summary(
    'localization_processing_duration', 'consumed time (sec) for each processing of map optimization',
    ['type', 'optimization_target'],
)
_map_statistics_scale_metric = prometheus_client.Summary(
    'localization_map_statistics_scale', 'map scale', ['target'],
)
_map_statistics_height_mean_metric = prometheus_client.Summary(
    'localization_map_statistics_height_mean', 'map height mean', ['target'],
)
_map_statistics_height_std_metric = prometheus_client.Summary(
    'localization_map_statistics_height_std', 'map height std', ['target'],
)
_map_merged_count_metric = prometheus_client.Gauge(
    'localization_map_merged_count', 'map merged count'
)
_vi_map_metric = prometheus_client.Gauge(
    'localization_vi_map_stats', 'vi_map related metrics', ['mission_id', 'type']
)

MAP_UPLOAD_RETRY_INTERVAL = 3600  # seconds


class _ProcessDurationCounter:
    """Counter to observe duration of each processing of building map"""

    def __init__(self, stat_metric, type_name):
        self._stat_metric = stat_metric
        self._type_name = type_name
        self._start_at = None

    def start(self):
        self._start_at = time.time()

    def end(self, is_single):
        """
        end the record of processing
        param: bool is_single: set True if the observation is about single mission optimization
        """
        if self._start_at:
            target = 'single' if is_single else 'multiple'
            duration = time.time() - self._start_at
            self._stat_metric.labels(
                type=self._type_name, optimization_target=target).observe(duration)
            self._start_at = None


class MapBuilder:
    _METRICS_MAP = {
        NestSlamState.BAG_CONVERSION.value:
            _ProcessDurationCounter(
                _processing_duration_metric, 'bag_conversion'),
        NestSlamState.BUILD_FEATURE_MAP.value:
            _ProcessDurationCounter(
                _processing_duration_metric, 'build_feature_map'),
        NestSlamState.SCALE_MAP.value:
            _ProcessDurationCounter(
                _processing_duration_metric, 'scale_feature_map'),
        NestSlamState.BUILD_DENSE_MAP.value:
            _ProcessDurationCounter(
                _processing_duration_metric, 'build_dense_map'),
    }

    def __init__(self, build_request_queue: RequestQueue, is_running_on_nest: bool, model: Model,
                 debug: bool, journal: bool) -> None:
        self._bag_utils = BagUtils(data_directories.bags)
        self._map_utils = MapUtils(data_directories.maps, data_directories.bags)
        self._spot_utils = SpotUtils(self._map_utils)

        self._redis_stm = create_stm_client()

        self._build_request_queue = build_request_queue
        self._is_running_on_nest = is_running_on_nest

        # running subprocess info
        self._build_map_process = BuildMapSubprocess(model,
                                                     output_to_console=debug,
                                                     journal=journal)

        # information regarding to building map
        self.processing_map_name = ''  # map name
        # bag name (the latest bag when merging maps)
        self.processing_bag_name = ''
        # bag names (all source bags when merging maps)
        self.processing_bag_names = []

        # map build state
        self._is_processing_event = False
        self._state_event: Optional[SlamState] = None
        self._change_state(NestSlamState.IDLE)

        self._map_build_metrics = MapBuildAttemptResultsMetric()

    def reset(self) -> None:
        self._map_build_metrics.reset()

    @property
    def metrics(self) -> MapBuildAttemptResultsMetric:
        return self._map_build_metrics

    @property
    def is_processing_event(self) -> bool:
        return self._is_processing_event

    def is_processing_map(self) -> bool:
        return self._is_processing_event

    def _is_processing_single_mission_map(self) -> bool:
        return self.processing_bag_name == self.processing_map_name

    def _change_state(self, state: SlamState) -> None:
        self._state_event = state
        self._redis_stm.set(redis_keys.state, str(state.value))
        self._redis_stm.set(redis_keys.is_busy, str(state != NestSlamState.IDLE))

    def _set_map_metrics(self, map_name: str, is_single: bool) -> None:
        target = 'single' if is_single else 'multiple'

        scale = self._map_utils.get_map_scale(map_name)
        _map_statistics_scale_metric.labels(target=target).observe(scale)

        height_mean, height_std = self._map_utils.get_map_statistics(map_name)
        _map_statistics_height_mean_metric.labels(target=target).observe(height_mean)
        _map_statistics_height_std_metric.labels(target=target).observe(height_std)

    def _record_processing_duration(self, old_state: NestSlamState, new_state: NestSlamState) -> None:
        # record the metrics of elapsed time for transition between states
        if new_state is not NestSlamState.BUILD_ERROR:
            end_metrics = self._METRICS_MAP.get(old_state.value)
            if end_metrics:
                end_metrics.end(self._is_processing_single_mission_map())
        start_metrics = self._METRICS_MAP.get(new_state.value)
        if start_metrics:
            start_metrics.start()

    def _change_state_and_record_metrics(self, state: NestSlamState) -> None:
        self._record_processing_duration(self._state_event, state)
        self._change_state(state)

    def remove_unused_resources(self) -> None:
        """Remove unused resources (maps and bags) from the system.
        Only on shaun (not running on nest),
        bags with corresponding build requests still in the queue are excluded.
        """
        exclusions = self._build_request_queue.get_map_names_in_requests() \
            if not self._is_running_on_nest else []
        self._map_utils.remove_unused_resources(exclusions=exclusions)

    def _initialize_build_single_mission_map(self, map_name: str) -> None:
        self._map_build_metrics.attempt()
        self.processing_map_name = map_name
        self.processing_bag_name = map_name
        self.processing_bag_names = [map_name]
        _logger.info('start building single mission map, '
                     f'bag:{self.processing_map_name}, map:{self.processing_bag_name}')
        self._map_utils.update_metadata(
            self.processing_map_name, source_list=[])

    async def _convert_bag(self) -> None:
        """
        Convert rosbag
        raise SlamBuildMapError if failed
        Mainly for coro1 to convert JPEG & imu data to ROS message
        """
        _logger.info('start bag conversion.')
        converted_bag = self._bag_utils.get_full_path(
            self.processing_bag_name + '_converted')
        if converted_bag.exists():
            _logger.debug(f'converted bag of {self.processing_bag_name} already exists. '
                          'skip converting bag.')
            await anyio.sleep(0)
            return

        try:
            original_bag = self._bag_utils.get_full_path(
                self.processing_bag_name)
            self._build_map_process.start_bag_conversion(
                original_bag, converted_bag)
            await self._build_map_process.wait_for_termination()
        except SlamError:
            _logger.error('failed to start bag conversion.')
            self._change_state_and_record_metrics(NestSlamState.BUILD_ERROR)
            raise SlamBuildMapError
        finally:
            with anyio.CancelScope(shield=True):
                await self._build_map_process.stop_process_and_wait()

        if not converted_bag.exists():
            _logger.error('failed to convert bag.')
            self._change_state_and_record_metrics(NestSlamState.BUILD_ERROR)
            raise SlamBuildMapError
        _logger.info('bag conversion finished.')

    async def _diminish_bag(self) -> None:
        _logger.info('start bag diminishing.')
        converted_bag = self._bag_utils.get_full_path(self.processing_bag_name + '_converted')
        diminished_bag = self._bag_utils.get_full_path(self.processing_bag_name + '_diminished')
        _map = self._map_utils.get_full_path(self.processing_map_name)
        # for root, dirs, files in os.walk(_map, topdown=False):
        #     for name in files:
        #         print(os.path.join(root, name))
        #     for name in dirs:
        #         print(os.path.join(root, name))
        _map = Path(_map) / 'feature_map' / 'vertices'
        missions_id = self._map_utils.get_mission_id_list_from_feature_map(self.processing_map_name)
        assert len(missions_id) == 1, "mission count should be 1"

        vertices_csv = _map / missions_id[0] / 'vertices.csv'
        _logger.info(f'vertices csv: {vertices_csv}')

        if diminished_bag.exists():
            _logger.debug(f'diminished bag of {self.processing_bag_name} already exists. '
                          'skip diminishing bag.')
            await anyio.sleep(0)
            return

        try:
            topics_str = '/depth/image_raw,/depth/camera_info'
            self._build_map_process.start_bag_diminish(original_bag=converted_bag, 
                                                       topics=topics_str, 
                                                       vertices_csv=vertices_csv, 
                                                       converted_bag=diminished_bag)
            await self._build_map_process.wait_for_termination()
        except SlamError as e:
            _logger.error(f'failed to start diminishing bag: "{e}"')
            self._change_state_and_record_metrics(NestSlamState.BUILD_ERROR)
            raise SlamBuildMapError
        finally:
            with anyio.CancelScope(shield=True):
                await self._build_map_process.stop_process_and_wait()

        if not diminished_bag.exists():
            _logger.error('failed to diminish bag.')
            self._change_state_and_record_metrics(NestSlamState.BUILD_ERROR)
            raise SlamBuildMapError

        # remove converted bag
        if converted_bag.exists():
            os.remove(converted_bag)
        _logger.info('diminishing bag finished.')

    async def _prune_bag(self) -> None:
        _logger.info('start bag pruning.')
        diminished_bag = self._bag_utils.get_full_path(self.processing_bag_name + '_diminished')
        # this will overwrite the original converted bag
        removal_bag = self._bag_utils.get_full_path(self.processing_bag_name + '_converted')
        # removal_bag = self._bag_utils.get_full_path(self.processing_bag_name + '_purned')

        if not diminished_bag.exists():
            _logger.error('diminished bag not found.')
            self._change_state_and_record_metrics(NestSlamState.BUILD_ERROR)
            raise SlamBuildMapError

        try:
            topics_to_keep = ['/depth/image_raw', '/depth/camera_info', '/tf', '/tf_static']
            self._build_map_process.start_bag_prune(diminished_bag, topics_to_keep, removal_bag)
            await self._build_map_process.wait_for_termination()
        except SlamError:
            _logger.error('failed to start pruning bag.')
            self._change_state_and_record_metrics(NestSlamState.BUILD_ERROR)
            raise SlamBuildMapError
        finally:
            with anyio.CancelScope(shield=True):
                await self._build_map_process.stop_process_and_wait()

        # remove diminish bag
        if diminished_bag.exists():
            os.remove(diminished_bag)
        _logger.info('pruning bag finished.')

    async def _build_single_mission_feature_map(self) -> None:
        """Build a single mission feature map
        raise SlamBuildMapError if failed
        """
        _logger.info('start building feature map.')
        converted_bag = self._bag_utils.get_full_path(
            self.processing_bag_name + '_converted')
        assert converted_bag.exists(), 'converted bag not found.'

        if self._map_utils.check_feature_map(self.processing_map_name):
            _logger.debug(f'feature map of {self.processing_map_name} already exists. '
                          'skip building feature map.')
            await anyio.sleep(0)
            return

        try:
            map_dir = self._map_utils.get_full_path(self.processing_map_name)
            camera_config_dir = data_directories.camera_config
            self._build_map_process.start_build_feature_map(
                converted_bag, map_dir, camera_config_dir)
            await self._build_map_process.wait_for_termination()
        except SlamError:
            _logger.error('failed to start building feature map.')
            raise SlamBuildMapError
        finally:
            with anyio.CancelScope(shield=True):
                await self._build_map_process.stop_process_and_wait()

        if not self._map_utils.check_feature_map(self.processing_map_name, verbose=True):
            _logger.error('failed to build single mission feature map.')
            raise SlamBuildMapError
        _logger.info('building feature map finished.')

    async def _scale_feature_map(self):
        """
        Scale a feature map (wheel odometry / visual odometry)
        Returns:
            nothing, if scaling file is existed
        
        Raise SlamBuildMapError if failed
        """
        _logger.info('start scaling feature map.')

        if self._map_utils.check_map_scale(self.processing_map_name):
            # lovot slam yaml already exists
            _logger.debug(f'scale of {self.processing_map_name} already obtained. '
                          'skip calculating scale.')
            return

        try:
            self._map_utils.update_metadata(self.processing_map_name, write_statistics=True)
            mission_ids = [self._map_utils.get_mission_id_list_from_feature_map(map_name)[0]
                           for map_name in self.processing_bag_names]
            map_name = self.processing_map_name
            source_maps = self.processing_bag_names
            self._build_map_process.start_scale_map(map_name, source_maps, mission_ids)
            await self._build_map_process.wait_for_termination()
        except SlamError:
            _logger.error('failed to start scaling feature map.')
            raise SlamBuildMapError
        else:
            _logger.info('scaling feature map subprocess finished. -> write to lovot_slam.yaml')
        finally:
            with anyio.CancelScope(shield=True):
                await self._build_map_process.stop_process_and_wait()

        # Check subprocess result.
        # Scale result will be written to lovot_slam.yaml
        #   scuccessed will be float values > 0.0
        #   failed will be -1.0
        lovot_slam_dict = self._map_utils.get_metadata(self.processing_map_name)
        if not all(v > 0.0 for v in lovot_slam_dict['scale'].values()):
            _logger.warning('scaling feature map is invalid.')
            
            raise SlamBuildMapError
        else:
            _logger.info('scaling feature map is valid.')

        if not self._map_utils.check_map_scale(self.processing_map_name, verbose=True):
            _logger.error('failed to scale feature map.')
            raise SlamBuildMapError
        if not self._map_utils.check_map_statistics(self.processing_map_name, verbose=True):
            _logger.error('built feature map statistics is invalid.')
            raise SlamBuildMapError
        _logger.info('scaling feature map finished.')

    async def _bulid_single_mission_dense_map(self) -> None:
        """Build a single mission dense map
        raise SlamBuildMapError if failed
        """
        _logger.info('start building dense map.')

        if self._map_utils.check_dense_map(self.processing_map_name):
            _logger.debug(f'dense of {self.processing_map_name} already existed. '
                          'skip building dense map.')
            await anyio.sleep(0)
            return

        try:
            maps_root = self._map_utils.root
            bags_root = self._bag_utils.root
            camera_config_dir = data_directories.camera_config
            map_name = self.processing_bag_name
            mission_id = self._map_utils.get_mission_id_list_from_feature_map(
                self.processing_bag_name)[0]
            publish_camera_info = not await is_camera_info_recorded(f'{bags_root}/{map_name}_converted.bag')
            self._build_map_process.start_build_dense_map(
                maps_root, bags_root, camera_config_dir, map_name, mission_id,
                publish_camera_info)
            await self._build_map_process.wait_for_termination()
        except SlamError:
            _logger.error('failed to start building dense map.')
            raise SlamBuildMapError
        finally:
            with anyio.CancelScope(shield=True):
                await self._build_map_process.stop_process_and_wait()
        
        dense_map_availability = self._map_utils.check_dense_map(self.processing_map_name, verbose=True)
        map_size_validity = self._map_utils.check_map_size(self.processing_map_name, verbose=True)
        if not dense_map_availability or not map_size_validity:
            _logger.error('failed to build dense map.')
            raise SlamBuildMapError
        _logger.info('dense map created.')

    async def _merge_feature_maps(self, option: MergeMapsOption) -> None:
        """Merge feature maps as specified with the option (input map and append maps)
        raise SlamBuildMapError if failed
        """
        source_list = self._map_utils.get_source_map_list(option.input_map)
        if not source_list:
            # if source list is empy, this is a single mission map
            # then, insert it to the source list.
            source_list = [option.input_map]
        source_list += option.maps_to_append
        self.processing_map_name = self._map_utils.create_map_name(source_list)
        self.processing_bag_name = source_list[-1]
        self.processing_bag_names = source_list

        _logger.info('start building merged map, '
                     f'bag:{self.processing_bag_name}, map:{self.processing_map_name}')
        self._map_utils.update_metadata(self.processing_map_name,
                                        source_list=self.processing_bag_names)
        try:
            _logger.debug(f'input_map: {option.input_map}')
            _logger.debug(f'maps_to_append: {option.maps_to_append}')
            self._build_map_process.start_merge_feature_maps(
                option.input_map, self.processing_map_name, option.maps_to_append)
            await self._build_map_process.wait_for_termination()
        except SlamError:
            _logger.error('failed to start merging feature maps.')
            raise SlamBuildMapError
        finally:
            with anyio.CancelScope(shield=True):
                await self._build_map_process.stop_process_and_wait()

        if not self._map_utils.check_feature_map(self.processing_map_name, verbose=True):
            _logger.error('failed to merge feature maps.')
            raise SlamBuildMapError

        # Update source list with actually merged single mission maps,
        # since some maps might not be merged due to lack of loop closures.
        source_list = self._map_utils.get_merged_single_mission_map_names(
            self.processing_map_name)
        self.processing_bag_name = source_list[-1]
        self.processing_bag_names = source_list
        self._map_utils.update_metadata(
            self.processing_map_name, source_list=source_list)
        _logger.info('merging feature maps finished.')

    async def _build_merged_dense_map(self) -> None:
        """Build a dense map of the merged map
        raise SlamBuildMapError if failed
        """
        _logger.info('start building dense map.')

        try:
            source_maps = self.processing_bag_names
            mission_ids = [self._map_utils.get_mission_id_list_from_feature_map(map_name)[0]
                           for map_name in self.processing_bag_names]
            self._build_map_process.start_merge_dense_maps(
                self.processing_map_name, source_maps, mission_ids)
            await self._build_map_process.wait_for_termination()
        except SlamError:
            _logger.error('failed to start building dense map.')
            raise SlamBuildMapError
        finally:
            with anyio.CancelScope(shield=True):
                await self._build_map_process.stop_process_and_wait()

        if not self._map_utils.check_dense_map(self.processing_map_name, verbose=True) \
                or not self._map_utils.check_map_size(self.processing_map_name, verbose=True):
            _logger.error('building dense map failed.')
            raise SlamBuildMapError
        _logger.info('dense map created.')

    def _save_mapname_missionid_pairs(self, map_name: str) -> None:
        pairs = self._map_utils.get_mapname_missionid_pairs(map_name)
        _logger.info(f'mapname_missionid_pairs: {pairs}')
        with open(data_directories.maps / map_name / 'mission_ids.json', 'w') as f:
            f.write(json.dumps(pairs))

    async def _finalize_build_single_mission_map(self) -> None:
        # remove the original rosbag
        self._bag_utils.remove(self.processing_bag_name,
                               original=True, converted=False)

        self._map_utils.remove_bulding_flag(self.processing_map_name)
        if not self._map_utils.check_map(self.processing_map_name, ignore_md5sum=True):
            _logger.error('building single mission map failed.')
            raise SlamBuildMapError

        # record md5sum of the map contents to a file.
        self._map_utils.create_md5sum_list(self.processing_map_name)
        sync_to_disk()

        _logger.info(f'map list: {self._map_utils.get_map_list(use_cache=False)}')
        self._set_map_metrics(self.processing_map_name,
                              self._is_processing_single_mission_map())

    def _finalize_build_merged_map(self) -> None:
        self._map_utils.remove_bulding_flag(self.processing_map_name)
        if not self._map_utils.check_map(self.processing_map_name, ignore_md5sum=True):
            _logger.error('merging map failed.')
            raise SlamBuildMapError

        _logger.info('finalizing build of merged map.')
        # transform entrance pose, register to redis and upload to cloud
        self._spot_utils.transform_spot_and_register('entrance')
        # update unwelcomed area
        self._spot_utils.transform_unwelcomed_area_and_register()
        # merged number metrics
        _map_merged_count_metric.set(
            self._map_utils.get_maps_number_in_latest_merged_map())

        # save map_name: mission_id pairs to a file
        self._save_mapname_missionid_pairs(self.processing_map_name)

        # finally, record md5sum of the map contents to a file.
        self._map_utils.create_md5sum_list(self.processing_map_name)
        sync_to_disk()

        # set prometheus metrics
        _logger.info(f'map list: {self._map_utils.get_map_list(use_cache=False)}')
        self._set_map_metrics(self.processing_map_name,
                              self._is_processing_single_mission_map())
        self._set_vi_map_metrics()

    def _set_vi_map_metrics(self) -> None:
        """Set vi_map metrics with the following labels:
        - mission_id: <mission id> or "accumulated"
        - type
            - landmark_count: landmark count
        """
        # landmark count metrics
        map_path = self._map_utils.get_full_path(self.processing_map_name)
        stats_file = map_path / MAP_FEATUREMAP / MAP_STATISTICS_YAML
        parser = MaplabMapStatsParser(stats_file)

        # Set accumulated landmark count with "accumulated" mission_id
        accumulated_metrics = parser.accumulated_metrics
        if accumulated_metrics:
            _vi_map_metric.labels(mission_id='accumulated', type='landmark_count').set(
                accumulated_metrics.landmark_count)
            _logger.debug(f"Updated accumulated landmark count metric: {accumulated_metrics.landmark_count}")

        # Set per-mission landmark counts
        mission_metrics = parser.mission_metrics
        for mission_id, metrics in mission_metrics.items():
            _vi_map_metric.labels(mission_id=mission_id, type='landmark_count').set(metrics.landmark_count)
            _logger.debug(f"Updated landmark count metric for mission {mission_id}: {metrics.landmark_count}")

    async def build_single_mission_map(self, map_name: str) -> Optional[MergeMapsOption]:
        """Build a single mission map.
        :param str map_name: map name e.g. '20230725_174000'
        :return: MergeMapsOption if the map is successfully built, otherwise None
        """
        async def build_map_func(func: Callable, state: NestSlamState):
            self._change_state_and_record_metrics(state)
            await func()
            sync_to_disk()


        try:
            self._is_processing_event = True
            self._initialize_build_single_mission_map(map_name)
            await build_map_func(self._convert_bag, NestSlamState.BAG_CONVERSION)
            # single mission map
            await build_map_func(self._build_single_mission_feature_map, NestSlamState.BUILD_FEATURE_MAP)
            await build_map_func(self._diminish_bag, NestSlamState.BAG_PROCESSING)
            await build_map_func(self._prune_bag, NestSlamState.BAG_PROCESSING)
            await build_map_func(self._scale_feature_map, NestSlamState.SCALE_MAP)
            await build_map_func(self._bulid_single_mission_dense_map, NestSlamState.BUILD_DENSE_MAP)
            await self._finalize_build_single_mission_map()

            # prepare merging feature maps
            latest_merged_map = self._map_utils.get_latest_merged_map()
            if latest_merged_map:
                merge_option = MergeMapsOption(latest_merged_map,
                                               [self.processing_map_name])
            else:
                merge_option = MergeMapsOption(self.processing_map_name)
        except SlamBuildMapError:
            _logger.error(
                'building single mission map finished without map completed.')
            self._map_build_metrics.fail(is_single=True, status=self._state_event)
            self.remove_unused_resources()
            return None
        else:
            return merge_option
        finally:
            self._is_processing_event = False
            self._change_state_and_record_metrics(NestSlamState.IDLE)

    async def build_merged_map(self, option: MergeMapsOption) -> bool:
        """Build merged map (merge feature maps and build dense map)
        :param MergeMapsOption option: option to merge feature maps
        :return: True if the map is successfully built, otherwise False
        """
        async def build_map_func(func: Callable, state: NestSlamState):
            self._change_state_and_record_metrics(state)
            await func()
            sync_to_disk()

        result = True

        try:
            self._is_processing_event = True
            # merged map
            await build_map_func(partial(self._merge_feature_maps, option),
                                 NestSlamState.BUILD_FEATURE_MAP)
            await build_map_func(self._scale_feature_map, NestSlamState.SCALE_MAP)
            await build_map_func(self._build_merged_dense_map, NestSlamState.BUILD_DENSE_MAP)
            self._finalize_build_merged_map()
            self._map_build_metrics.success()
        except SlamBuildMapError:
            _logger.error(
                'building merged map finished without map completed.')
            self._map_build_metrics.fail(
                is_single=False, status=self._state_event)
            result = False
        finally:
            self.remove_unused_resources()
            self._is_processing_event = False
            self._change_state_and_record_metrics(NestSlamState.IDLE)

        return result

