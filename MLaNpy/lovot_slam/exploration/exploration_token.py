import time
from logging import getLogger
from typing import Callable, Optional, Tuple

import redis

from lovot_slam.env import data_directories
from lovot_slam.exploration.exploration_status import ExplorationStatusMonitor
from lovot_slam.flags.cloudconfig import CLOUDCONFIG_DISABLE_BUILD_MAP
from lovot_slam.map_build.map_build_metrics import MapBuildAttemptResultsMetric
from lovot_slam.map_build.request_queue import RequestQueue
from lovot_slam.utils.file_util import get_directory_size
from lovot_slam.utils.map_utils import MAX_DATA_DIR_SIZE, MapUtils

_logger = getLogger(__name__)

# it reaches to max ttl (26 hours) in approx 5~6 days (128 hours)
MAX_DAILY_TOKEN = 8


class ExplorationTokenManager:
    """Exploration token manager.
    Token is inquired by localization-tom via SlamServicer.

    Issue token when all the conditions met:
    - extent time is passed from the last token issue
    - not bulding maps
    - exploration status is can_explore
    """

    _LTM_TOKEN_TTL_KEY = 'slam:exploration_token_ttl'
    _MIN_BUILD_FAIL_COUNT_TO_EXTEND_TTL = 12  # must be over a day
    _FALLBACK_TOKEN_TTL = 3600
    _TTL_EXTENSION_RATE = 1.2
    _MAX_TOKEN_TTL = 26 * 3600  # 26 hours (intendedly shifted from exactly a day)

    def __init__(self,
                 ltm_client: redis.StrictRedis,
                 exploration_status_monitor: ExplorationStatusMonitor,
                 map_build_metrics: MapBuildAttemptResultsMetric,
                 is_building_map_func: Callable[[], bool],
                 is_ttl_from_waketime: bool,
                 request_queue: RequestQueue,
                 map_util: MapUtils) -> None:
        self._ltm_client = ltm_client
        self._exploration_status_monitor = exploration_status_monitor
        self._map_build_metrics = map_build_metrics
        self._is_building_map = is_building_map_func

        self._request_queue = request_queue
        self._map_util = map_util

        # Cloud config setting
        if CLOUDCONFIG_DISABLE_BUILD_MAP:
            _logger.warning('Build map is disabled')

        # NOTE: the token timestamp is not persistented to such as LTM,
        # so the token would be always issued after the Nest is restarted.
        self._token_timestamp = 0.
        if is_ttl_from_waketime:
            # coro2 only
            self._default_token_ttl = self._calculate_daily_ttl()
        else:
            self._default_token_ttl = self._FALLBACK_TOKEN_TTL
        _logger.info(f'Default exploration token TTL is set to {self._default_token_ttl / 60:.0f} min')


    def _get_sleep_time(self) -> Optional[Tuple[int, int]]:
        """
        Get sleep time from LTM.
        :return: (sleep_time_end, sleep_time_start) both are in "seconds" from 0:00
        """
        def convert_to_seconds(time: str) -> int:
            if time.count(':') != 1:
                raise ValueError(f'Invalid time format: {time}')
            h, m = map(int, time.split(':'))
            return h * 60 * 60 + m * 60

        end, start = self._ltm_client.mget('colony:sleep_time:end', 'colony:sleep_time:start')
        if end and start:
            return convert_to_seconds(end), convert_to_seconds(start)
        return None

    def _calculate_daily_ttl(self) -> float:
        """
        Set the token ttl to the daily schedule of Coro.
        """
        sleep_time = self._get_sleep_time()
        if not sleep_time:
            _logger.info('Sleep time is not set. Use fallback token TTL.')
            return self._FALLBACK_TOKEN_TTL

        wake_time, bed_time = sleep_time
        if wake_time > bed_time:
            bed_time += 24 * 60 * 60
        interval = (bed_time - wake_time) / (MAX_DAILY_TOKEN - 1)
        interval = max(interval, self._FALLBACK_TOKEN_TTL)
        return interval

    def _get_token_ttl(self) -> float:
        token_ttl = self._ltm_client.get(self._LTM_TOKEN_TTL_KEY)
        return float(token_ttl) if token_ttl else self._default_token_ttl

    def _extend_token_ttl(self) -> None:
        """Extend exploration token ttl.
        Call this only when issueing new token.
        """
        new_token_ttl = min(self._get_token_ttl() * self._TTL_EXTENSION_RATE,
                            self._MAX_TOKEN_TTL)
        self._ltm_client.set(self._LTM_TOKEN_TTL_KEY, new_token_ttl)
        _logger.info(f'Exploration token TTL is extended: {new_token_ttl / 60:.0f} min')

    def _reset_token_ttl(self) -> None:
        # delete the key when we want to use the default value
        if self._ltm_client.delete(self._LTM_TOKEN_TTL_KEY):
            _logger.info('Reset exploration token TTL.')

    def _check_data_dir_size(self) -> bool:
        """Check the size of the data directory which comprises the maps and the bags.
        If the size exceeds the threshold, return False.

        NOTE: This is a temporary measure to prevent the disk from being full.
        This may cause inconsistency in ExplorationStatus.
        For instance, the status may be set to 'can_explore' even if the disk is full.
        So, we should move this check to ExplorationStatusMonitor.
        """
        dir_size = get_directory_size(data_directories.data_root)
        _logger.debug(f'Data directory size: {dir_size / 1024 / 1024:.2f} MB')
        return dir_size < MAX_DATA_DIR_SIZE

    def inquire_token(self) -> Tuple[bool, str]:
        """
        排他制御。既にトークンが発行されていないか。
        空いていたらトークンを生成して、トークンに対するタイマーを生成する。
        また、地図生成中はトークンを発行しない。
        また、exploration statusがcan not exploreなら発行しない。
        トークンを発行後、一定時間は発行しなくなるだけ。
        
        Func of issuing tokens
        Args:
            None
        Returns:
            success: bool
            token: str
        """
        now = time.time()
        success = True
        token = ''

        # Extend TTL
        continuous_build_fail_count = self._map_build_metrics.get_total_continuous_fail_count()
        should_extend_ttl = self._MIN_BUILD_FAIL_COUNT_TO_EXTEND_TTL <= continuous_build_fail_count
        if not should_extend_ttl:
            self._reset_token_ttl()

        # Cloud config setting(No TTL exisits)
        if CLOUDCONFIG_DISABLE_BUILD_MAP:
            return False, ''
        
        map_exist = len(self._map_util.get_latest_merged_map()) > 0

        # Day 0(init/reset) case
        # _request_queue is increased "After" an explore behavior is done
        # so, if map exists and _request_queue >= 1 means it try to create more than 1 map
        if not map_exist and len(self._request_queue) >= 1:
            success = False
            _logger.debug('It is day_0 so no create more than 1 map')

        if now - self._token_timestamp <= self._get_token_ttl():
            success = False
            _logger.debug('ExploreRights token NOT issued: existing token still alive')

        if self._is_building_map():
            success = False
            _logger.debug('ExploreRights token NOT issued: currently building a map')

        if not self._exploration_status_monitor.status.can_explore():
            success = False
            _logger.debug('ExploreRights token NOT issued: exploration status does not meet conditions')

        if not self._check_data_dir_size():
            success = False
            # NOTE: this is INFO because there is no other way to know the size exceeds the threshold
            _logger.info('ExploreRights token NOT issued: data directory size is over the threshold')

        if success:
            self._token_timestamp = now
            token = str(now)
            _logger.info(f'ExploreRights token issued: {token}')
            if should_extend_ttl:
                self._extend_token_ttl()

        return success, token
