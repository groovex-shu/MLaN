import json
from logging import getLogger
from typing import Dict

import prometheus_client

from lovot_slam.env import NestSlamState
from lovot_slam.redis.clients import create_ltm_client

_logger = getLogger(__name__)


class MapBuildAttemptResultsMetric:
    """Counter metrics regarding the map build attempt, success and fail.
    Only the continuous-failure-count is persistented to LTM,
    while other metrics just use prometheus_client and are not persistented to each machine.

    The continous failure counts increments on the map build failure,
    and are cleared on
    - the map reset
    - the map build success

    The continuous failure counts are jsonized as below and saved to LTM.
    {
        "single_bag_conversion": 0,
        "single_build_feature_map": 0,
        ...
    }

    NOTE: `status` was renamed to `state` in NestSlamState, but we keep using `status` here for backward compatibility.
    """

    _LTM_CONTINUOUS_FAIL_COUNTS_KEY = 'slam:map_build:continuous_fail_counts'

    def __init__(self) -> None:
        # self._attempt_metric = prometheus_client.Counter(
        #     'localization_map_build_attempt_count',
        #     'Attempt count of map build')
        # self._success_metric = prometheus_client.Counter(
        #     'localization_map_build_success_count',
        #     'Success count of map build')
        # self._failure_metric = prometheus_client.Counter(
        #     'localization_map_build_failure_count',
        #     'Failure count of map build',
        #     labelnames=['target', 'status'])

        self._ltm_client = create_ltm_client()

    def _get_continuous_fail_counts(self) -> Dict[str, int]:
        continuous_fail_counts = {}

        json_str = self._ltm_client.get(self._LTM_CONTINUOUS_FAIL_COUNTS_KEY)
        try:
            continuous_fail_counts = json.loads(json_str)
        except TypeError as e:
            # almost only the case that json_str is None
            _logger.debug(f'failed to decode json string {json_str}: {e}')
        except json.decoder.JSONDecodeError as e:
            _logger.warning(f'failed to decode json string {json_str}: {e}')
        return continuous_fail_counts

    def _save_continuous_fail_counts(self, continuous_fail_counts: Dict[str, int]) -> None:
        json_str = json.dumps(continuous_fail_counts)
        self._ltm_client.set(self._LTM_CONTINUOUS_FAIL_COUNTS_KEY, json_str)

    def _clear_continuous_fail_counts(self) -> None:
        if self._ltm_client.delete(self._LTM_CONTINUOUS_FAIL_COUNTS_KEY):
            _logger.info('Reset the map build continuous fail counts.')

    @staticmethod
    def _create_dict_key(is_single: bool, status: NestSlamState) -> str:
        """Key of the dictionary which stores fails count per each build step.
            "single_bag_conversion": 0,
            "single_build_feature_map": 0,
            ...
        """
        target = 'single' if is_single else 'multiple'
        return f'{target}_{status}'

    def reset(self) -> None:
        self._clear_continuous_fail_counts()

    def attempt(self) -> None:
        # self._attempt_metric.inc()
        pass

    def success(self) -> None:
        # self._success_metric.inc()
        self._clear_continuous_fail_counts()

    def fail(self, is_single: bool, status: NestSlamState):
        target = 'single' if is_single else 'multiple'
        # self._failure_metric.labels(target=target, status=str(status)).inc()

        # Increment the persistent counter for each build step.
        key = f'{target}_{status}'
        continuous_fail_counts = self._get_continuous_fail_counts()
        if key not in continuous_fail_counts:
            continuous_fail_counts[key] = 0
        continuous_fail_counts[key] += 1
        self._save_continuous_fail_counts(continuous_fail_counts)

    def get_continuous_fail_count(self, is_single: bool, status: NestSlamState) -> float:
        """Get the continuous fail count for each build step.
        """
        return self._get_continuous_fail_counts().get(self._create_dict_key(is_single, status), 0)

    def get_total_continuous_fail_count(self) -> float:
        """Get sum of every continuous fail counts.
        """
        return sum(self._get_continuous_fail_counts().values())
