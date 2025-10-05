from logging import getLogger
from typing import Optional

import purerpc
from trio_util import AsyncValue

from lovot_slam import ContextMixin
from lovot_slam.wifi.type import Fingerprint, TransformEstimation

_logger = getLogger(__name__)



class FingerprintSync(ContextMixin):
    """
    Periodically poll localization status (confidence, reliability, etc.) and store them.
    Return the latest status with transform when requested with timestamp.
    現状の実装は上記通りになっていない
    TODO
    - 定期的にpollして以下のpropertyを取得し、stampとともにbufferに保存する
        - map_name
        - confidence
        - reliability
        - localizer
        - transform
    - できれば
        - tfも一緒に保存して、lookup時にtfと一緒に比較する
    """

    def __init__(self) -> None:
        self._fingerprint_event = AsyncValue(None)

    @property
    def fingerprint_event(self) -> AsyncValue[Optional[Fingerprint]]:
        return self._fingerprint_event

    async def _get_transform_estimation(self, stamp: float) -> Optional[TransformEstimation]:
        lc_client = self.context.localization_client
        tf_client = self.context.lovot_tf_client

        try:
            transform = await tf_client.get_transform('omni_map', 'base', stamp=stamp)
        except purerpc.grpclib.exceptions.FailedPreconditionError as e:
            # this is expected when the transform is not available
            _logger.debug(f'Failed to get transform: {e}')
            return None
        except purerpc.GRPCError as e:
            _logger.warning(f'Failed to get transform: {e}')
            return None

        # TODO: timestampを比較
        # covariance, reliabilityはもう少し高頻度に確認しておいて、bufferingする？
        map_name = lc_client.get_map_name()
        localizer = lc_client.get_localizer()
        covariance = lc_client.get_covariance()
        reliability = lc_client.get_failure_detection_results()

        return TransformEstimation(
            stamp=stamp,
            transform=transform.transform,
            map_id='1',
            map_name=map_name,
            localizer=localizer,
            covariance=covariance,
            reliability=reliability,
        )

    async def run(self):
        wifi_scan = self.context.wifi_scan

        # TODO: read device_id from redis
        device_id = ''

        async for access_points, _ in wifi_scan.access_points_event.transitions():
            # Get transform at the last_seen time
            tf_estimation = await self._get_transform_estimation(access_points.stamp)
            if not tf_estimation:
                continue

            _logger.debug(f'Current transform: '
                          f'[{tf_estimation.transform.translation.x:.2f}, '
                          f'{tf_estimation.transform.translation.y:.2f}]')

            self._fingerprint_event.value = Fingerprint(
                access_points.stamp, device_id, access_points, tf_estimation)
