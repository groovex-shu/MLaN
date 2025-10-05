import datetime
import math
from logging import getLogger
from typing import Optional

import numpy as np
import prometheus_client
import purerpc
import trio

from lovot_apis.lovot_tf.tf.tf_pb2 import Header, Quaternion, Transform, TransformStamped, Vector3

from lovot_slam import ContextMixin
from lovot_slam.utils.protobuf_util import unix_time_to_pb_timestamp
from lovot_slam.wifi.mapping.mapping import RadioMap

_logger = getLogger(__name__)

_wifi_fingerrpinting_positioning_error_metric = prometheus_client.Histogram(
    'localization_wifi_fingerrpinting_positioning_error',
    'localization error by WiFi fingerprinting',
    buckets=(
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        15.0, 20.0,
        math.inf
    )
)


def _unixtime_to_human_readable(unixtime: float) -> str:
    dt = datetime.datetime.fromtimestamp(unixtime)
    return dt.strftime("%Y/%m/%d %H:%M:%S")


class InferenceEvaluator(ContextMixin):
    """Evaluate the inference result.
    The map is copied periodically from the current radio map for the evaluation,
    and applied with a delay.
    This is to avoid inference with the training data.

    We use the transform estimation with good confidence and reliability as the ground truth.
    """

    _TARGET_MAP_UPDATE_PERIOD = 60 * 60
    _TARGET_MAP_APPLY_DELAY = 60
    _MINIMUM_FINGERPRINT_SIZE = 100

    # this is only for evaluation
    _PARENT_FRAME_ID = 'wifi_map_ev'
    _FRAME_ID = 'wifi_base_ev'

    def __init__(self) -> None:
        self._target_map: Optional[RadioMap] = None

        # history of [x, y, error]
        # TODO: persistent to db
        self._error_history = np.empty((0, 3))

    def average_error(self) -> Optional[float]:
        if len(self._error_history) == 0:
            return None
        return np.mean(self._error_history[:, 2])

    async def _update_target_map(self, *, task_status=trio.TASK_STATUS_IGNORED) -> None:
        """Periodically update the target radio map for evaluation.
        There is a delay to apply the target map.
        """
        first = True
        while True:
            if len(self.context.radio_map.fingerprints) < self._MINIMUM_FINGERPRINT_SIZE:
                await trio.sleep(10)
                continue
            # Copy current radio map as the target for evaluation.
            target_map = self.context.radio_map.copy()
            if first:
                first = False
                task_status.started()
            else:
                # Delay not to apply the target map immediately,
                # to avoid query similar with the training data.
                await trio.sleep(self._TARGET_MAP_APPLY_DELAY)
            if not self._target_map or self._target_map != target_map:
                self._target_map = target_map
                stamp = self._target_map.latest_stamp()
                stamp = _unixtime_to_human_readable(stamp) if stamp else stamp
                _logger.info(f'Evaluation target map is updated: '
                             f'{len(self._target_map.fingerprints)} fingerprints '
                             f'with {len(self._target_map._ssids)} SSIDs '
                             f'and stamp of {stamp}')
            await trio.sleep(self._TARGET_MAP_UPDATE_PERIOD)

    async def _set_transform(self, stamp: float, position: np.ndarray) -> None:
        transform = TransformStamped(
            header=Header(seq=0, stamp=unix_time_to_pb_timestamp(stamp), frame_id=self._PARENT_FRAME_ID),
            child_frame_id=self._FRAME_ID,
            transform=Transform(
                translation=Vector3(x=position[0], y=position[1], z=0),
                rotation=Quaternion(x=0, y=0, z=0, w=1)
            )
        )
        try:
            await self.context.lovot_tf_client.set_transform(transform)
        except purerpc.GRPCError as e:
            _logger.debug(f'Failed to set transform: {e}')

    async def _evaluate(self) -> None:
        """Periodically evaluate the inference result with the copied target radio map.
        The evaluation is done by comparing the predicted position with the transform estimation
        with good confidence and reliability.

        TODO
        - ground truth付近のfingerprintsが疎な時には許してあげたい
        """
        fingerprint_sync = self.context.fingerprint_sync

        async for fingerprint, _ in fingerprint_sync.fingerprint_event.transitions():
            if not self._target_map:
                continue

            position = self._target_map.predict(fingerprint.access_points, 4)
            if position is None:
                continue

            await self._set_transform(fingerprint.stamp, position)

            if fingerprint.transform.is_good():
                trans = fingerprint.transform.transform.translation
                error = position - np.array([trans.x, trans.y])
                error_meter = np.linalg.norm(error)
                _logger.debug(f'Predicted position: [{position[0]:.2f}, {position[1]:.2f}], '
                              f'estimation error: {error_meter:.2f} m')
                _wifi_fingerrpinting_positioning_error_metric.observe(error_meter)

                self._error_history = \
                    np.append(self._error_history, [[position[0], position[1], error_meter]], axis=0)

    async def run(self) -> None:
        async with trio.open_nursery() as nursery:
            await nursery.start(self._update_target_map)
            nursery.start_soon(self._evaluate)
