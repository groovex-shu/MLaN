from logging import getLogger
from math import inf
from typing import Dict, NamedTuple, Optional

import prometheus_client
from trio_util import periodic

from lovot_slam.redis import create_stm_client

logger = getLogger(__name__)

_DETECTION_RESULT_KEY = "slam:failure_detection:result"
_METRIC_LABELS = ('reliability', 'detection', 'likelihood')

_reliability_metric = prometheus_client.Histogram(
    'localization_reliability', 'localization reliability',
    buckets=(
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, inf
    ),
    labelnames=["prob_type"],
)


class LocalizationReliability(NamedTuple):
    timestamp: float
    reliability: float
    detection: float
    likelihood: float

    @classmethod
    def from_hashed_values(cls, values: Dict[str, str]) -> Optional['LocalizationReliability']:
        try:
            return cls(
                timestamp=float(values['timestamp']),
                reliability=float(values['reliability']),
                detection=float(values['detection']),
                likelihood=float(values['likelihood']),
            )
        except (KeyError, ValueError):
            return


class LocalizationReliabilityMonitor:
    """Monitor localization reliability from localization failure detection.
    Periodically check the result of localization failure detection and
    push the result to prometheus.
    The actual detection is performed by 'failure_detection_node' ROS node,
    and the result is stored in the redis as a hash table.
    """
    MONITOR_INTERVAL_SEC = 10

    def __init__(self) -> None:
        self._stm_client = create_stm_client()

        self._reliability: Optional[LocalizationReliability] = None

    async def run(self):
        async for _ in periodic(self.MONITOR_INTERVAL_SEC):
            res = self._stm_client.hgetall(_DETECTION_RESULT_KEY)
            reliability = LocalizationReliability.from_hashed_values(res)
            if reliability is None \
                or (self._reliability
                    and reliability.timestamp == self._reliability.timestamp):
                continue
            self._reliability = reliability

            for key in _METRIC_LABELS:
                _reliability_metric.labels(prob_type=key).observe(getattr(reliability, key))
