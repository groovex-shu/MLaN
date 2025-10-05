import time
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Dict, Mapping, Tuple

import attr
import numpy as np
from attr import define, field

from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import Covariance as Covariance_pb
from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import Fingerprint as Fingerprint_pb
from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import Localizer as Localizer_pb
from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import Reliability as Reliability_pb
from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import TransformEstimation as TransformEstimation_pb
from lovot_apis.lovot_minid.wifi.wifi_pb2 import AP, GetAvailableAPResponse
from lovot_apis.lovot_tf.tf.tf_pb2 import Transform

from lovot_slam.utils.immutable_mapping import ImmutableMapping
from lovot_slam.utils.protobuf_util import unix_time_to_pb_timestamp


@define(frozen=True)
class Covariance:
    timestamp: float
    # 3x3 covariance matrix [x, y, yaw]
    matrix: np.ndarray = field(eq=attr.cmp_using(eq=np.array_equal), factory=partial(np.zeros, (3, 3)))

    def sigma_ellipse(self) -> Tuple[np.ndarray, float]:
        """Obtain ellipse shape from covariance matrix.
        return lengths of major and minor axis (magnitude of sigma),
        and angle of the principal axis.
        accepts only a symmetric matrix (if not, returns complex values)
        """
        eigen_value, eigen_vector = np.linalg.eigh(self.matrix[:2, :2])
        # np.linalg.eigh returns eigenvalues in ascending order, so
        # the principal axis is eigen_vecotr[:, 1]
        angle = np.arctan2(eigen_vector[1, 1], eigen_vector[0, 1])
        # if the eigenvalues are almost 0,
        # the obtained values can be minus (e.g. -2.49920001e-15),
        # due to the numerical calculation error.
        # so clip them to positive and realistic range (sigma = 10.0 m, variance = 10.0**2)
        variances = np.clip(eigen_value, 0, 100.0)
        return np.sqrt(variances), angle

    @staticmethod
    def from_csv_string(string: str, timestamp: float) -> 'Covariance':
        covariance = list(map(float, string.split(',')))
        if len(covariance) == 36:
            covariance = np.array(covariance).reshape((6, 6))
            covariance = np.delete(covariance, [2, 3, 4], axis=0)
            covariance = np.delete(covariance, [2, 3, 4], axis=1)
        elif len(covariance) == 9:
            covariance = np.array(covariance).reshape((3, 3))
        else:
            raise ValueError(f'invalid covariance: {covariance}')
        return Covariance(timestamp, covariance)

    def as_proto(self) -> Covariance_pb:
        message = Covariance_pb(stamp=unix_time_to_pb_timestamp(self.timestamp))
        message.matrix.extend(self.matrix.flatten())
        return message

    @staticmethod
    def from_proto(message: Covariance_pb) -> 'Covariance':
        return Covariance(
            timestamp=message.stamp.seconds+message.stamp.nanos*1e-9,
            matrix=np.array(message.matrix).reshape((3, 3)),
        )


@define(frozen=True)
class Reliablity:
    # NOTE: protobufそのまま使っても良いかも
    timestamp: float
    reliability: float
    detection: float
    likelihood: float

    def __attrs_post_init__(self):
        epsilon = 0.01
        if self.reliability < -epsilon or 1.0 + epsilon < self.reliability:
            raise ValueError(f'invalid reliability: {self.reliability}')
        if self.detection < -epsilon or 1.0 + epsilon < self.detection:
            raise ValueError(f'invalid detection: {self.detection}')
        if self.likelihood < -epsilon or 1.0 + epsilon < self.likelihood:
            raise ValueError(f'invalid likelihood: {self.likelihood}')

    def as_proto(self) -> Reliability_pb:
        return Reliability_pb(
            stamp=unix_time_to_pb_timestamp(self.timestamp),
            reliability=self.reliability,
            detection=self.detection,
            likelihood=self.likelihood,
        )

    @staticmethod
    def from_proto(message: Reliability_pb) -> 'Reliablity':
        return Reliablity(
            timestamp=message.stamp.seconds+message.stamp.nanos*1e-9,
            reliability=message.reliability,
            detection=message.detection,
            likelihood=message.likelihood,
        )


class Localizer(Enum):
    VISUAL = 'visual'
    DEPTH = 'depth'

    @staticmethod
    def value_of(target_value) -> 'Localizer':
        for e in Localizer:
            if e.value == target_value:
                return e
        raise ValueError

    def as_proto(self) -> Localizer_pb:
        if self == Localizer.VISUAL:
            return Localizer_pb.VISUAL
        elif self == Localizer.DEPTH:
            return Localizer_pb.DEPTH

    @staticmethod
    def from_proto(message: Localizer_pb) -> 'Localizer':
        if message == Localizer_pb.VISUAL:
            return Localizer.VISUAL
        elif message == Localizer_pb.DEPTH:
            return Localizer.DEPTH
        else:
            raise ValueError


@define(frozen=True)
class TransformEstimation:
    stamp: float = None
    transform: Transform = None
    map_id: str = None
    map_name: str = None
    localizer: Localizer = None
    covariance: Covariance = None
    reliability: Reliablity = None

    def is_good(self) -> bool:
        if None in (self.transform, self.covariance, self.reliability):
            return False

        radii, _ = self.covariance.sigma_ellipse()
        max_sigma = np.max(radii)

        if self.localizer == Localizer.VISUAL:
            return (max_sigma < 0.5 and
                    self.reliability.reliability > 0.5)
        elif self.localizer == Localizer.DEPTH:
            return (max_sigma < 0.5 and
                    self.reliability.reliability > 0.8)
        return False

    def as_proto(self) -> TransformEstimation_pb:
        tf_estimation = TransformEstimation_pb(
            transform=self.transform,
            stamp=unix_time_to_pb_timestamp(self.stamp),
        )

        if self.map_id:
            tf_estimation.map_id = self.map_id
        if self.map_name:
            tf_estimation.map_name = self.map_name
        if self.localizer:
            tf_estimation.localizer = self.localizer.as_proto()
        if self.covariance:
            tf_estimation.covariance.CopyFrom(self.covariance.as_proto())
        if self.reliability:
            tf_estimation.reliability.CopyFrom(self.reliability.as_proto())

        return tf_estimation

    @staticmethod
    def from_proto(message: TransformEstimation_pb) -> 'TransformEstimation':
        """
        All member values are required.
        NOTE: But they are not `required` in protobuf, since it may be problematic for the future.
        https://developers.google.com/protocol-buffers/docs/proto#specifying-rules

        raise ValueError if some of the member messages are invalid
        """
        return TransformEstimation(
            stamp=message.stamp.seconds+message.stamp.nanos*1e-9,
            transform=message.transform,
            map_id=message.map_id,
            map_name=message.map_name,
            localizer=Localizer.from_proto(message.localizer),
            covariance=Covariance.from_proto(message.covariance),
            reliability=Reliablity.from_proto(message.reliability),
        )


@define(frozen=True, cache_hash=True)
class Ssid:
    bssid: bytes
    essid: str

    @staticmethod
    def from_strings(bssid: str, essid: str) -> 'Ssid':
        bssid = bytes.fromhex(bssid.replace(':', ''))
        if len(bssid) != 6:
            raise ValueError(f'invalid bssid: {bssid}')
        return Ssid(bssid=bssid, essid=essid)

    def bssid_str(self) -> str:
        return ':'.join(f'{b:02x}' for b in self.bssid)


class StampedAccessPoints(ImmutableMapping[Ssid, AP]):
    """Frozen mapping of access points with a timestamp.

    When constructed by StampedAccessPoints.from_response,
    the stamp is calculated from the last_seen and the diff of unix time and monotonic time at that time.
    The diff may change at each time so it is cached to ensure the same stamp for the same last_seen.
    """
    _LAST_SEEN_UNIX_DIFF_MAP: Dict[int, float] = OrderedDict()
    _LAST_SEEN_UNIX_DIFF_MAP_MAX_SIZE = 100

    def __init__(self, access_points: Mapping, stamp: float) -> None:
        """
        :param access_points: Mapping of access point hardware addresses to their properties.
        :param stamp: Time of the last access point measurement in unix time.
        """
        super().__init__(access_points)
        self._stamp = stamp

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, StampedAccessPoints):
            return self._data == __o._data and self._stamp == __o._stamp
        return NotImplemented

    @property
    def stamp(self) -> float:
        return self._stamp

    @classmethod
    def _convert_monotonic_to_unix_time(cls, monotonic: int) -> float:
        if monotonic not in cls._LAST_SEEN_UNIX_DIFF_MAP:
            cls._LAST_SEEN_UNIX_DIFF_MAP[monotonic] = time.time() - time.monotonic()
            if len(cls._LAST_SEEN_UNIX_DIFF_MAP) > cls._LAST_SEEN_UNIX_DIFF_MAP_MAX_SIZE:
                cls._LAST_SEEN_UNIX_DIFF_MAP.popitem(last=False)
        return cls._LAST_SEEN_UNIX_DIFF_MAP[monotonic] + monotonic

    @classmethod
    def from_response(cls, response: GetAvailableAPResponse) -> 'StampedAccessPoints':
        """Create a StampedAccessPoints from a GetAvailableAPResponse.
        Each AP has a last_seen which is monotonic time, we convert it to unix time.
        """
        if not response.ap:
            raise ValueError('empty access points')

        access_points = {
            Ssid.from_strings(ap.hw_address, ap.ssid): ap
            for ap in response.ap}
        stamp = np.mean([cls._convert_monotonic_to_unix_time(ap.last_seen)
                         for ap in access_points.values()])
        return cls(access_points, stamp=stamp)


@define(frozen=True)
class Fingerprint:
    """Radio fingerprint which contains access points and transform estimation.
    """
    stamp: float
    device_id: str
    access_points: StampedAccessPoints
    transform: TransformEstimation

    def as_proto(self) -> Fingerprint_pb:
        message = Fingerprint_pb(
            stamp=unix_time_to_pb_timestamp(self.stamp),
            device_id=self.device_id,
            transform=self.transform.as_proto(),
        )
        for ssid, ap in self.access_points.items():
            message.access_points[ssid.bssid_str()].CopyFrom(ap)
        return message

    @staticmethod
    def from_proto(message: Fingerprint_pb) -> 'Fingerprint':
        stamp = message.stamp.seconds+message.stamp.nanos*1e-9
        return Fingerprint(
            stamp=stamp,
            device_id=message.device_id,
            access_points=StampedAccessPoints({
                Ssid.from_strings(bssid, ap.ssid): ap
                for bssid, ap in message.access_points.items()}, stamp),
            transform=TransformEstimation.from_proto(message.transform),
        )
