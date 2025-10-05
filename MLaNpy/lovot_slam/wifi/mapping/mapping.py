import copy
from collections import OrderedDict, defaultdict
from logging import getLogger
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import prometheus_client
import trio
from attr import define, field
from google import protobuf
from trio_util import AsyncBool, move_on_when

from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import RadioMap as RadioMap_pb
from lovot_apis.lovot_tf.tf.tf_pb2 import Quaternion, Transform

from lovot_slam import ContextMixin
from lovot_slam.env import data_directories
from lovot_slam.regressor.knn import Regressor
from lovot_slam.wifi.type import Fingerprint, Ssid, StampedAccessPoints

_logger = getLogger(__name__)

# FIXME: move to instance variable
RADIO_MAP_FILE = trio.Path(data_directories.maps / 'wifi_fingerprints')

REFERENCE_POINTS_RESOLUTION = 0.75
REFERENCE_POINTS_ORIENTATION_RESOLUTION = np.radians(90)

_access_points_in_map_metric = prometheus_client.Gauge(
    'localization_wifi_fingerrpinting_access_points_in_map',
    'number of access points in the current fingerprint map')
_fingerprints_in_map_metric = prometheus_client.Gauge(
    'localization_wifi_fingerrpinting_fingerprints_in_map',
    'number of fingerprints in the current fingerprint map')


def _quaternion_to_euler_yaw(q: Quaternion) -> float:
    """range: [-pi, pi]
    """
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _pose2d_from_transform(transform: Transform) -> np.ndarray:
    return np.array([transform.translation.x, transform.translation.y,
                     _quaternion_to_euler_yaw(transform.rotation)])


@define(frozen=True, cache_hash=True)
class _ReferencePointCoordinate:
    x: int
    y: int
    # quantized orientation
    o: int = field(factory=lambda: 0)

    @o.validator
    def _check_o(self, attribute, value):
        range_ = range(int(2 * np.pi / REFERENCE_POINTS_ORIENTATION_RESOLUTION))
        if value not in range_:
            raise ValueError(f"o must be in {range_}")

    @staticmethod
    def from_transform(transform: Transform) -> '_ReferencePointCoordinate':
        pose = _pose2d_from_transform(transform)
        point = pose[:2].astype(np.int32)
        o = int(np.clip((pose[2] + np.pi) // REFERENCE_POINTS_ORIENTATION_RESOLUTION,
                        0, 2 * np.pi // REFERENCE_POINTS_ORIENTATION_RESOLUTION - 1))
        return _ReferencePointCoordinate(*point, o)


@define
class _ReferencePoint:
    # each element of the stamps is the key of RadioMap._fingerprints
    stamps: Set[float] = field(factory=set)

    def add(self, timestamp: float) -> None:
        self.stamps.add(timestamp)

    def remove(self, timestamp: float) -> None:
        self.stamps.remove(timestamp)


@define(frozen=True, cache_hash=True)
class _RegressorStore:
    timestamp: float
    regressor: Regressor
    query_ssids: Set[Ssid]
    use_reference_points: bool


class RadioMap:
    """Wifi fingerprints database.

    NOTE
    self._ssids may contain SSIDs that are not in self._fingerprints,
    because we don't remove SSIDs when we remove fingerprints due to the size limit.
    """

    _FALLBACK_STRENGTH = 0  # strength value when the SSID is not found in a fingerprint
    _MINIMUM_FINGERPRINTS_TO_PREDICT = 100
    _MAXIMUM_FINGERPRINTS = 1000  # approx ~16 MB with Fingerprints with 256 APs each

    def __init__(self, fingerprints: Optional[Dict[float, Fingerprint]] = None,
                 ssids: Optional[Set[Ssid]] = None) -> None:
        self._fingerprints: OrderedDict[float, Fingerprint] = OrderedDict()
        self._ssids: Set[Ssid] = set()

        assert (fingerprints is None) == (ssids is None), \
            'Both fingerprints and ssids must be given or not given'
        if fingerprints and ssids:
            self._fingerprints = fingerprints
            self._check_size_limit()
            self._ssids = ssids
        self._append_count = len(self._fingerprints)

        self._reference_points: Dict[_ReferencePointCoordinate, _ReferencePoint] = \
            defaultdict(lambda: _ReferencePoint())
        for key, fingerprint in self._fingerprints.items():
            coordinate = _ReferencePointCoordinate.from_transform(fingerprint.transform.transform)
            self._reference_points[coordinate].add(key)

        # regressor cache
        self._regressor_store: _RegressorStore = None

    def _check_size_limit(self) -> None:
        """Check the size limit of the fingerprints and remove the oldest fingerprints if needed.
        Currently checks only the size of the fingerprints,
        but in the future, we may want to check the size of each reference point.
        """
        while len(self._fingerprints) > self._MAXIMUM_FINGERPRINTS:
            stamp, fingerprint = self._fingerprints.popitem(last=False)
            coordinate = _ReferencePointCoordinate.from_transform(fingerprint.transform.transform)
            self._reference_points[coordinate].remove(stamp)

    def __eq__(self, __o: 'RadioMap') -> bool:
        return self._fingerprints == __o._fingerprints and self._ssids == __o._ssids

    def copy(self) -> 'RadioMap':
        ssids = copy.deepcopy(self._ssids)
        fingerprints = copy.deepcopy(self._fingerprints)
        return RadioMap(fingerprints, ssids)

    @property
    def fingerprints(self) -> Iterable[Fingerprint]:
        return self._fingerprints.values()

    @property
    def ssids(self) -> Set[Ssid]:
        return self._ssids

    def latest_stamp(self) -> float:
        if not self._fingerprints:
            return 0
        return self._fingerprints[next(reversed(self._fingerprints))].stamp

    def append_fingerprint(self, fingerprint: Fingerprint) -> None:
        self._fingerprints[fingerprint.stamp] = fingerprint
        coordinate = _ReferencePointCoordinate.from_transform(fingerprint.transform.transform)
        self._reference_points[coordinate].add(fingerprint.stamp)
        self._check_size_limit()

        self._ssids.update(fingerprint.access_points.keys())
        # print([ssid.essid if ssid.essid else ssid.bssid
        #        for ssid in self._ssids])
        self._append_count += 1
        if self._append_count % 10 == 0:
            _logger.info(f'Radio map is updated: '
                         f'{len(self._fingerprints)} fingerprints '
                         f'with {len(self._ssids)} SSIDs')

    def _reindex_access_points_by_ssid(self, access_points: StampedAccessPoints, ssids: List[Ssid]) -> np.ndarray:
        strengths = [access_points[ssid].strength
                     if ssid in access_points else self._FALLBACK_STRENGTH
                     for ssid in ssids]
        return np.array(strengths)

    def serialize_fingerprints_as_ndarray(self, ssids: List[Ssid], use_reference_points: bool) -> Optional[np.ndarray]:
        """Serialize fingerprints as a numpy array.
        shape of the returned array is (N x (M + 2)), where
            M is the number of SSIDs while +2 means x and y coordinates
            N is the number of fingerprints or the number of reference points if use_reference_points is True.
        if use_reference_points is True, the returned array is the mean of the fingerprints in each reference point.
        """
        if use_reference_points:
            return np.array([
                np.mean([np.hstack([self._reindex_access_points_by_ssid(self._fingerprints[stamp].access_points, ssids),
                                    [self._fingerprints[stamp].transform.transform.translation.x,
                                     self._fingerprints[stamp].transform.transform.translation.y]])
                         for stamp in rp.stamps], axis=0)
                for rp in self._reference_points.values()])
        else:
            return np.array([
                np.hstack([self._reindex_access_points_by_ssid(fingerprint.access_points, ssids),
                           [fingerprint.transform.transform.translation.x,
                            fingerprint.transform.transform.translation.y]])
                for fingerprint in self._fingerprints.values()])

    def _create_regressor(self, ssids: List[Ssid], use_reference_points: bool) -> Optional[Regressor]:
        if len(self._fingerprints) < self._MINIMUM_FINGERPRINTS_TO_PREDICT:
            return None

        array = self.serialize_fingerprints_as_ndarray(ssids, use_reference_points=use_reference_points)
        return Regressor(array[:, :-2], array[:, -2:])

    def update_regressor(self, use_reference_points: bool) -> None:
        latest_stamp = self.latest_stamp()
        if self._regressor_store and \
                self._regressor_store.timestamp == latest_stamp and \
                self._regressor_store.use_reference_points == use_reference_points:
            return

        query_ssids = list(self._ssids)
        regressor = self._create_regressor(query_ssids, use_reference_points)
        if regressor:
            self._regressor_store = _RegressorStore(
                latest_stamp,
                regressor,
                query_ssids,
                use_reference_points,
            )

    def predict(self, access_points: StampedAccessPoints, k: int,
                use_reference_points: bool = True) -> Optional[np.ndarray]:
        """Predict the position of the given WiFi fingerprint.

        :param access_points: WiFi access points
        :param k: number of neighbors to use for prediction
        :param use_reference_points: if True,
            the prediction is done by the mean of the fingerprints in each reference point.
        :return: predicted position (x, y) or None if the fingerprints are not enough
        """
        self.update_regressor(use_reference_points)
        if self._regressor_store:
            regressor = self._regressor_store.regressor
            query_ssids = self._regressor_store.query_ssids
            pos = regressor.predict(self._reindex_access_points_by_ssid(access_points, query_ssids), k)
            return pos

    def as_proto(self) -> RadioMap_pb:
        return RadioMap_pb(
            fingerprints=[fingerprint.as_proto() for fingerprint in self._fingerprints.values()])

    @staticmethod
    async def from_fingerprints(fingerprints: List[Fingerprint]) -> 'RadioMap':
        """Create a RadioMap from a list of Fingerprints.
        The fingerprints are sorted by timestamp and the size is limited to _MAXIMUM_FINGERPRINTS.
        NOTE: 0.10 sec for 1000 fingerprints with 256 APs each (i7-8700 CPU @ 3.20GHz)
        """
        fingerprints_: OrderedDict[float, Fingerprint] = OrderedDict()
        ssids: Set[Ssid] = set()
        sorted_fingerprints = sorted(fingerprints, key=lambda m: m.stamp)
        for fingerprint in sorted_fingerprints[-RadioMap._MAXIMUM_FINGERPRINTS:]:
            await trio.sleep(0)
            fingerprints_[fingerprint.stamp] = fingerprint
            ssids.update(fingerprint.access_points.keys())
        return RadioMap(fingerprints_, ssids)

    @staticmethod
    async def from_proto(message: RadioMap_pb) -> 'RadioMap':
        return await RadioMap.from_fingerprints(
            [Fingerprint.from_proto(fingerprint)
             for fingerprint in message.fingerprints])


class Mapping(ContextMixin):
    def __init__(self) -> None:
        self._remove_event = AsyncBool(False)

    async def _store(self, path) -> None:
        async with await trio.open_file(path, 'wb') as f:
            await f.write(self.context.radio_map.as_proto().SerializeToString())
        _logger.info(f'Radio map is saved: '
                     f'{len(self.context.radio_map.fingerprints)} fingerprints '
                     f'with {len(self.context.radio_map.ssids)} SSIDs')

    async def _load(self, path) -> None:
        # Load radio map protobuf message from file
        try:
            async with await trio.open_file(path, 'rb') as f:
                message = RadioMap_pb()
                message.ParseFromString(await f.read())
        except EnvironmentError as e:
            _logger.warning(f'Failed to load radio map from file: {e}')
            return
        except protobuf.message.DecodeError as e:
            _logger.warning(f'Failed to decode radio map: {e}')
            return

        # Convert protobuf message to RadioMap
        try:
            self.context.radio_map = await RadioMap.from_proto(message)
        except ValueError as e:
            _logger.warning(f'Failed to load radio map: {e}')
            return
        _logger.info(f'Radio map is loaded: '
                     f'{len(self.context.radio_map.fingerprints)} fingerprints '
                     f'with {len(self.context.radio_map.ssids)} SSIDs')

    def _update_metrics(self) -> None:
        _access_points_in_map_metric.set(len(self.context.radio_map.ssids))
        _fingerprints_in_map_metric.set(len(self.context.radio_map.fingerprints))

    async def remove_map(self) -> None:
        """Reset the WiFi fingerprints map and remove the map file.
        """
        self._remove_event.value = True
        if await RADIO_MAP_FILE.exists():
            await RADIO_MAP_FILE.unlink()
            _logger.info('WiFi fingerprints map file is removed')
        await self._remove_event.wait_value(False)

    async def run(self) -> None:
        """Synchronize with the fingerprint stream and update the radio map if the transform estimation is good.
        """
        fingerprint_sync = self.context.fingerprint_sync

        if await RADIO_MAP_FILE.exists():
            await self._load(RADIO_MAP_FILE)
        self._update_metrics()

        try:
            while True:
                async with move_on_when(self._remove_event.wait_value, True):
                    async for fingerprint, _ in fingerprint_sync.fingerprint_event.transitions():
                        if not fingerprint.transform.is_good():
                            continue
                        # Update the radio map only when the transform estimation is good
                        self.context.radio_map.append_fingerprint(fingerprint)
                        self._update_metrics()
                _logger.info('Reset WiFi fingerprints map')
                self._remove_event.value = False
                self.context.radio_map = RadioMap()
                self._update_metrics()
                # TODO: wait for the visual map is loaded (and move wait to the beginning of the loop)
                await trio.sleep(60)
        finally:
            with trio.CancelScope(shield=True):
                await RADIO_MAP_FILE.parent.mkdir(exist_ok=True)
                await self._store(RADIO_MAP_FILE)
