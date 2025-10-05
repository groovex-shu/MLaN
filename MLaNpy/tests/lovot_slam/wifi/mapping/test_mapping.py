import copy
import tempfile
import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

import numpy as np
import pytest
import redis
import trio

from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import Covariance as Covariance_pb
from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import Fingerprint as Fingerprint_pb
from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import Localizer as Localizer_pb
from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import RadioMap as RadioMap_pb
from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import Reliability as Reliability_pb
from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import TransformEstimation as TransformEstimation_pb
from lovot_apis.lovot_minid.wifi.wifi_pb2 import AP, GetAvailableAPResponse
from lovot_apis.lovot_tf.tf.tf_pb2 import GetTransformResponse, Header, Quaternion, Transform, TransformStamped, Vector3

from lovot_slam import Context, context
from lovot_slam.client import open_localization_client, open_lovot_tf_client, open_wifi_service_client
from lovot_slam.client.localization_client import _LOVOT_REDIS_STM
from lovot_slam.utils.protobuf_util import unix_time_to_pb_timestamp
from lovot_slam.wifi.mapping.mapping import Mapping, RadioMap
from lovot_slam.wifi.type import (Covariance, Fingerprint, Localizer, Reliablity, Ssid, StampedAccessPoints,
                                  TransformEstimation)
from lovot_slam.wifi.updater.fingerprint_sync import FingerprintSync
from lovot_slam.wifi.updater.wifi_scan import WiFiScan

from ...client.test_lovot_tf_client import MockTfServicer
from ...client.test_wifi_service_client import MockWifiServicer
from ...client.util import open_servicer_and_client

stamp_message = unix_time_to_pb_timestamp(1674023274.5)
DUMMY_FINGERPRINT = Fingerprint_pb(
    stamp=stamp_message,
    device_id='device_id',
    access_points={
        '00:00:00:00:00:00': AP(hw_address='00:00:00:00:00:00', strength=50, last_seen=10),
        '00:00:00:00:00:01': AP(hw_address='00:00:00:00:00:01', strength=100, last_seen=10),
        '00:00:00:00:00:02': AP(hw_address='00:00:00:00:00:02', strength=80, last_seen=10),
    },
    transform=TransformEstimation_pb(
        stamp=stamp_message,
        transform=Transform(
            translation=Vector3(x=0.1, y=0.2, z=0.3),
            rotation=Quaternion(x=0.1, y=0.2, z=0.3, w=0.4),
        ),
        map_id='1',
        map_name='test',
        localizer=Localizer_pb.VISUAL,
        covariance=Covariance_pb(
            matrix=[0.01, 0.0, 0.00, 0.0, 0.01, 0.0, 0.0, 0.0, 0.005],
            stamp=stamp_message,
        ),
        reliability=Reliability_pb(
            stamp=stamp_message,
            reliability=0.9,
            detection=0.9,
            likelihood=0.8,
        ),
    )
)
VALID_RADIOMAP_MESSAGE = RadioMap_pb(fingerprints=[DUMMY_FINGERPRINT])
DUMMY_AVAILABLE_AP_RESPONSE = GetAvailableAPResponse(ap=[
    AP(hw_address='00:00:00:00:00:00', strength=40, last_seen=10),
    AP(hw_address='00:00:00:00:00:03', strength=90, last_seen=10),
    AP(hw_address='00:00:00:00:00:04', strength=60, last_seen=10),
])
DUMMY_GET_TRANSFORM_RESPONSE = GetTransformResponse(
    transform_stamped=TransformStamped(
        header=Header(
            frame_id="map",
            stamp=stamp_message,
        ),
        child_frame_id="base_link",
        transform=Transform(
            translation=Vector3(x=0.0, y=0.0, z=0.0),
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        ),
    ),
)
DUMMY_LOCALIZATION_REDIS_KEYS = {
    'slam:pose:localizer': 'visual',
    'slam:pose:covariance': {
        'covariance': '0.01,0,0,0,0.01,0,0,0,0.005',
        'timestamp': '123456789.123456789',
    },
    'slam:failure_detection:result': {
        'timestamp': '123456789.123456789',
        'reliability': '0.9',
        'detection': '0.9',
        'likelihood': '0.85',
    },
    'slam:map': {'name': '20230113_184300'},
}


def _generate_stamped_access_points(stamp: float, n: int, last_seen: int, bssid_prefix: int = 0) -> StampedAccessPoints:
    assert n <= 256
    assert bssid_prefix <= 65535
    # generate continuous mac address in string
    bssids = [f'{bssid_prefix//256:02x}:{bssid_prefix%256:02x}:00:00:00:{i:02x}' for i in range(n)]
    essids = [f'essid_{i}' for i in range(n)]

    access_points = {}
    for bssid, essid in zip(bssids, essids):
        access_points[Ssid.from_strings(bssid, essid)] = AP(
            hw_address=bssid,
            ssid=essid,
            flags=0,
            frequency=2412,
            is_connected=False,
            strength=50,
            last_seen=last_seen,
            max_bitrate=0)

    return StampedAccessPoints(access_points, stamp)


def _generate_fingerprint(stamp: float, access_points: StampedAccessPoints,
                          translation: Optional[Vector3] = None) -> Fingerprint:
    if translation is None:
        translation = Vector3(x=0.0, y=0.0, z=0.0)
    transform = TransformEstimation(
        stamp,
        Transform(
            translation=translation,
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        '1',
        'map_name',
        Localizer.VISUAL,
        Covariance(stamp, np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.005]])),
        Reliablity(stamp, 1.0, 1.0, 0.85)
    )
    return Fingerprint(
        stamp,
        'device_id',
        access_points,
        transform,
    )


@asynccontextmanager
async def _timeit(label):
    time_start = time.time()
    print('st')
    yield
    elapsed = time.time() - time_start
    print(f"task '{label}' takes {elapsed:.3} s")


async def test_radio_map_init(monkeypatch):
    # テスト時間の短縮のため (デフォルトは1000)
    monkeypatch.setattr(RadioMap, '_MAXIMUM_FINGERPRINTS', 100)

    radio_map = RadioMap()
    assert list(radio_map._fingerprints) == []
    assert radio_map.ssids == set()

    # 全てのSpotMesaurementが同じAPからのFingerprintを持つ場合
    number_of_aps = 256
    fingerprints = []
    for i in range(RadioMap._MAXIMUM_FINGERPRINTS):
        stamp = float(i)
        fingerprint = _generate_stamped_access_points(stamp, number_of_aps, int(stamp))
        fingerprint = _generate_fingerprint(stamp, fingerprint)
        fingerprints.append(fingerprint)
    async with _timeit('same aps'):
        radio_map = await RadioMap.from_fingerprints(fingerprints)
    assert len(radio_map._fingerprints) == RadioMap._MAXIMUM_FINGERPRINTS
    assert list(radio_map.fingerprints) == fingerprints
    # NOTE: not test with `set`, because using `set()` deletes overlapping elements by itself
    assert sorted([ssid.bssid_str() for ssid in radio_map.ssids]) == \
        sorted([f'00:00:00:00:00:{i:02x}' for i in range(number_of_aps)])

    # 全てユニークなAPからのFingerprintを持つ場合
    number_of_aps = 256
    fingerprints = []
    for i in range(RadioMap._MAXIMUM_FINGERPRINTS):
        stamp = float(i)
        fingerprint = _generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=i)
        fingerprint = _generate_fingerprint(stamp, fingerprint)
        fingerprints.append(fingerprint)
    async with _timeit('unique aps'):
        radio_map = await RadioMap.from_fingerprints(fingerprints)
    assert len(radio_map._fingerprints) == RadioMap._MAXIMUM_FINGERPRINTS
    assert list(radio_map.fingerprints) == list(fingerprints)
    # NOTE: only testing length of ssids
    assert len(radio_map.ssids) == number_of_aps * RadioMap._MAXIMUM_FINGERPRINTS

    # サイズが規定より大きい場合
    number_of_aps = 256
    fingerprints = []
    for i in range(RadioMap._MAXIMUM_FINGERPRINTS + 100):
        stamp = float(i)
        fingerprint = _generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=i)
        fingerprint = _generate_fingerprint(stamp, fingerprint)
        fingerprints.append(fingerprint)
    async with _timeit('too large'):
        radio_map = await RadioMap.from_fingerprints(fingerprints)
    assert len(radio_map._fingerprints) == RadioMap._MAXIMUM_FINGERPRINTS
    assert list(radio_map.fingerprints) == list(fingerprints[-RadioMap._MAXIMUM_FINGERPRINTS:])
    # NOTE: only testing length of ssids
    assert len(radio_map.ssids) == number_of_aps * RadioMap._MAXIMUM_FINGERPRINTS


async def test_map_copy_and_eq():
    radio_map = RadioMap()

    # 全てユニークなAPからのFingerprintを持つ場合
    number_of_aps = 256
    fingerprints = []
    for i in range(100):
        stamp = float(i)
        fingerprint = _generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=i)
        fingerprint = _generate_fingerprint(stamp, fingerprint)
        fingerprints.append(fingerprint)
    radio_map = await RadioMap.from_fingerprints(fingerprints)
    assert list(radio_map.fingerprints) == fingerprints
    # NOTE: only testing length of ssids
    assert len(radio_map.ssids) == number_of_aps * 100

    # コピーしたので同じ
    copied = radio_map.copy()
    assert copied == radio_map

    stamp = float(101)
    fingerprint = _generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=101)
    fingerprint = _generate_fingerprint(stamp, fingerprint)
    radio_map.append_fingerprint(fingerprint)

    # コピー元のみ変更されている
    assert copied != radio_map
    assert copied.fingerprints != radio_map.fingerprints
    assert copied.ssids != radio_map.ssids


def test_radio_map_append_fingerprint(monkeypatch):
    # テスト時間の短縮のため (デフォルトは1000)
    monkeypatch.setattr(RadioMap, '_MAXIMUM_FINGERPRINTS', 100)

    radio_map = RadioMap()
    number_of_aps = 256
    count = 0

    # 最初のfingerprint
    stamp = float(count)
    fingerprint = _generate_stamped_access_points(stamp, number_of_aps, int(stamp))
    fingerprint = _generate_fingerprint(stamp, fingerprint)
    radio_map.append_fingerprint(fingerprint)
    assert list(radio_map.fingerprints) == [fingerprint]
    # NOTE: not test with `set`, because using `set()` deletes overlapping elements by itself
    assert sorted([ssid.bssid_str() for ssid in radio_map.ssids]) == \
        sorted([f'00:00:00:00:00:{i:02x}' for i in range(number_of_aps)])

    # 規定数 + 1のfingerprint
    for i in range(RadioMap._MAXIMUM_FINGERPRINTS):
        count += 1
        stamp = float(count)
        fingerprint = _generate_stamped_access_points(stamp, number_of_aps, int(stamp))
        fingerprint = _generate_fingerprint(stamp, fingerprint)
        radio_map.append_fingerprint(fingerprint)
    assert len(radio_map.fingerprints) == RadioMap._MAXIMUM_FINGERPRINTS
    assert set(ssid.bssid_str() for ssid in radio_map.ssids) == \
        set([f'00:00:00:00:00:{i:02x}' for i in range(number_of_aps)])

    # 別のユニークな256個のAPからのfingerprint
    count += 1
    stamp = float(count)
    fingerprint = _generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=1)
    fingerprint = _generate_fingerprint(stamp, fingerprint)
    radio_map.append_fingerprint(fingerprint)
    assert len(radio_map.fingerprints) == RadioMap._MAXIMUM_FINGERPRINTS
    assert set(ssid.bssid_str() for ssid in radio_map.ssids) == \
        set([f'00:00:00:00:00:{i:02x}' for i in range(number_of_aps)] +
            [f'00:01:00:00:00:{i:02x}' for i in range(number_of_aps)])


def test_radio_map_predict(monkeypatch):
    monkeypatch.setattr(RadioMap, '_MINIMUM_FINGERPRINTS_TO_PREDICT', 1)
    radio_map = RadioMap()

    number_of_aps = 256
    count = 50

    # 全て違うfingerprints
    for i in range(count):
        stamp = float(i)
        translation = Vector3(x=i*10, y=i*20)
        fingerprint = _generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=i)
        fingerprint = _generate_fingerprint(stamp, fingerprint, translation=translation)
        radio_map.append_fingerprint(fingerprint)

    # trainに一致するfingerprintがある場合
    stamp = float(count)
    access_points = _generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=0)
    actual = radio_map.predict(access_points, k=1)
    assert np.allclose(actual, np.array([0, 0]), atol=1e-6)

    access_points = _generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=5)
    actual = radio_map.predict(access_points, k=1)
    assert np.allclose(actual, np.array([50, 100]), atol=1e-6)

    actual = radio_map.predict(access_points, k=3)
    assert np.allclose(actual, np.array([50, 100]), atol=1e-4)

    # 2つのtrainの真ん中 [50, 100]と[60, 120]の中間
    access_points = _generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=5)
    merged_aps = dict(access_points)
    merged_aps.update(_generate_stamped_access_points(stamp, number_of_aps, int(stamp), bssid_prefix=6))
    actual = radio_map.predict(StampedAccessPoints(merged_aps, 0), k=2)
    assert np.allclose(actual, np.array([55, 110]), atol=1e-6)


async def test_radio_map_proto_conversion():
    # 有効なデータの行って返っての変換
    map_ = await RadioMap.from_proto(VALID_RADIOMAP_MESSAGE)
    assert map_.as_proto() == VALID_RADIOMAP_MESSAGE

    # RadioMap.fingerprintsが空の場合
    assert await RadioMap.from_proto(RadioMap_pb())

    # 空のFingerprintがある場合: 例外
    message = RadioMap_pb()
    message.fingerprints.extend([Fingerprint_pb()])
    with pytest.raises(ValueError):
        await RadioMap.from_proto(message)

    # 空のTransformEstimationがある場合: 例外
    message = RadioMap_pb()
    message.fingerprints.extend([Fingerprint_pb(
        transform=TransformEstimation_pb()
    )])
    with pytest.raises(ValueError):
        await RadioMap.from_proto(message)

    # 部分的に不正なデータ (covarianceの行列のサイズが不正): 例外
    copied_message = copy.deepcopy(VALID_RADIOMAP_MESSAGE)
    copied_message.fingerprints[0].transform.covariance.matrix.pop()
    with pytest.raises(ValueError):
        await RadioMap.from_proto(copied_message)

    # 部分的に不正なデータ (reliabilityが0~1の範囲外): 例外
    copied_message = copy.deepcopy(VALID_RADIOMAP_MESSAGE)
    copied_message.fingerprints[0].transform.reliability.reliability = 10.0
    with pytest.raises(ValueError):
        await RadioMap.from_proto(copied_message)

    # 部分的に不正なデータ (localizerが不正): 例外
    copied_message = copy.deepcopy(VALID_RADIOMAP_MESSAGE)
    copied_message.fingerprints[0].transform.ClearField('localizer')
    with pytest.raises(ValueError):
        await RadioMap.from_proto(copied_message)


@pytest.fixture
async def mock_context(nursery):
    async with AsyncExitStack() as stack:
        wifi_scan = WiFiScan()
        fingerprint_sync = FingerprintSync()

        mock_wifi_service, wifi_client = \
            await stack.enter_async_context(open_servicer_and_client(MockWifiServicer, open_wifi_service_client))
        mock_tf_service, lovot_tf_client = \
            await stack.enter_async_context(open_servicer_and_client(MockTfServicer, open_lovot_tf_client))
        
        localization_client = open_localization_client()

        context.set(Context(
            slam_servicer_client=None,
            wifi_client=wifi_client,
            lovot_tf_client=lovot_tf_client,
            localization_client=localization_client,
            fingerprint_sync=fingerprint_sync,
            wifi_scan=wifi_scan,
            radio_map=RadioMap(),
        ))

        mock_wifi_service.response = GetAvailableAPResponse()
        mock_tf_service.response = GetTransformResponse()

        nursery.start_soon(wifi_scan.run)
        nursery.start_soon(fingerprint_sync.run)

        yield mock_wifi_service, mock_tf_service

        nursery.cancel_scope.cancel()


@pytest.fixture
async def mock_radio_map_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        map_file = trio.Path(tmpdir) / 'radio_map.pb'

        from lovot_slam.wifi.mapping import mapping as mapping_module
        monkeypatch.setattr(mapping_module, 'RADIO_MAP_FILE', map_file)

        async with await trio.open_file(map_file, 'wb') as f:
            await f.write(VALID_RADIOMAP_MESSAGE.SerializeToString())

        yield map_file


@pytest.fixture
def redis_keys_setter(request):
    redis_host, redis_port, redis_db = _LOVOT_REDIS_STM
    stm_client = redis.Redis(host=redis_host, port=int(redis_port), db=int(redis_db), decode_responses=True)

    # set keys to redis
    for key, value in DUMMY_LOCALIZATION_REDIS_KEYS.items():
        if isinstance(value, dict):
            stm_client.hset(key, mapping=value)
        elif isinstance(value, str):
            stm_client.set(key, value)

    yield

    # delete keys from redis
    for key in DUMMY_LOCALIZATION_REDIS_KEYS:
        stm_client.delete(key)


async def test_mapping_run(monkeypatch, redis_keys_setter, mock_context, mock_radio_map_file):
    """
    以下の順でテストを行う
    1. ファイルからマップの読み込み
    2. マップの更新
    3. ファイルへのマップの書き込み (cancel時)

    NOTE: don't use autojump_clock, as it sometimes makes the test fail
    """
    mock_wifi_service, mock_tf_service = mock_context

    async with trio.open_nursery() as nursery:
        mapping = Mapping()
        nursery.start_soon(mapping.run)

        # wait for the map to be loaded
        with trio.fail_after(10):
            while len(context.get().radio_map.fingerprints) != 1:
                await trio.sleep(0.1)
        loaded_map = context.get().radio_map
        assert loaded_map.as_proto() == VALID_RADIOMAP_MESSAGE

        mock_wifi_service.response = DUMMY_AVAILABLE_AP_RESPONSE
        mock_tf_service.response = DUMMY_GET_TRANSFORM_RESPONSE

        # wait for the new fingerprint
        with trio.fail_after(10):
            while len(context.get().radio_map.fingerprints) != 2:
                await trio.sleep(0.1)
        assert context.get().radio_map.ssids == \
            set(Ssid.from_strings(f'00:00:00:00:00:{i:02x}', '') for i in (0, 1, 2, 3, 4))

        nursery.cancel_scope.cancel()

    # check the stored map file
    async with await trio.open_file(mock_radio_map_file, 'rb') as f:
        message = RadioMap_pb()
        message.ParseFromString(await f.read())

    assert message.fingerprints[0] == VALID_RADIOMAP_MESSAGE.fingerprints[0]
    key = DUMMY_AVAILABLE_AP_RESPONSE.ap[0].hw_address
    assert message.fingerprints[1].access_points[key] == DUMMY_AVAILABLE_AP_RESPONSE.ap[0]


# It's expected to raise a warning during parsing the corrupted file
@pytest.mark.filterwarnings("ignore:Unexpected end-group tag")
async def test_load_map_from_corrupted_file(monkeypatch, redis_keys_setter,
                                            mock_context, mock_radio_map_file):
    """
    以下の順でテストを行う
    1. ファイルからマップの読み込み (失敗する)
    2. マップの更新
    3. ファイルへのマップの書き込み (cancel時) = 2で更新した分のみ

    NOTE: don't use autojump_clock, as it sometimes makes the test fail
    """
    mock_wifi_service, mock_tf_service = mock_context

    # corrupt the map file
    async with await trio.open_file(mock_radio_map_file, 'wb') as f:
        await f.seek(100)
        await f.write(b'0')

    async with trio.open_nursery() as nursery:
        mapping = Mapping()
        nursery.start_soon(mapping.run)

        mock_wifi_service.response = DUMMY_AVAILABLE_AP_RESPONSE
        mock_tf_service.response = DUMMY_GET_TRANSFORM_RESPONSE

        # wait for the new fingerprint
        with trio.fail_after(10):
            while len(context.get().radio_map.fingerprints) != 1:
                await trio.sleep(0.1)
        assert context.get().radio_map.ssids == \
            set(Ssid.from_strings(f'00:00:00:00:00:{i:02x}', '') for i in (0, 3, 4))

        nursery.cancel_scope.cancel()

    # check the stored map file
    async with await trio.open_file(mock_radio_map_file, 'rb') as f:
        message = RadioMap_pb()
        message.ParseFromString(await f.read())

    key = DUMMY_AVAILABLE_AP_RESPONSE.ap[0].hw_address
    assert message.fingerprints[0].access_points[key] == DUMMY_AVAILABLE_AP_RESPONSE.ap[0]


async def test_mapping_remove(monkeypatch, redis_keys_setter, mock_context, mock_radio_map_file):
    """
    以下の順でテストを行う
    1. ファイルからマップの読み込み
    2. remove_map
    3. ファイルへのマップの書き込み (cancel時)

    NOTE: don't use autojump_clock, as it sometimes makes the test fail
    """
    mock_wifi_service, mock_tf_service = mock_context

    async with trio.open_nursery() as nursery:
        mapping = Mapping()
        nursery.start_soon(mapping.run)

        # wait for the map to be loaded
        with trio.fail_after(10):
            while len(context.get().radio_map.fingerprints) != 1:
                await trio.sleep(0.1)
        loaded_map = context.get().radio_map
        assert loaded_map.as_proto() == VALID_RADIOMAP_MESSAGE

        mock_wifi_service.response = DUMMY_AVAILABLE_AP_RESPONSE
        mock_tf_service.response = DUMMY_GET_TRANSFORM_RESPONSE

        # remove
        await mapping.remove_map()

        radio_map = context.get().radio_map
        assert len(radio_map.ssids) == 0
        assert len(radio_map.fingerprints) == 0

        nursery.cancel_scope.cancel()

    # check the stored map file
    async with await trio.open_file(mock_radio_map_file, 'rb') as f:
        message = RadioMap_pb()
        message.ParseFromString(await f.read())

    assert len(message.fingerprints) == 0
