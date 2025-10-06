import os
import pathlib
from enum import Enum, auto
from logging import getLogger
from typing import Union
from lovot_slam.redis.clients import create_ltm_client
from lovot_slam.redis.clients import REDIS_DB_DEVICE, REDIS_DB_STM, REDIS_HOST_DEVICE, REDIS_HOST_STM, REDIS_PORT_DEVICE, REDIS_PORT_STM
from lovot_slam.redis.keys import COLONY_ID_KEY

from lovot_slam.redis.keys import RedisKeyRepository
from lovot_map.utils.map_dir import DataDirectories

import redis


logger = getLogger(__name__)

# TODO: separate this file into some files
# env.py which only contains only paths?
# grpc stuffs
# model stuffs
# network stuffs
# status -> *_slam_manager.py?

MAPSET_ROOT_DIR = pathlib.Path(os.getenv('LOCALIZATION_MAPSET_ROOT_DIR', '/data/localization-mapset'))


DATA_DIR = pathlib.Path(os.getenv('LOCALIZATION_DATA_PATH', '/data/localization'))
data_directories = DataDirectories(DATA_DIR)

redis_keys = RedisKeyRepository()


_SHARE_DIR = pathlib.Path(os.getenv('LOCALIZATION_SHARE_PATH',
                                    '/opt/lovot/share/lovot-localization'))
OMNI_CAMERA_YAML = _SHARE_DIR / 'configs' / 'top_camera_realtime.yaml'
OMNI_CONVERSION_YAML = _SHARE_DIR / 'configs' / 'top_camera.yaml'

_ROS_HOME = pathlib.Path(os.getenv('ROS_HOME', '/var/log/ros'))
ROS_LOG_ROOT = _ROS_HOME / "log"

ENV_PATH = pathlib.Path(os.getenv('LOCALIZATION_ENV_PATH', '/opt/lovot/lib/python3/envs/lovot-localization'))

MAP_2DMAP = '2d_map'
MAP_FEATUREMAP = 'feature_map'
MAP_SUMMARYMAP = 'summary_map'
MAP_YAML = 'lovot_slam.yaml'    # TODO: don't share name with SLAM_YAML
MAP_MD5SUM_YAML = 'md5sum_list.yaml'

MAP_VERTICES = os.path.join(MAP_FEATUREMAP, 'vertices')
MAP_VERTICES_CSV = 'vertices.csv'
MAP_FEATUREMAP_SENSORS_YAML = 'vi_map/sensors.yaml'
MAP_FEATUREMAP_MISSIONS_YAML = 'vi_map/missions.yaml'
MAP_STATISTICS_YAML = 'map_statistics.yaml'

SLAM_YAML = 'lovot_slam.yaml'   # TODO: don't share name with MAP_YAML
SLAM_YAML_KEY_LOVOT_SLAM = 'lovot_slam'
SLAM_YAML_KEY_DATE = 'date'
SLAM_YAML_KEY_SOURCE = 'source'

GRPC_PORT = '50051'  # Changed from 39092 to avoid conflicts
GRPC_TIMEOUT = 10
GRPC_STREAM_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB (allows 100 KiB / sec as min. bandwidth)
GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 3 * 1024 * 1024

CLOUD_UPLOAD_PORT = os.getenv("LOVOT_CLOUD_UPLOAD_PORT", "48480")
CLOUD_HOME_MAP_EVENT_ENDPOINT = "navigation/home-map-event"
CLOUD_SPOT_EVENT_ENDPOINT = "navigation/spot-event"
CLOUD_UNWELCOMED_AREA_EVENT_ENDPOINT = "navigation/unwelcomed-area-event"
MAP_ID = '1'  # 複数地図に対応するまで共通の MAP_ID を使う
AREA_ID = '1'  # 来ないでエリアのid。複数の来ないでエリアに対応するまでは1固定

LOCALHOST = 'localhost'
SPIKE_LOCALHOST = 'spike'

PUSHBAG_RETRY_INTERVAL = 60

WEBVIEW_PORT = int(os.getenv('LOCALIZATION_WEBVIEW_PORT', '48700'))
PROMETHEUS_PORT = int(os.getenv('LOCALIZATION_PROMETHEUS_PORT', '48500'))

MAX_MAP_BUILD_FAILED_COUNT = 3  # remove current map if map update is failed over this count


def get_sentry_info(logger, cpu_name):
    # TOOD: replace this with redis.clients.create_device_client, etc
    db_key_tag_map = {
        (REDIS_HOST_STM, REDIS_PORT_STM, REDIS_DB_STM): {
            f"{cpu_name}:version": f"os_version_{cpu_name}",
        },
        (REDIS_HOST_DEVICE, REDIS_PORT_DEVICE, REDIS_DB_DEVICE): {
            "robot:device_id": "device_id",
            "robot:model": "model",
            "robot:alias": "alias",
        }
    }

    sentry_info = {}
    for redis_info, key_tag_map in db_key_tag_map.items():
        host, port, db = redis_info
        with redis.StrictRedis(host, port, db, socket_connect_timeout=5.0, socket_timeout=1.0, decode_responses=True) as rds:
            for key, tag in key_tag_map.items():
                try:
                    sentry_info[tag] = rds.get(key)
                except (redis.ConnectionError, AttributeError):
                    logger.error(f"Could not fetch {key}")
                    sentry_info[tag] = "unknown"

    return sentry_info


# def get_colony_id():
#     value = create_ltm_client().get(COLONY_ID_KEY)
#     return value if value else None
colony_ID = create_ltm_client().get(COLONY_ID_KEY)


class NestSlamState(Enum):
    IDLE = 0
    BAG_CONVERSION = 1
    BUILD_FEATURE_MAP = 2
    SCALE_MAP = 3
    BUILD_DENSE_MAP = 4
    BAG_PROCESSING = 5
    BUILD_ERROR = auto()

    # NOTE: consider change the values of enum
    def __str__(self) -> str:
        if self.value == NestSlamState.IDLE.value:
            return "idle"
        if self.value == NestSlamState.BAG_CONVERSION.value:
            return "bag_conversion"
        if self.value == NestSlamState.BUILD_FEATURE_MAP.value:
            return "build_feature_map"
        if self.value == NestSlamState.SCALE_MAP.value:
            return "scale_map"
        if self.value == NestSlamState.BUILD_DENSE_MAP.value:
            return "build_dense_map"
        if self.value == NestSlamState.BAG_PROCESSING.value:
            return "bag_processing"
        if self.value == NestSlamState.BUILD_ERROR.value:
            return "error"


class LovotSlamState(Enum):
    LOCALIZATION = 0
    RECORD = 1


SlamState = Union[NestSlamState, LovotSlamState]
