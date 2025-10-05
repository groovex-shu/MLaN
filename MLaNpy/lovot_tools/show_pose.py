import argparse
import base64
import binascii
import json
import math
import sys
import uuid
from enum import Enum
from logging import DEBUG, INFO, WARNING, getLogger
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import coloredlogs
import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from websocket import WebSocketTimeoutException, create_connection

logger = getLogger(__name__)


class _DrawSet(NamedTuple):
    """Poseを描画するための情報"""
    parent_frame: str
    child_frame: str
    color: Tuple[int, int, int]
    covariance_key: Optional[str]
    color_ellipse: Optional[Callable[['TransformWithCovariance'], Tuple[int, int, int]]]
    localizer_key: Optional[str]
    only_position: bool


DRAW_SETS = [
    _DrawSet('omni_map', 'base', (0, 191, 255),
             'slam:pose:covariance',
             lambda tf_w_c: (0, 191, 255) if tf_w_c.localizer == Localizer.VISUAL else (255, 0, 0),
             'slam:pose:localizer',
             False),
    # 以下、必要な情報を追加して使う
    # visual only
    # _DrawSet('omni_map_visual', 'base', (191, 230, 0),
    #          'slam:pose_visual:covariance',
    #          lambda _: (0, 230, 0),
    #          None,
    #          False),
    # depth only
    # _DrawSet('omni_map_depth', 'base', (255, 191, 0),
    #          'slam:pose:amcl',
    #          lambda _: (255, 0, 0),
    #          None,
    #          False),
    # others...
    # _DrawSet('wifi_map_ev', 'wifi_base_ev', (140, 90, 210),
    #          None, None, None, True),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Get map and pose of the lovot and plot them')
    parser.add_argument('host', help='host name or ip address')
    parser.add_argument('--debug', action='store_true', default=False, help='output debug log')
    args = parser.parse_args()
    return args


def setup_logger(debug=False):
    loglevel = DEBUG if debug else INFO
    coloredlogs.install(level=loglevel)
    # suppress matplotlib log
    mpl_logger = getLogger('matplotlib')
    mpl_logger.setLevel(WARNING)


class Map(NamedTuple):
    """
    2D map data
    please refer: http://wiki.ros.org/map_server
    """
    width: int  # width of the map image in pixel
    height: int  # height of the map image in pixel
    resolution: float  # resolution of the grid [meter / pixel], normally 0.05
    image: np.ndarray  # 2d map image
    # the 2d pose of the lower-left pixel in the map
    # [px, py, pz, ox, oy, oz, ow]
    # currently ignore rotation
    origin: List[float]

    @classmethod
    def from_hashed_values(cls, hashed_valuse: List[str]) -> Optional['Map']:
        logger.debug(f'map values: {hashed_valuse}')
        width, height, resolution, image, origin = hashed_valuse
        try:
            width, height = int(width), int(height)
            resolution = float(resolution)
            origin = list(map(float, origin.split(',')))
        except (ValueError, TypeError):
            logger.error(f'map decode error {hashed_valuse}')
            return None

        try:
            image = base64.b64decode(image)
            image = np.frombuffer(image, dtype='uint8')
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except (binascii.Error, cv2.error):
            logger.error(f'map decode error {hashed_valuse}')
            return None

        assert image.shape == (height, width, 3)

        return cls(width, height, resolution, image, origin)


class Transform(NamedTuple):
    parent_frame: str  # parent frame id
    child_frame: str  # child frame id
    timestamp: float  # unix timestamp in seconds
    translation: np.ndarray  # translation in the map coordinate system
    rotation: np.ndarray  # quaternion in the map coordinate system in order of x, y, z, w

    @classmethod
    def from_dict(cls, dict) -> Optional['Transform']:
        logger.debug(f'tf dict: {dict}')
        try:
            parent_frame = dict['parent_frame']
            child_frame = dict['child_frame']
            timestamp = float(dict['timestamp'])
            translation = np.array([float(dict['translation'][key]) for key in 'xyz'])
            rotation = np.array([float(dict['rotation'][key]) for key in 'xyzw'])
            return cls(parent_frame, child_frame, timestamp, translation, rotation)
        except (TypeError, KeyError):
            logger.error(f'tf decode error {dict}')
            return None

    @property
    def yaw(self) -> float:
        q = self.rotation
        return math.atan2(2 * (q[3] * q[2] + q[0] * q[1]),
                          1 - 2 * (q[1] * q[1] + q[2] * q[2]))


class Covariance(NamedTuple):
    matrix: np.ndarray  # 3x3 covariance matrix [x, y, yaw]

    def sigma_ellipse(self) -> Tuple[np.ndarray, float]:
        """Obtain ellipse shape from covariance matrix.
        return lengths of major and minor axis (magnitude of sigma),
        and angle of the principal axis.
        accepts only a symmetric matrix (if not, returns complex values)
        """
        eigen_value, eigen_vector = np.linalg.eig(self.matrix[:2, :2])
        angle = np.arctan2(eigen_vector[1, 0], eigen_vector[0, 0])
        # if the eigenvalues are almost 0,
        # the obtained values can be minus (e.g. -2.49920001e-15),
        # due to the numerical calculation error.
        # so clip them to positive and realistic range (sigma = 10.0 m, variance = 10.0**2)
        variances = np.clip(eigen_value, 0, 100.0)
        return np.sqrt(variances), angle


class Localizer(Enum):
    VISUAL = 'visual'
    DEPTH = 'depth'

    @classmethod
    def value_of(cls, target_value):
        for e in Localizer:
            if e.value == target_value:
                return e
        return None


class TransformWithCovariance(NamedTuple):
    transform: Transform
    covariance: Optional[Covariance]
    localizer: Optional[Localizer]


class LtClient:
    def __init__(self, host: str):
        self._host = host
        self._ws = create_connection(f'ws://{host}:38001')
        self._ws.settimeout(2)

    def get_map(self) -> Optional[Map]:
        attrs = {'keys': ['slam:map'],
                 'fields': ['width', 'height', 'resolution', 'image', 'origin']}
        result = self._query_and_receive('?STM,HMGET', attrs)
        if 'slam:map' not in result or not result['slam:map']:
            logger.error('failed to get map')
            return
        return Map.from_hashed_values(result['slam:map'])

    def get_tf(self, parent_frame: str, child_frame: str) -> Optional[Transform]:
        attrs = {
            "parent_frame_id": parent_frame,
            "child_frame_id": child_frame,
            "timestamp": 0,
            "fixed_frame_id": "",
            "parent_stamp": 0,
            "child_stamp": 0,
        }
        result = self._query_and_receive('?TF,GET', attrs)
        if not result:
            logger.error('failed to get tf')
            return
        return Transform.from_dict(result)

    def get_localizer(self, key: str) -> Optional[Localizer]:
        attrs = {
            'keys': [key]
        }
        result = self._query_and_receive('?STM,MGET', attrs)
        if not result:
            logger.error('failed to localizer type')
            return None
        return Localizer.value_of(result[0])

    def get_covariance(self, key: str) -> Tuple[Optional[Covariance], Optional[float]]:
        attrs = {
            'keys': [key],
            'fields': ['timestamp', 'covariance']
        }
        result = self._query_and_receive('?STM,HMGET', attrs)
        if not result or key not in result:
            logger.error('failed to get covariance')
            return None, None
        covariance = list(map(float, result[key][1].split(",")))
        if len(covariance) == 36:
            # extract only [x, y, yaw]
            covariance = np.array(covariance).reshape(6, 6)[[0, 1, 5], :][:, [0, 1, 5]]
        elif len(covariance) == 9:
            covariance = np.array(covariance).reshape(3, 3)
        else:
            raise RuntimeError(f'invalid covariance size: {len(covariance)}')
        timestamp = float(result[key][0])
        return Covariance(covariance), timestamp

    def get_transform_with_covariance(self, parent_frame: str, child_frame: str,
                                      covariance_key: Optional[str] = None,
                                      localizer_key: Optional[str] = None) -> Optional[TransformWithCovariance]:
        tf = self.get_tf(parent_frame, child_frame)
        if not tf:
            return None

        covariance = None
        if covariance_key:
            covariance, timestamp = self.get_covariance(covariance_key)

        localizer = None
        if localizer_key:
            localizer = self.get_localizer(localizer_key)

        return TransformWithCovariance(tf, covariance, localizer)

    def _query(self, cmd: str, attrs: Dict[str, str]) -> str:
        qid = str(uuid.uuid4())
        query = {
            "cmd": cmd,
            "qid": qid,
            "attrs": attrs
        }
        self._ws.send(json.dumps(query))
        return qid

    def _query_and_receive(self, cmd: str, attrs: Dict[str, str]):
        qid = self._query(cmd, attrs)
        try:
            response = json.loads(self._ws.recv())
            if "qid" not in response or response["qid"] != qid:
                return
            if "result" not in response:
                return
            return response["result"]
        except WebSocketTimeoutException:
            return


class PoseDrawer:
    SCALE = 5

    def __init__(self, grid_map: Map):
        self._grid_map = grid_map

        # flip image upside down,
        # because the origin of opencv coords is upper-left while the origin of tf is lower-left.
        self._flipped_img = np.copy(grid_map.image[::-1, :, :])
        self._flipped_img = cv2.resize(self._flipped_img, None, fx=self.SCALE, fy=self.SCALE)

    @property
    def image(self) -> np.ndarray:
        return self._flipped_img[::-1, :, :]

    def to_point(self, position: np.ndarray) -> Tuple[int, int]:
        return tuple((position / self._grid_map.resolution * self.SCALE).astype(np.int32).tolist())

    def draw_position(self, position: np.ndarray, color: Tuple[int, int, int],
                      radius_m: float = 0.15) -> None:
        pos1 = position[:2] - self._grid_map.origin[:2]
        self._flipped_img = cv2.circle(
            self._flipped_img,
            self.to_point(pos1),
            int(radius_m / self._grid_map.resolution * self.SCALE),
            color, thickness=-1)

    def draw_transform(self, transform: Transform, color: Tuple[int, int, int]) -> None:
        self.draw_position(transform.translation, color)

        # start and end position of the arrow in meter
        arrow_len_m = 0.35
        pos1 = transform.translation[:2] - self._grid_map.origin[:2]
        pos2 = pos1 + arrow_len_m * np.array((math.cos(transform.yaw), math.sin(transform.yaw)))
        self._flipped_img = cv2.arrowedLine(
            self._flipped_img,
            self.to_point(pos1),
            self.to_point(pos2),
            color, thickness=self.SCALE, tipLength=0.5)

    def draw_covariance_ellipse(self, transform: Transform, covariance: Covariance,
                                color: Tuple[int, int, int], alpha: float = 0.3) -> None:
        pos1 = transform.translation[:2] - self._grid_map.origin[:2]

        axes, angle = covariance.sigma_ellipse()
        overlay = self._flipped_img.copy()
        overlay = cv2.ellipse(overlay,
                              (self.to_point(pos1),
                               self.to_point(2 * axes),
                               math.degrees(angle)),
                              color, thickness=-1)
        overlay = cv2.ellipse(overlay,
                              self.to_point(pos1),
                              (int(0.5 / self._grid_map.resolution * self.SCALE),
                               int(0.5 / self._grid_map.resolution * self.SCALE)),
                              math.degrees(transform.yaw),
                              -math.degrees(covariance.matrix[2, 2]),
                              math.degrees(covariance.matrix[2, 2]),
                              color, thickness=-1)

        self._flipped_img = cv2.addWeighted(
            overlay, alpha, self._flipped_img, 1 - alpha, 0)
        self._flipped_img = cv2.ellipse(
            self._flipped_img,
            (self.to_point(pos1), self.to_point(2 * axes), math.degrees(angle)),
            color)

    def draw_transform_with_covariance(self, transform_with_covariance: TransformWithCovariance,
                                       color: Tuple[int, int, int], color_ellipse: Tuple[int, int, int]) -> None:
        self.draw_covariance_ellipse(transform_with_covariance.transform,
                                     transform_with_covariance.covariance,
                                     color_ellipse)
        self.draw_transform(transform_with_covariance.transform, color)


def run():
    args = parse_args()
    setup_logger(args.debug)

    lt_client = LtClient(args.host)
    # get map data via websocket
    grid_map = lt_client.get_map()
    if not grid_map:
        logger.error('failed to get map')
        sys.exit(-1)

    logger.info(f'map is loaded: {grid_map.width} x {grid_map.height} and '
                f'origin is {grid_map.origin[:2]} ')

    fig, ax = plt.subplots()
    im = plt.imshow(grid_map.image)

    def update(*args):
        drawer = PoseDrawer(grid_map)
        for i, draw_set in enumerate(DRAW_SETS):
            tf_w_c = lt_client.get_transform_with_covariance(
                draw_set.parent_frame, draw_set.child_frame,
                draw_set.covariance_key, draw_set.localizer_key)
            if not tf_w_c:
                logger.warning('failed to get tf')
                continue

            if draw_set.only_position:
                drawer.draw_position(tf_w_c.transform.translation[:2], draw_set.color)
            else:
                drawer.draw_transform_with_covariance(
                    tf_w_c, draw_set.color, draw_set.color_ellipse(tf_w_c))

            if i == 0:
                logger.info(f'[{tf_w_c.transform.timestamp:.9f}]: '
                            f'x {tf_w_c.transform.translation[0]:5.2f}, '
                            f'y {tf_w_c.transform.translation[1]:5.2f}, '
                            f'yaw {math.degrees(tf_w_c.transform.yaw):7.2f} deg, '
                            f'{tf_w_c.localizer}')

        nonlocal im
        im.set_array(drawer.image)
        return [im]

    _ = animation.FuncAnimation(fig, update, interval=50)
    plt.show()


if __name__ == '__main__':
    run()
