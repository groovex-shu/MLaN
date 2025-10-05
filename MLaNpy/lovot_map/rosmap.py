import base64
import cv2
import math
import os

import numpy as np
from scipy import ndimage
import yaml

from lovot_nav.protobufs.navigation_pb2 import HomeMap, Coordinate
from lovot_nav.protobufs.domain_event_pb2 import HomeMapEvent
from lovot_map.occupancy_grid import OccupancyGrid
from lovot_slam.env import MAP_ID


def quaternion_from_euler(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return list(map(float, [qx, qy, qz, qw]))


def quaternion_to_euler(x: float, y: float, z: float, w: float):
    """ Convert quaternion to euler

    :return: Euler angles (rad). [roll, pitch, yaw]
    """
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def enc(d):
    if d < 100:  # Obstacle
        return '2'
    elif d < 210:  # Unknown
        return '1'
    else:  # No obstacles
        return '0'


def dec(d):
    if d == 2:  # Obstacle
        return 0
    elif d == 1:  # Unknown
        return 205
    else:  # No obstacles
        return 254


def create_kernel(robot_radius=0.15, map_resolution=0.05):
    radius = round(robot_radius / map_resolution)
    diameter = round(radius * 2 + 1)
    xx, yy = np.mgrid[:diameter, :diameter]
    circle = (xx - radius) ** 2 + (yy - radius) ** 2
    kernel = circle <= round(radius ** 2)
    return kernel


def get_navigatable_area(map_data, width, height, map_resolution):
    ''' generate a "navigatable_area" map from map_data.
        navigatable_area map is a floor map on which LOVOT
        can specify destination of global navigation.

        returns: map composed of 0 (navigatable area) and 2 (un-navigatable area)
    '''
    NO_OBSTACLE = 0
    UNKNOWN = 1
    OBSTACLE = 2

    # convert map_data into 2d numpy array
    map2d = np.array(map_data).astype(np.uint8).reshape(height, width)

    # strip off Unknown area from map, then dilate the obstacle
    #     obstacle: 1, unknown & floor: 0
    obstacle_map = np.where(map2d == UNKNOWN, NO_OBSTACLE, map2d)
    dilated_obstacle_map = ndimage.binary_dilation(
        obstacle_map, structure=create_kernel(map_resolution=map_resolution)).astype(int)

    # bring back the unknown area (as obstacle) into the dilated map
    unknown_map = np.where(map2d == OBSTACLE, NO_OBSTACLE, map2d)
    navigatable_map = dilated_obstacle_map | unknown_map

    # convert UNKNOWN code to OBSTACLE to maintain consistency to original map
    navigatable_map = np.where(navigatable_map == UNKNOWN, OBSTACLE, navigatable_map)

    # convert 2d int array to 1d string list, as Rosmap.data
    return list(map(str, navigatable_map.flatten().tolist()))


class RosMap:

    def __init__(self, name, free_thresh, negate, occupied_thresh, origin_pos_2d, origin_yaw, resolution, image):
        self.name = name
        self.free_thresh = free_thresh
        self.negate = negate
        self.occupied_thresh = occupied_thresh
        self.origin_pos_2d = origin_pos_2d
        self.origin_yaw = origin_yaw
        self.resolution = resolution
        self.image = image
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.position = [float(self.origin_pos_2d[0]), float(self.origin_pos_2d[1]), 0.0]
        self.orientation = quaternion_from_euler(0.0, 0.0, self.origin_yaw)
        self.data = [enc(d) for d in self.image[::-1][:].astype(np.uint8).flatten().tolist()]
        self.image = self.to_png(self.data, self.width, self.height)

    @staticmethod
    def decode_to_image(data, width, height):
        imageArray = np.zeros((height, width, 1), np.uint8)
        for h in range(0, height):
            for w in range(0, width):
                i = h * width + w
                imageArray[height - h - 1, w] = dec(int(data[i]))
        return imageArray

    @staticmethod
    def to_png(data, width, height):
        imageArray = np.zeros((height, width, 1), np.uint8)
        for h in range(0, height):
            for w in range(0, width):
                i = h * width + w
                imageArray[height - h - 1, w] = 255 - int(data[i]) * 127
        result, png = cv2.imencode(".png", imageArray)
        return png if result else None

    @classmethod
    def from_map_yaml(cls, map_name, map_yaml_path):
        directory = os.path.dirname(map_yaml_path)
        with open(map_yaml_path, 'r') as f:
            map_conf = yaml.safe_load(f)
        image_path = os.path.join(directory, os.path.basename(map_conf['image']))
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return cls(
            name=map_name,
            free_thresh=map_conf['free_thresh'],
            negate=map_conf['negate'],
            occupied_thresh=map_conf['occupied_thresh'],
            origin_pos_2d=np.array(map_conf['origin'][0:2]),
            origin_yaw=map_conf['origin'][2],
            resolution=map_conf['resolution'],
            image=image,
        )

    @classmethod
    def from_hashed_values(cls, hashed_values):
        width, height, resolution, data, origin, name = hashed_values
        origin = [float(value) for value in origin.split(',')]
        image = cls.decode_to_image(data, int(width), int(height))
        # origin: [px, py, pz, ox, oy, oz, ow]
        _, _, yaw = quaternion_to_euler(origin[3], origin[4], origin[5], origin[6])
        return cls(
            name=name,
            free_thresh=0.196,
            negate=0,
            occupied_thresh=0.65,
            origin_pos_2d=np.array(origin[:2]),
            origin_yaw=yaw,
            resolution=float(resolution),
            image=image,
        )

    @classmethod
    def from_occupancy_grid(cls, occupancy_grid: OccupancyGrid, name=''):
        image = occupancy_grid.img
        resolution = occupancy_grid.resolution
        origin_pos_2d = occupancy_grid.origin
        origin_yaw = occupancy_grid.origin_yaw
        free_thresh = occupancy_grid.free_thresh
        occupied_thresh = occupancy_grid.free_thresh
        negate = occupancy_grid.free_thresh
        return RosMap(name=name,
                      free_thresh=free_thresh, negate=negate, occupied_thresh=occupied_thresh,
                      origin_pos_2d=origin_pos_2d, origin_yaw=origin_yaw,
                      resolution=resolution, image=image)

    def as_occupancy_grid(self):
        img = self.decode_to_image(self.data, self.width, self.height)
        img = img.reshape(img.shape[:2])
        return OccupancyGrid(img, self.resolution, np.array(self.origin_pos_2d),
                             origin_yaw=self.origin_yaw)

    @property
    def origin_array(self):
        return self.position + self.orientation

    @property
    def origin_str(self):
        return ', '.join(map(str, self.origin_array))

    def to_dict_for_redis(self):
        return {
            'width': self.width,
            'height': self.height,
            'resolution': self.resolution,
            'data': ''.join(self.data),
            'image': base64.b64encode(self.image),
            'name': self.name,
            'origin': self.origin_str,
        }

    def to_proto(self, colony_id: str) -> HomeMap:
        return HomeMap(
            colony_id=colony_id,
            map_id=MAP_ID,
            name=self.name,
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            data=''.join(self.data),
            origin=Coordinate(
                px=self.position[0],
                py=self.position[1],
                pz=self.position[2],
                ox=self.orientation[0],
                oy=self.orientation[1],
                oz=self.orientation[2],
                ow=self.orientation[3],
            ),
            data_navigatable=''.join(get_navigatable_area(self.data, self.width, self.height, self.resolution))
        )

    def to_event_proto(self, colony_id: str) -> HomeMapEvent:
        home_map = self.to_proto(colony_id)
        return HomeMapEvent(
            colony_id=colony_id,
            map_id=home_map.map_id,
            event=HomeMapEvent.home_map_generated,
            home_map=home_map,
        )

    def get_inversion_matrix(self) -> np.ndarray:
        """omni_map座標系の位置を2d map画像のピクセルに変換する(2,3)行列を計算"""
        # TODO: take .origin_yaw into account
        origin_inv_2d = np.array([[0., 1., 0.], [1., 0., 0.]])
        origin_inv_2d[::-1, 2] = -self.origin_pos_2d
        origin_inv_2d /= self.resolution
        return origin_inv_2d
