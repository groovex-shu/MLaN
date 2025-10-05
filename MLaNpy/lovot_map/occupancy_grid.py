import pathlib
from logging import getLogger
from typing import Optional

import cv2
import numpy as np
import yaml

logger = getLogger(__name__)


class OccupancyGrid:
    """
    Container class of occupancy grid with some utility methods.
    This class handles the codes (0, 205, 254) as they are, without converting them to probabilities.
    """
    OCCUPIED_CODE = 0
    UNKNOWN_CODE = 205
    FREE_CODE = 254

    def __init__(self, img: np.ndarray, resolution: float, origin: np.ndarray,
                 origin_yaw: float = 0.0,
                 free_thresh: float = 0.196, occupied_thresh: float = 0.65, negate: float = 0):
        self._img = img
        self._resolution = resolution
        self._origin = origin
        self._origin_yaw = origin_yaw
        self._free_thresh = free_thresh
        self._occupied_thresh = occupied_thresh
        self._negate = negate

    @property
    def img(self) -> np.ndarray:
        return self._img

    @property
    def resolution(self) -> float:
        return self._resolution

    @property
    def origin(self) -> np.ndarray:
        return self._origin

    @property
    def origin_yaw(self) -> float:
        return self._origin_yaw

    @property
    def free_thresh(self) -> float:
        return self._free_thresh

    @property
    def occupied_thresh(self) -> float:
        return self._occupied_thresh

    @property
    def negate(self) -> float:
        return self._negate

    @classmethod
    def from_yaml_file(cls, yaml_path: str) -> 'OccupancyGrid':
        directory = pathlib.Path(yaml_path).parent
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            origin = np.array(config.get('origin', [0, 0, 0])[0:2])
            origin_yaw = float(config.get('origin', [0, 0, 0])[2])
            resolution = config.get('resolution', 0.05)
            free_thresh = config.get('free_thresh', 0.0)
            occupied_thresh = config.get('occupied_thresh', 0)
            negate = config.get('negate', 0)
            # read image file
            image_name = pathlib.Path(config.get('image', 'map.pgm')).name
            image_path = directory / image_name
            img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
        except OSError as e:
            raise RuntimeError(f"Map file doesn't exist: {e}")
        except (yaml.YAMLError, AttributeError, IndexError) as e:
            raise RuntimeError(f"Invalid map yaml: {e}")

        if resolution == 0.0:
            raise RuntimeError("Invalid map data with resolution=0.0.")
        if img is None:
            raise RuntimeError("Failed to read image file.")
        if len(img.shape) != 2:
            raise RuntimeError(f"Image shape {img.shape} is not supported.")
        if img.dtype not in (np.uint8, np.uint16):
            raise RuntimeError(f"Bit depth {img.dtype} is not supported.")

        return cls(img, resolution, origin,
                   origin_yaw=origin_yaw,
                   free_thresh=free_thresh, occupied_thresh=occupied_thresh, negate=negate)

    def save(self, yaml_path: str) -> None:
        yaml_path_ = pathlib.Path(yaml_path)
        directory = yaml_path_.parent
        if not directory.exists():
            directory.mkdir(parents=True)

        # save map image to pgm
        image_path = directory / (yaml_path_.stem + '.pgm')
        cv2.imwrite(str(image_path), self._img)

        # save map information to yaml
        config = {}
        config['image'] = image_path.name
        config['resolution'] = self._resolution
        config['origin'] = self._origin.tolist() + [self._origin_yaw]
        config['free_thresh'] = self._free_thresh
        config['occupied_thresh'] = self._occupied_thresh
        config['negate'] = self._negate
        with open(yaml_path, 'w') as f:
            f.write(yaml.safe_dump(config))

    def get_area_pixel_square(self) -> int:
        try:
            return self._img.shape[0] * self._img.shape[1]
        except IndexError:
            raise RuntimeError("invalid map image")

    def get_area_meter_square(self) -> float:
        try:
            return self._img.shape[0] * self._img.shape[1] * self._resolution * self._resolution
        except IndexError:
            raise RuntimeError("invalid map image")

    def get_area_of(self, code: int) -> float:
        """Get area [m^2] of the given code
        """
        if len(self.img.shape) == 2 and self.resolution:
            return np.count_nonzero(self.img == code) * self.resolution * self.resolution
        return 0

    def get_free_area(self) -> float:
        """Get area [m^2] of free area
        """
        return self.get_area_of(OccupancyGrid.FREE_CODE)

    def get_occupied_area(self) -> float:
        """Get area [m^2] of occupied area
        """
        return self.get_area_of(OccupancyGrid.OCCUPIED_CODE)

    def get_unknown_area(self) -> float:
        """Get area [m^2] of unknown area
        """
        return self.get_area_of(OccupancyGrid.UNKNOWN_CODE)

    def realcoords_to_cvcoords(self, position: np.ndarray) -> np.ndarray:
        # x in opencv coords is column in numpy
        # y in opencv coords is row in numpy
        return self.realcoords_to_npcoords(position)[::-1]

    def realcoords_to_npcoords(self, position: np.ndarray) -> np.ndarray:
        """Convert real world coordinate (x, y) [m] to numpy index.
        NOTE: numpy index is column first (y axis first)
        """
        pixel = ((position - self._origin) / self._resolution + 0.5).astype(np.int32)
        return np.array([self._img.shape[0] - pixel[1] - 1, pixel[0]])

    def npcoords_to_realcoords(self, point: np.ndarray) -> np.ndarray:
        """Convert numpy index to real world coordinate (x, y) [m].
        NOTE: numpy index is column first (y axis first)
        """
        origin_offset = np.round(self._origin / self._resolution)
        return (np.array([point[1], self._img.shape[0] - point[0] - 1]) + origin_offset) * self._resolution

    def is_inside(self, point: np.ndarray) -> bool:
        """Check if the given point is inside the map"""
        return np.all(0 <= point) and np.all(point < self._img.shape[::-1])

    def _create_kernel(self, radius: float) -> np.ndarray:
        pixel_radius = int(radius / self._resolution + 0.5)
        pixel_size = int(pixel_radius * 2) + 1
        kernel = np.zeros((pixel_size, pixel_size), np.uint8)
        kernel = cv2.circle(kernel, (pixel_radius, pixel_radius), pixel_radius, 1, thickness=-1)
        return kernel

    def dilate_obstacle(self, radius: float) -> None:
        """Dilate obstacle, without any changes to the border between floor and unknown
        """
        _, thresh = cv2.threshold(self._img, 127, 255, cv2.THRESH_BINARY)
        kernel = self._create_kernel(radius)
        obstacle = cv2.erode(thresh, kernel, iterations=1)
        self._img = cv2.bitwise_and(self._img, self._img, mask=obstacle)

    def erode_unknown(self, radius: float) -> None:
        """Erode unknown, without any changes to the border between floor and obstacles
        See test codes for examples.
        """
        pad_width = int(2 * radius / self._resolution)
        extended_img = np.pad(self._img, pad_width, 'edge')

        # filter the floor
        # remove small voids (smaller than the specified radius) by applying dilate and erode
        # this filtering does not change the border between floor and unknown/obstacle
        kernel = self._create_kernel(radius)
        _, floor_mask = cv2.threshold(extended_img, OccupancyGrid.UNKNOWN_CODE + 1, 255, cv2.THRESH_BINARY)
        filtered_floor_mask = cv2.dilate(floor_mask, kernel, iterations=1)
        filtered_floor_mask = cv2.erode(filtered_floor_mask, kernel, iterations=1)
        filtered_floor = np.where(filtered_floor_mask == 255,
                                  OccupancyGrid.FREE_CODE,
                                  OccupancyGrid.UNKNOWN_CODE).astype(np.uint8)

        # extract obstacle
        obstacle = np.where(extended_img == OccupancyGrid.OCCUPIED_CODE,
                            OccupancyGrid.OCCUPIED_CODE,
                            OccupancyGrid.UNKNOWN_CODE).astype(np.uint8)

        # combine the filtered floor and obstacle, while prioritizing the obstacle
        result = np.where(obstacle == OccupancyGrid.OCCUPIED_CODE,
                          OccupancyGrid.OCCUPIED_CODE,
                          filtered_floor).astype(np.uint8)

        self._img = result[pad_width:-pad_width, pad_width:-pad_width]

    def apply_closing_unknown(self, point_of_interst: np.ndarray, radius: float) -> bool:
        """Apply closing operation only within the region of interest (ROI)
        which contains the given point, while preserving the isolation of the floor areas.

        :param point_of_interst: it's used to determine the region of interest (ROI)
        :param radius: radius of the closing operation [m]
        :return: True if the operation is successful, False if it fails
        """
        poi = self.get_nearest_free_cell(point_of_interst)
        if poi is None:
            return False
        poi_idx = self.realcoords_to_npcoords(poi)

        # pad the image to avoid the border effect
        pad_width = int(2 * radius / self._resolution) + 1
        extended_img = np.pad(self.img, pad_width, 'constant', constant_values=OccupancyGrid.UNKNOWN_CODE)
        poi_idx += pad_width

        # create masks
        free_mask = (extended_img == OccupancyGrid.FREE_CODE)
        occupied_mask = (extended_img == OccupancyGrid.OCCUPIED_CODE)

        # extract the ROI that contains the start point
        _, labeled_free_areas = cv2.connectedComponents(free_mask.astype(np.uint8))
        target_label = labeled_free_areas[poi_idx[0], poi_idx[1]]
        # no free area to close (this should not happen...?)
        assert target_label != 0
        roi_mask = (labeled_free_areas == target_label)

        # apply closing operation only within the ROI
        kernel = self._create_kernel(radius)
        closed_roi_mask = cv2.morphologyEx(roi_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # recover the surrounding shape of the ROI, to avoid merging the isolated floor areas
        _, labeled_unknown_areas = cv2.connectedComponents((~(roi_mask | occupied_mask)).astype(np.uint8))
        # (0, 0) is always UNKNOWN_CODE, because of the padding
        background_label = labeled_unknown_areas[0, 0]
        closed_roi_mask[labeled_unknown_areas == background_label] = 0

        # combine the filtered free/unknown and occupied, while keeping the original occupied pixels
        filtered_img = np.where(closed_roi_mask | free_mask,
                                OccupancyGrid.FREE_CODE,
                                OccupancyGrid.UNKNOWN_CODE).astype(np.uint8)
        filtered_img = np.where(occupied_mask,
                                OccupancyGrid.OCCUPIED_CODE,
                                filtered_img).astype(np.uint8)
        # crop the image to the original size
        self._img = filtered_img[pad_width:-pad_width, pad_width:-pad_width]
        return True

    def fill_circle(self, center: np.ndarray, radius: float, color: int) -> None:
        center_point = self.realcoords_to_cvcoords(center)
        self._img = cv2.circle(self._img, tuple(center_point.tolist()),
                               int(radius / self._resolution + 0.5), color, -1)

    def filter(self, radius: float = 0.2) -> None:
        _, thresh = cv2.threshold(self._img, 127, 255, cv2.THRESH_BINARY)
        kernel = self._create_kernel(radius)
        # in order not to erase narrow walls about 1px width,
        # erode first, then dilate.
        obstacle = cv2.erode(thresh, kernel, iterations=1)
        obstacle = cv2.dilate(obstacle, kernel, iterations=1)

        self._img[np.where(self._img == OccupancyGrid.OCCUPIED_CODE)] = OccupancyGrid.FREE_CODE
        self._img = cv2.bitwise_and(self._img, self._img, mask=obstacle)

    def get_nearest_free_cell(self, start: np.ndarray) -> Optional[np.ndarray]:
        start_pos = self.realcoords_to_npcoords(start)
        logger.debug('start_pos: {}'.format(start_pos))
        index_of_the_value = np.asarray(np.where(self._img == OccupancyGrid.FREE_CODE))
        if index_of_the_value.shape[1] == 0:
            return
        distances = (index_of_the_value[0, :] - start_pos[0]) ** 2 + \
                    (index_of_the_value[1, :] - start_pos[1]) ** 2
        nearest_index = index_of_the_value[:, np.argmin(distances)]
        position = self.npcoords_to_realcoords(nearest_index)
        return position
