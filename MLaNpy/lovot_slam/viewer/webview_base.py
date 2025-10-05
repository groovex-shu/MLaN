import base64
import pathlib
from collections import OrderedDict
from enum import Enum
from functools import partial
from http import HTTPStatus
from logging import getLogger
from math import atan2
from typing import Tuple

import h11
import trio
from attr import dataclass
from jinja2 import Environment, FileSystemLoader

from lovot_slam.env import MAP_2DMAP
from lovot_slam.slam_manager import SlamManager
from MLaNpy.lovot_map.rosmap import RosMap
from lovot_slam.viewer import http_server

logger = getLogger(__name__)

MAP_VIZ_SCALE = 2  # 2x expand map image

# 500x500 px is default map image size
# (1px=0.05m, max map size=25m^2)
DEFAULT_MAP_CANVAS_WIDTH = 500
DEFAULT_MAP_CANVAS_HEIGHT = 500

BASE_DIR = pathlib.Path(__file__).parent


class ServerName(Enum):
    lovot = 1
    nest = 2


@dataclass
class PoseOnMap:
    x: int
    y: int
    yaw: float


def transpose_position(position, scale=MAP_VIZ_SCALE):
    return [position[0]*MAP_VIZ_SCALE, position[1]*MAP_VIZ_SCALE]


def quaternion_to_rpy_yaw(q) -> float:
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return atan2(2 * (w * z + x * y),
                 1 - 2 * (y * y + z * z))


def get_drawpose_js(pose: PoseOnMap, name):
    return f'slamview.drawPose({pose.x},{pose.y},{pose.yaw},"{name}");'


def get_drawul_js(unwelcomed_area):
    # vertices is 2d list containing the edges of an unwelcomed_area
    # ex. slamview.drawUnwelcomedArea([[0,0],[1,0],[1,1],[0,1]]);"
    return f"slamview.drawUnwelcomedArea({unwelcomed_area});"


class WebViewServerBase():
    def __init__(self, server_name: str, slam_manager: SlamManager):
        self.server_name = server_name
        self.slam_manager = slam_manager
        self.map_name = ""

        self._static_dir = pathlib.Path(__file__).parent / 'static'

        self._env = Environment(loader=FileSystemLoader(BASE_DIR, encoding='utf8'))

    def update_map_name(self):
        latest_map_name = self.slam_manager.map_utils.get_latest_merged_map()
        if not latest_map_name:
            self.map_name = ""
            return False

        self.map_name = latest_map_name
        return True

    def get_map_image(self):
        map_utils = self.slam_manager.map_utils
        if map_utils.check_map(self.map_name):
            map2d_path = map_utils.get_full_path(self.map_name) / MAP_2DMAP / 'map.yaml'
            rosmap = RosMap.from_map_yaml(self.map_name, map2d_path)
            png = RosMap.to_png(rosmap.data, rosmap.width, rosmap.height)
            return png, rosmap.width, rosmap.height
        else:
            logger.warning(f'unable to load map image for {self.map_name}')
            return None, 0, 0

    def get_map_metadata(self) -> OrderedDict:
        map_config = self.slam_manager.map_utils.get_metadata(self.map_name)
        metadata = OrderedDict()
        metadata['map'] = self.map_name
        metadata['version'] = map_config['lovot_slam']['version']
        metadata['date'] = map_config['lovot_slam']['date']
        metadata['feature_map'] = map_config['lovot_slam']['feature_map']
        metadata['scale_odom_loc'] = map_config['lovot_slam']['scale_odom_loc']
        metadata['source'] = map_config['lovot_slam']['source']
        return metadata

    def get_map_origin(self) -> Tuple[float, float]:
        grid_map = self.get_grid_map()
        if grid_map is None:
            logger.warning('failed to get grid map')
            return 0.0, 0.0

        return tuple(grid_map.origin[:2].tolist())

    def get_grid_map(self):
        try:
            return self.slam_manager.map_utils.get_occupancy_grid(self.map_name)
        except RuntimeError as e:
            logger.warning(e)
            return None

    def get_slam_spot_on_map_img_origin(self, spot_name):
        grid_map = self.get_grid_map()
        if grid_map is None:
            logger.warning('failed to get grid map')
            return None

        # Spot on omni_map
        spot = self.slam_manager.spot_utils.get_spot_coordinate_from_redis(spot_name)
        if spot is None:
            logger.info(f'spot "{spot_name}" is not set yet')
            return None
        logger.debug(f'spot "{spot_name}" (on omni_map): {spot}')

        # position [x, y] on map image (default scale)
        pos_on_map = grid_map.realcoords_to_cvcoords(spot[:2])
        # orientation
        yaw = quaternion_to_rpy_yaw(spot[3:])

        pos_on_map = transpose_position(pos_on_map)
        logger.debug(f'spot "{spot_name}" on map_img_origin: {pos_on_map, yaw}')
        return PoseOnMap(pos_on_map[0], pos_on_map[1], yaw)

    def get_slam_unwelcomed_area(self):
        grid_map = self.get_grid_map()
        if grid_map is None:
            logger.warning('failed to get grid map')
            return None

        # load unwelcomed_area, and
        # transform the edges of each area for map image origin

        # returned area is a list of unwelcomed_area.Polygons
        unwelcomed_area = self.slam_manager.spot_utils.get_unwelcomed_area_from_redis()
        if unwelcomed_area is None:
            logger.info('unwelcomed_area is not set yet')
            return None

        ret = []
        for u in unwelcomed_area:
            # one unwelcomed_area has 4 edges (np.array),
            # should be transposed for each
            transposed_ul = [list(grid_map.realcoords_to_cvcoords(p)) for p in u.vertices]

            # after transposed to the edges for map image origin,
            # then transpose again to fit into the webview scale
            transposed_ul = [transpose_position(p) for p in transposed_ul]
            ret.append(transposed_ul)

        return ret

    async def serve(self, host, http_port, *, task_status=trio.TASK_STATUS_IGNORED):
        """Web view server task."""
        async with trio.open_nursery() as nursery:
            http_serve = partial(http_server.http_serve, request_handler=self.http_request_handler, debug=False)
            logger.info(f'listening on http://{host}:{http_port}')
            await nursery.start(partial(trio.serve_tcp, http_serve, http_port, host=host))
            task_status.started()

    def _get_html_with_map(self, custom_content: str) -> str:
        metadata = self.get_map_metadata()

        png_array, width, height = self.get_map_image()
        base64_png = base64.b64encode(png_array).decode()
        map_canvas_width = MAP_VIZ_SCALE * DEFAULT_MAP_CANVAS_WIDTH
        map_canvas_height = MAP_VIZ_SCALE * DEFAULT_MAP_CANVAS_HEIGHT
        map_width = MAP_VIZ_SCALE * width
        map_height = MAP_VIZ_SCALE * height

        # Tom / Spike specific javascript
        js = self.get_custom_js()

        template = self._env.get_template('html/index.html')
        return template.render({
            'server_name': self.server_name,
            'map_metadata': metadata,
            'png': base64_png,
            'map_width': map_width,
            'map_height': map_height,
            'map_canvas_width': map_canvas_width,
            'map_canvas_height': map_canvas_height,
            'custom_content': custom_content,
            'js': js
        })

    def _get_html_without_map(self, custom_content: str) -> str:
        js = ("let slamview = {};\n"
              "slamview.client = new LocalizationClient(location.hostname);")

        template = self._env.get_template('html/index.html')
        return template.render({
            'server_name': self.server_name,
            'custom_content': custom_content,
            'js': js
        })

    async def http_request_handler(self, server, request: h11.Request):
        logger.debug(f"http_request_handler {request.target}")

        if request.target == b'/':
            # Tom / Spike specific html content
            custom_html_content = self.get_custom_html_content()

            if self.update_map_name():
                content = self._get_html_with_map(custom_html_content)
            else:
                content = self._get_html_without_map(custom_html_content)

            status = HTTPStatus.OK

        elif request.target:
            static_file = self._static_dir / ('.' + request.target.decode())
            if static_file.exists():
                content_type = ("text/javascript; charset=utf-8" if static_file.suffix == 'js' else
                                "text/html; charset=utf-8")

                async with await trio.open_file(static_file) as f:
                    content = await f.read()

                await http_server.send_simple_response(server,
                                                       200,
                                                       content_type,
                                                       content.encode("utf-8"))
                return

            content = "Not Found"
            status = HTTPStatus.NOT_FOUND

        else:
            content = "Not Found"
            status = HTTPStatus.NOT_FOUND

        await http_server.send_simple_response(server,
                                               int(status),
                                               "text/html; charset=utf-8",
                                               content.encode("utf-8"))

    def get_custom_html_content(self):
        raise NotImplementedError()

    def get_custom_js(self):
        raise NotImplementedError()
    
