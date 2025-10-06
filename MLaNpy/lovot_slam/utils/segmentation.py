"""マップのセグメンテーション

Overview
    Section
    - ObstacleCostMap が no cost である場所に内接円を当てて Section をつくる。
    - Section の円周上それぞれから、ObstacleCostMap の示す通過コストに従って
      Dijkstra-likeにマップ上を埋める。全てのセルはSection円周への累計到達が最低となるSectionに属する。

    Hub
    - Sectionの中心にHubを置く。
    - Sectionから閾値以下のコスト(壁を縦断しない程度)で到達できる区域(=到達可能エリア)が
      他のSectionの到達可能エリアと隣接している場合、その境界線の中央にHubを置く。
      (境界のHubの隣には、隣接するSectionに属する別のHubが置かれて対になる)

    cost_from_hubs
    - 各Hubから、そのHubの属するSectionの到達可能エリアについて、到達コストが事前に計算される。
      これによって、到達可能エリア内の任意の点から、そのSectionに属する各Hubへの経路は高速で計画できる。

    Highway
    - 同じSectionに属するHubの間の経路は事前に計算され、Highwayになる。
    - 対になるHubに対しても、2点のみからなる経路がHighwayとして置かれる。

    precomputed-path
    - Hubをノード、Highwayをそのコストを重みとするエッジとしたグラフについて、
      各Hub間を最短で繋ぐHighwayのパスが事前に計算される。
      start/goal Hubの対に対して、最初に使用すべきHighwayと、goalへの最短到達コストが記憶される。

classes:
    - ObstacelCostMap : step cost on each point of the map,
                        including identity of the map and unwelcomed area
    - Section : each section, a part of the map divided by segmentation
    - Hub: each of section center and section border
    - Highway: pre-computed path between two Hubs
    - Segmentation : performs segmentation and holds the result

api:
    - rebuild_segmentation() : performs segmentation with given rosmap and unwelcomed area
                               to return Segmentation instance
"""


import base64
import itertools
import json
import time
from bisect import bisect
from collections import defaultdict
from contextlib import contextmanager
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree as KDTree

from lovot_map.rosmap import RosMap, create_kernel
from .unwelcomed_area import Polygon, calc_unwelcomed_area_hash, rasterize_unwelcomed_area

logger = getLogger(__name__)


SEGMENTATION_VERSION = 3  # セグメンテーションのjsonフォーマットのバージョン

NO_OBSTACLE = 0
UNKNOWN = 1
OBSTACLE = 2


@contextmanager
def _timeit(label):
    logger.debug(f"task '{label}' begins")
    time_start = time.time()
    yield
    elapsed = time.time() - time_start
    logger.debug(f"task '{label}' ended in {elapsed:.3} s")


def _b64encode_numpy_array(array):
    return base64.b64encode(array.reshape(-1)).decode('ascii') if array is not None else None


def _trace_to_origin(total_cost_map, start_point, mask=None):
    """ある地点から、total_cost_mapを減少方向にたどる。経路の点列とコストのタプルを返す"""
    trace = []
    current_point = start_point
    while True:
        trace.append(current_point)
        current_cost = total_cost_map[current_point]
        if current_cost == 0:
            break
        y, x = current_point
        best_score = current_cost
        best_next_point = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                next_point = (y + dy), (x + dx)
                if mask is not None and next_point not in mask:
                    continue
                score = total_cost_map[next_point]
                if score < best_score:
                    best_score = score
                    best_next_point = next_point
        assert best_next_point
        current_point = best_next_point
    return trace, total_cost_map[start_point]


class ObstacleCostMap:
    """マップ上の各地点の移動コスト

    セグメンテーション計算時の材料であるとともに、
    セグメンテーションの結果と、マップと来ないでエリア設定の整合性を担保するために
    Segmentationの情報に含まれる。
    """

    def __init__(self, map_id: str, origin_pos, origin_yaw, resolution, data: np.ndarray):
        # マップ情報
        self.map_id = map_id
        self.origin_pos = tuple(float(value) for value in origin_pos)
        self.origin_yaw = float(origin_yaw)
        self.resolution = float(resolution)

        # コストマップ
        # 1.0以下は走行可能、1.0より大きいと通行不可
        self.data = data

        # コストマップにマージされた来ないでエリアの情報
        # NOTE: neodm側でロードされた来ないでエリア及びセグメンテーションの情報の整合性の確認に使われる。
        #       nest-slamでは、クラウドからの来ないでエリアの更新と共にセグメンテーションも再計算されるが、
        #       lovot側では来ないでエリアのみがnestからリロードされ、(マップ及び)セグメンテーションが
        #       更新されていない、という状況が起こり得るため。
        # TODO: マップ・セグメンテーションと来ないでエリアのlovot側での更新タイミングが同期することが望ましい。
        self.unwelcomed_area_hash = calc_unwelcomed_area_hash(None)

    @classmethod
    def from_rosmap(cls, rosmap, robot_radius):
        """boundary cost weight を 1.0 とした neodm の obstacle_cost_map に対応するデータを計算"""
        map_data = np.array(rosmap.data, np.uint8).reshape(rosmap.height, rosmap.width)

        # Build cost map
        #   by adding unknown, obstacles and boundary of obstacles with some weight each.
        # separate (unknown) / (floor + obstacles)
        unknown_map = np.where(map_data == OBSTACLE, NO_OBSTACLE, map_data)
        map_ignoring_unknown = np.where(map_data == UNKNOWN, 0, map_data)

        # dilate obstacles area considering robot radius
        structure = create_kernel(robot_radius=robot_radius, map_resolution=rosmap.resolution)
        obstacle_map = ndimage.binary_dilation(map_ignoring_unknown, structure=structure).astype(np.float64)

        # add some costs to the floor around obstacles
        sigma = robot_radius / rosmap.resolution
        boundary_map = ndimage.gaussian_filter(obstacle_map, sigma)

        # obstacles area           ( > 1) : collision (prohibited to enter)
        # boundary or unkonwn area (<= 1) : has some cost to enter
        # no obstacles area        (== 0) : free to enter (floor)
        cost_map = obstacle_map + boundary_map + unknown_map

        # map boundary
        cost_map[:2, :] = 2
        cost_map[-2:, :] = 2
        cost_map[:, :2] = 2
        cost_map[:, -2:] = 2

        return cls(rosmap.name, rosmap.origin_pos_2d, rosmap.origin_yaw, rosmap.resolution, cost_map)

    def merge_unwelcomed_area(
            self, unwelcomed_area: List['Polygon'], conversion_matrix: np.array, robot_radius):
        """projects unwelcomed_area onto the obstacle costmap with boundary_cost_weight.

        - conversion_matrix is an array of shape (2, 3) which converts
            image-pose coordinates to map-image coordinates
        """
        unwelcomed = rasterize_unwelcomed_area(
            unwelcomed_area, self.data.shape, conversion_matrix).astype(np.float64)
        sigma = robot_radius / (self.resolution * 3)
        unwelcomed_gaussian = ndimage.gaussian_filter(unwelcomed, sigma)
        np.maximum(unwelcomed, unwelcomed_gaussian, out=unwelcomed)
        np.maximum(self.data, unwelcomed, out=self.data)
        self.unwelcomed_area_hash = calc_unwelcomed_area_hash(unwelcomed_area)


def _calc_obstacleCostMap(
        rosmap: Optional['RosMap'],
        unwelcomed_area: Optional[List['Polygon']],
        *, robot_radius=.15
) -> Optional[ObstacleCostMap]:
    if not rosmap:
        return None
    costmap = ObstacleCostMap.from_rosmap(rosmap, robot_radius)
    if unwelcomed_area:
        origin_inv_2d = rosmap.get_inversion_matrix()
        costmap.merge_unwelcomed_area(unwelcomed_area, origin_inv_2d, robot_radius)
    return costmap


class Section(NamedTuple):
    """セグメンテーションされた各領域の情報

    このクラス自体は中心円の情報と名前のみを持ち、
    実際の領域はSegmentation.sectionsのindexと
    Segmentation.segmentation_code_mapに保持される。
    """
    center: Tuple[int, int]  # indices on costmap and segmentation code map
    center_area_radius: float  # pixels

    def json_serialize(self, origin_x, origin_y, resolution):
        """マップの座標系[m]に変換してserialize"""
        center_x = (self.center_x + .5) * resolution + origin_x
        center_y = (self.center_y + .5) * resolution + origin_y
        center_area_radius = self.center_area_radius * resolution
        return {
            'center_x': float(center_x),
            'center_y': float(center_y),
            'center_area_radius': float(center_area_radius),
        }

    @property
    def center_x(self):
        return self.center[1]

    @property
    def center_y(self):
        return self.center[0]

    def border_distance(self, other):
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        center_distance = (dx * dx + dy * dy) ** .5
        border_distance = center_distance - (self.center_area_radius + other.center_area_radius)
        return border_distance


class Hub(NamedTuple):
    """Sectionの中心や境界などのマップ上の特徴的な点

    Segmentation内のHubは
    - .hubs: List[Hub]
    - ._sectionwise_hubs: List[List[Hub]]
    に並行して保持される。
    .hubs に入れられた順序は hub_index で保持され、主に Highway の両端のHubの同定や、
    .precomputed_path の開始位置や目標位置のspecifierとして使われる。
    一方Hub自体は、Sectionごとに構成され、また
    各HubからSection内の到達可能点への到達コストを事前計算して .cost_from_hubs に格納するとき、
    .cost_from_hubsのメモリサイズを節約するために別SectionのHubと到達コストのマップを共有しているため
    「subindex := 所属するSectionでの何番目のHubか」の情報を使う。
    これはneodm側でも、経路計画の開始位置の所属するSectionに属するHubを検索するときなどに使われる。
    hub_indexの情報は、serializeされた .hubs の構造が暗示するため、serialization では省略される。
    """

    position: Tuple[int, int]  # indices on costmap and segmentation code map
    section_index: int  # index of belonging section
    subindex: int  # index on segmentation._sectionwise_hubs[section_index]
    hub_index: int  # index on segmentation.hubs  * not for serialization

    def json_serialize(self):
        return {
            'position': tuple(map(float, self.position)),
            'section_index': int(self.section_index),
            'subindex': int(self.subindex),
        }


class Highway(NamedTuple):
    """相異なる2つのHubを結ぶ点列"""
    start_hub_index: int  # points[0]の点にあるHubのindex
    end_hub_index: int  # points[-1]の点にあるHubのindex
    points: Tuple[Tuple[int, int], ...]
    cost: float

    def json_serialize(self):
        return {
            'start_hub_index': int(self.start_hub_index),
            'end_hub_index': int(self.end_hub_index),
            'points': tuple(tuple(map(float, p)) for p in self.points),
            'cost': float(self.cost),
        }


class Segmentation:
    """マップのセグメンテーション

    アルゴリズム概略:
    1. マップ上の各点(以下「ノード」)から、それぞれ最も近い障害物までの距離を計算
    2. それに基づき、マップ上に、壁に触れずに配置できる最大の円を配置し、Sectionとする。
       ただし、2個目以降は半径が.CENTER_AREA_RADIUS_MIN以上で、
       かつ既に配置された円から.CENTER_AREA_BORDER_MARGIN以上の隙間ができなければならない。
    3. それぞれの円の外縁から探索を行い、円外の各ノードを、そこからの到達コストが最も近いSectionに所属させる。
    4. 到達不能なノードに対しては、外縁への直線距離が最も近いSectionに所属させる。
    """

    CENTER_AREA_RADIUS_MIN = .5 - .185  # 配置する内接円(2個目以降)の最小半径 / m
    CENTER_AREA_BORDER_MARGIN = .2  # 配置する内接円の円周間の最小距離 / m

    # .segmentation_code_map は、_SC_DTYPE に定める符号無し整数で
    # 下位 _SC_INDEX_BITS ビットが .sections のindex、
    # 残りの上位が'辺境度合'(内接円の円周からの累計obstacle cost, 内接円内は0)
    # 辺境度合は、cost_mapの1で距離1進んだ時に _SC_BOUNDARY_SCALE だけ加算される
    # 初期値は _SC_INIT (辺境度合(実質)inf, section index 無効値) に定められる
    # cost_map 1 以上の地点からのステップは _SC_OVER_BOUNDARY_RATIO 倍されることで、
    # cost_map 1 未満の地点をたどる方が優先的に扱われる
    # _SC_AFTER_BOUNDARY は、マップ上辿り着けないはずの場所にassignされるsegmentation code の下限値
    _SC_INDEX_BITS = 7
    _SC_INDEX_MASK = (1 << _SC_INDEX_BITS) - 1
    _SC_BOUNDARY_SCALE = 8
    _SC_MAX_SECTIONS = 100
    _SC_COST_OFFSET = 2
    _SC_INIT = 0x8000_0000 | _SC_INDEX_MASK
    _SC_DTYPE = np.uint32
    _SC_OVER_BOUNDARY_RATIO = 256
    _SC_AFTER_BOUNDARY = (_SC_BOUNDARY_SCALE << _SC_INDEX_BITS) * _SC_OVER_BOUNDARY_RATIO
    _SECTION_INDEX_DTYPE = np.int8
    assert 1 << _SC_INDEX_BITS > _SC_MAX_SECTIONS
    assert _SC_OVER_BOUNDARY_RATIO * _SC_BOUNDARY_SCALE * 2000 * 2 ** _SC_INDEX_BITS < _SC_INIT
    assert _SC_INDEX_MASK <= np.iinfo(_SECTION_INDEX_DTYPE).max

    _MAX_HUBS_PER_SECTION = 31  # including the center

    @classmethod
    def _sc_encode_center_area(cls, section_index):
        return section_index

    def __init__(self, costmap: ObstacleCostMap):
        self.costmap = costmap
        self.sections: Iterable[Section] = ()
        self.segmentation_code_map: Optional[np.ndarray] = None  # Segmentation._SC_DTYPE array
        self.hubs: Iterable[Hub] = []
        self._sectionwise_hubs: Iterable[Iterable[Hub]] = []  # not for serialize
        self._major_hub_indices: Iterable[int] = []  # not for serialize
        # cost_from_hubs: array of shape (max(Hub.sub_index)+1, *map_size), dtype float
        #     cost_from_hubs[*point, sub_index] は、あるsection内の
        #     (section_index, sub_index)のHubからpointへ行くための累計コスト。
        #     unreachableな場所や、そのセクションにHubの存在しないsub_indexに対してはNaN
        self.cost_from_hubs: Optional[np.ndarray] = None
        self.highways: Iterable[Highway] = []
        # precomputed_paths: array of shape (len(hubs), len(hubs)), dtype _PRECOMPUTED_PATHS_DTYPE
        #     precomputed_paths[from, to] は、hubs[from]からhubs[to]へ行く時に
        #     次に向かうべきhighwayのindexと、hub[to]までのcostのタプル
        #     - route がないときはvalueがない ([x, x] もない)
        #     - positive n: highways[n] を順にたどる
        #     - negative n: highways[~n] を逆にたどる
        self.precomputed_paths: Dict[Tuple[int, int], Tuple[int, float]] = {}

    def encode(self):
        NO_SECTION = -1

        section_index = (self.segmentation_code_map & self._SC_INDEX_MASK).astype(self._SECTION_INDEX_DTYPE)
        no_section = section_index >= len(self.sections)
        section_index[no_section] = NO_SECTION
        distance_from_center_area = (self.segmentation_code_map >> self._SC_INDEX_BITS) / \
            (self.costmap.resolution / self._SC_BOUNDARY_SCALE)
        distance_from_center_area[no_section] = np.nan

        costmap = self.costmap
        unwelcomed_area_hash = costmap.unwelcomed_area_hash
        j = {
            'version': SEGMENTATION_VERSION,
            'map_id': costmap.map_id,
            'origin_pos': costmap.origin_pos,
            'resolution': costmap.resolution,
            'unwelcomed_area_hash': unwelcomed_area_hash,
            'prebuilt_obstacle_cost': _b64encode_numpy_array(costmap.data.astype(np.float32)),
            'prebuilt_obstacle_cost_dtype': 'float32',
            'sections': [section.json_serialize(*costmap.origin_pos, costmap.resolution) for section in self.sections],
            'section_index': _b64encode_numpy_array(section_index),
            'section_index_dtype': self._SECTION_INDEX_DTYPE.__name__,
            'hubs': tuple(map(Hub.json_serialize, self.hubs)),
            'cost_from_hubs': _b64encode_numpy_array(self.cost_from_hubs.astype(np.float32)),
            'cost_from_hubs_dtype': 'float32',
            'highways': tuple(map(Highway.json_serialize, self.highways)),
            'precomputed_paths': tuple(self.precomputed_paths.items()),
            'width': costmap.data.shape[1],
            'height': costmap.data.shape[0],
        }
        return json.dumps(j)

    def image_preview(
            self, rosmap: Optional['RosMap'], unwelcomed_area: Optional[List['Polygon']],
            outpath: Path, from_hub_path_getter: Callable[[int], Path]
    ):
        logger.info("segmentation image preview begins")
        if not rosmap:
            return
        # base 2d-map
        map_data = np.array(rosmap.data, np.uint8).reshape(rosmap.height, rosmap.width)
        background = np.full((*map_data.shape, 1), 128, np.uint8)
        background[map_data == OBSTACLE] = 0
        background[map_data == NO_OBSTACLE] = 255
        from_hub_images = []

        # darken unwelcomed area
        if unwelcomed_area:
            origin_inv_2d = rosmap.get_inversion_matrix()
            unwelcomed = rasterize_unwelcomed_area(unwelcomed_area, map_data.shape, origin_inv_2d)
            background[unwelcomed] >>= 2

        if self.sections:
            # segmentation layer
            score_control = [
                0, 1 << self._SC_INDEX_BITS, 100 << self._SC_INDEX_BITS,
                self._SC_AFTER_BOUNDARY, self._SC_AFTER_BOUNDARY * 100, self._SC_AFTER_BOUNDARY * 200,
                self._SC_INIT,
            ]
            alp_control = np.array([200, 150, 80, 80, 40, 40, 100]) / 255

            mask = (1 << self._SC_INDEX_BITS) - 1
            score = self.segmentation_code_map & ~mask

            hsv = np.empty((*score.shape, 3), np.float64)
            hsv[..., 0] = (self.segmentation_code_map & mask) * (.6 / len(self.sections))
            hsv[..., 1] = np.where(score < self._SC_AFTER_BOUNDARY, 1.0, 0.7)
            hsv[..., 2] = np.where(score, .75, .9)
            hsv *= 255
            bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(float)

            # alpha composition
            alpha = np.interp(score, score_control, alp_control)
            img = (background + (bgr - background) * alpha[..., np.newaxis]).astype(np.uint8)
            # highways
            highway_layer = np.zeros(img.shape[:2], bool)
            for highway in self.highways:
                for point in highway.points[1:-1]:
                    highway_layer[point] = True
            img[highway_layer] ^= 255

            # from-hub cost maps
            for costmap in self.cost_from_hubs:
                hsv = np.zeros((*costmap.shape, 3), np.float64)
                f = np.isfinite(costmap)
                hsv[f, 0] = costmap[f] * (.6 / costmap[f].max())
                hsv[f, 1:] = .9
                hsv *= 255
                bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(float)
                from_hub_image = (background + (bgr - background) * .8).astype(np.uint8)
                from_hub_image[highway_layer] ^= 255
                from_hub_images.append(from_hub_image)
        else:
            img = np.tile(background, 3)

        cv2.imwrite(str(outpath), img[::-1])
        logger.info(f"wrote segmentation image to '{outpath}'")
        for index, from_hub_image in enumerate(from_hub_images):
            path = from_hub_path_getter(index)
            cv2.imwrite(str(path), from_hub_image[::-1])
            logger.info(f"wrote '{path}'")

    def run(self):
        if not self.costmap:
            logger.debug("Segmentation: no map, no segmentation")
            return

        height, width = self.costmap.data.shape
        resolution = self.costmap.resolution

        # build kdtree of unreachable points
        all_points = np.mgrid[:height, :width].transpose(1, 2, 0)
        unreachable = self.costmap.data >= 1
        obstacles = all_points[unreachable]
        assert len(obstacles)

        # calculate broadness of free points
        free_points = all_points[~unreachable]
        broadness = KDTree(obstacles).query(free_points, k=1)[0]

        # enumerate centers
        segmentation_code_map = np.full(self.costmap.data.shape, self._SC_INIT, self._SC_DTYPE)
        sections = []
        min_broadness = self.CENTER_AREA_RADIUS_MIN / resolution
        center_border_margin = self.CENTER_AREA_BORDER_MARGIN / resolution
        indices = np.argsort(-broadness)
        # open_nodes は、dijkstra-likeに各sectionの領土を決めるための探索起点のリスト
        # node_scores は、open_nodesの各ノードに対応した、内接円からの累計コストと
        #     所属sectionのindexが、segmentationにエンコードされた状態で
        #     昇順ソートされたリスト。node_scoresを二分探索することで、
        #     node_scoresのソートを保ちながらopen_nodesとnode_scoresが更新される
        open_nodes = []
        node_scores = []

        for index in indices:
            b = broadness[index]
            if sections and b < min_broadness:
                break
            point = free_points[index]

            section_index = len(sections)
            new_section = Section(tuple(point), b)
            if any(new_section.border_distance(existing_section) < center_border_margin
                    for existing_section in sections):
                continue

            sections.append(new_section)
            d = (all_points - point) ** 2
            center_area_points = all_points[d[..., 0] + d[..., 1] <= b ** 2]
            center_area_code = self._sc_encode_center_area(section_index)
            segmentation_code_map[center_area_points[:, 0], center_area_points[:, 1]] = center_area_code
            # 内接円内のノードを追加
            # 最初は累計コスト0なのでそのまま右に追加することでsection indexに昇順ソートされた状態になる
            open_nodes += center_area_points.tolist()
            node_scores += [center_area_code] * len(center_area_points)

            if len(sections) == self._SC_MAX_SECTIONS:
                break

        if not sections:
            logger.info("Segmentation: no section")
            self.segmentation_code_map = segmentation_code_map
            return

        # fill free points walking from centers
        outer_nodes = []  # 各sectionの境界、壁や来ないでエリアを乗り越えないと到達できないノード
        edge_cost = self.costmap.data * self._SC_BOUNDARY_SCALE
        edge_cost[unreachable] *= self._SC_OVER_BOUNDARY_RATIO
        edge_cost += self._SC_COST_OFFSET
        regular_cost = np.ceil(edge_cost).astype(self._SC_DTYPE)
        regular_cost <<= self._SC_INDEX_BITS
        diagonal_cost = np.ceil(edge_cost * 2 ** .5).astype(self._SC_DTYPE)
        diagonal_cost <<= self._SC_INDEX_BITS

        while open_nodes:
            y, x = open_nodes.pop(0)
            score = node_scores.pop(0)
            if segmentation_code_map[y, x] < score:
                continue
            regular_score = score + regular_cost[y, x]
            diagonal_score = score + diagonal_cost[y, x]
            for nx, ny, new_score in (
                    (x - 1, y, regular_score), (x + 1, y, regular_score),
                    (x, y - 1, regular_score), (x, y + 1, regular_score),
                    (x - 1, y - 1, diagonal_score), (x + 1, y + 1, diagonal_score),
                    (x + 1, y - 1, diagonal_score), (x - 1, y + 1, diagonal_score)):
                if segmentation_code_map[ny, nx] > new_score:
                    # 初めてorより近い経路で到達できた隣接ノード
                    segmentation_code_map[ny, nx] = new_score
                    if new_score < self._SC_AFTER_BOUNDARY:
                        # 到達可能な地点の場合はopen_nodesに追加
                        idx = bisect(node_scores, new_score)
                        open_nodes.insert(idx, (ny, nx))
                        node_scores.insert(idx, new_score)
                    else:
                        # 到達不能な地点の場合はouter_nodesに登録
                        outer_nodes.append((ny, nx))

        # build kdtree of open points (boundary points of all the sections)
        unassigned = segmentation_code_map == self._SC_INIT
        unassigned_points = all_points[unassigned]
        if len(unassigned_points):
            # fill unreachable points
            # outer_nodesのKD-treeをつくり、
            # すべての未開の地点に対して、それぞれの地点から最も近いouter_nodeを検索し、
            # そこからの直線距離に暫定コストBOUNDARY*2.5を乗じたものをouter_nodeのコストに加算
            outer_nodes = np.array(outer_nodes, np.int32)
            distances, indices = KDTree(outer_nodes).query(unassigned_points, k=1)
            heuristic_costs = (
                distances * (self._SC_BOUNDARY_SCALE * 2.5 * self._SC_OVER_BOUNDARY_RATIO)
            ).astype(self._SC_DTYPE)
            heuristic_costs <<= self._SC_INDEX_BITS
            segmentation_code_map[unassigned_points[:, 0], unassigned_points[:, 1]] = \
                segmentation_code_map[outer_nodes[indices, 0], outer_nodes[indices, 1]] + heuristic_costs

        self.sections = sections
        self.segmentation_code_map = segmentation_code_map

    def _calc_edge_costs(self):
        edge_cost = self.costmap.data * self._SC_BOUNDARY_SCALE
        edge_cost[self.costmap.data >= 1] *= self._SC_OVER_BOUNDARY_RATIO
        edge_cost += 1  # obstacle_costが0のときに短い経路を優先する為
        regular_cost = edge_cost.astype(self._SC_DTYPE)
        regular_cost <<= self._SC_INDEX_BITS
        edge_cost *= 2 ** .5
        diagonal_cost = edge_cost.astype(self._SC_DTYPE)
        diagonal_cost <<= self._SC_INDEX_BITS
        return regular_cost, diagonal_cost

    def build_hubs(self):
        """Sectionの境界をトレースしてhubを構築"""
        assert self.sections
        assert not (self.hubs or self._sectionwise_hubs or self._major_hub_indices)

        # 各Sectionの中心にHubを置く
        for index, section in enumerate(self.sections):
            center_hub = Hub(section.center, index, 0, index)
            self.hubs.append(center_hub)
            self._sectionwise_hubs.append([center_hub])

        # 各Sectionの到達可能エリアの境界をトレースし、外側の状態に応じて境界線の情報に応じてHubを置く
        for section_index, section_hubs in enumerate(self._sectionwise_hubs[:-1]):
            continuous_borders = []  # 境界線:=共通の隣接Sectionのみに接している連続した境界点 のリスト
            trace = ()  # トレース中の境界線
            current_neighbour = None  # トレース中の隣接Sectionのindex
            for border_point, neighbour_point, neighbour_section_index in \
                    self._trace_section_border(section_index):
                # 自分より小さいindexのsectionとの境界は、既に逆側から検索されているので除外
                if neighbour_section_index is not None and neighbour_section_index < section_index:
                    neighbour_section_index = None

                # 国境の状態の変化をチェック
                if current_neighbour != neighbour_section_index:
                    # トレースしていた国境を continuous_borders に追加
                    if current_neighbour is not None:
                        continuous_borders.append((current_neighbour, trace))
                    # トレースの情報を更新
                    trace = [(border_point, neighbour_point)] \
                        if neighbour_section_index is not None else ()
                    current_neighbour = neighbour_section_index
                elif current_neighbour is not None:
                    # トレース中の国境がつづいている場合は、現在地点をトレースに追加
                    trace.append((border_point, neighbour_point))
            if (current_neighbour is not None  # 国境トレースがアクティブで１周を終え、
                    and trace[0][0] != trace[-1][0]):  # 包領でない
                if (continuous_borders  # 登録済みの国境がある
                        and continuous_borders[0][0] == current_neighbour
                        and continuous_borders[0][1][0][0] == trace[-1][0]):  # 開始位置に接続
                    continuous_borders[0] = (current_neighbour, trace + continuous_borders[0][1][1:])
                else:  # 通常の追加
                    continuous_borders.append((current_neighbour, trace))

            # 各セクションのHub上限を超えないようにHubを追加
            #     同一Section内のHubの間にはHighwayが引かれるため、
            #     Highwayの数が発散しないようにSectionごとのHubの数には上限(._MAX_HUBS_PER_SECTION)がある。
            #     この上限の範囲内で、境界線の長いものから優先して、境界線の中央にHubを置く。
            #
            #     また、このとき置かれたHubの外側、隣接Section側の境界にもHubを置く。
            #     このHubの対の間には、2点からなるHighwayが置かれる。
            #     このとき、便宜的に走査した境界線側(section_indexの若い方)のHubを major hub、
            #     他方を minor hub と呼び、major hub の hub_index を ._major_hub_indices に持っておき、
            #     Highwayを構築する際に参照する。
            #     .hubs への追加のされ方により、minor hub の hub_index は常に major hub の hub_index + 1 になる。
            continuous_borders.sort(key=lambda border: (-len(border[1]), border[0]))
            for neighbour_section_index, trace in continuous_borders:
                major_subindex = len(section_hubs)
                if major_subindex == self._MAX_HUBS_PER_SECTION:
                    break
                minor_hubs = self._sectionwise_hubs[neighbour_section_index]
                minor_subindex = len(minor_hubs)
                if minor_subindex == self._MAX_HUBS_PER_SECTION:
                    continue
                major_hub_point, minor_hub_point = trace[len(trace) // 2]
                hub_index = len(self.hubs)
                major_hub = Hub(major_hub_point, section_index, major_subindex, hub_index)
                minor_hub = Hub(minor_hub_point, neighbour_section_index, minor_subindex, hub_index + 1)

                self._major_hub_indices.append(hub_index)
                self.hubs += major_hub, minor_hub
                section_hubs.append(major_hub)
                minor_hubs.append(minor_hub)

    def _trace_section_border(self, section_index):
        """指定されたSectionのreachable境界をtraceする

        国境の座標, 接続する隣接地の座標, 隣接地のSection index のタプルをyield
        unreachableや複数のSectionと接している場合は
        国境の座標, None, None をyield
        """
        # 左から、反時計回りの8近傍の座標差分
        #
        # [7] [6] [5]
        # [0]  @  [4]
        # [1] [2] [3]
        directions = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
        index_mask = (1 << self._SC_INDEX_BITS) - 1

        # 最初は左方向にある section の境界を探すので、index=0 にする。
        # ここから左に45°回るとindex=1、右に回るとindex=-1になる。
        # 一周まわったことを検知するためにindexは連続的に変化させる。
        # このためindexは負や8以上になり得るが、directions[index & 7] でマスクして処理する。
        current_direction_index = 0

        # section.center から (0, -1) 方向にある最遠の地点から始める
        section = self.sections[section_index]
        y, x = section.center

        # section内の穴を避けるため、section中心から最も左(min x)にあるsection内の点を探す。
        left_codes = self.segmentation_code_map[y, :x + 1]
        x = np.where(((left_codes & index_mask) == section_index)
                     & (left_codes < self._SC_AFTER_BOUNDARY))[0].min()

        # みつかった境界点から、reachableなsectionの境界を反時計回りにたどる時、
        # この境界点の"前の"点を探す。
        # (0, -1) 方向に他領域があるので、
        # - +(-1, -1) 地点から (1, 1)方向で来る
        # - +(-1, 0) 地点から (1, 0)方向で来る
        # - +(-1, 1) 地点から (1, -1)方向で来る
        # - ...
        # の優先順位となる。
        #
        #  [3]   [2]   [1]
        #     ↘  ↓  ↙︎
        #  外 …  内 ←  [0]
        #     ↗︎  ↑  ↖︎
        # [-3]  [-2]  [-1]
        #
        # prev_direction_indexの候補は、3 を最優先とし、以下 1 ずつ少ない値を探す。
        for i in range(8):
            prev_direction_index = current_direction_index + 3 - i
            pdy, pdx = directions[prev_direction_index & 7]
            prev_y = y - pdy
            prev_x = x - pdx
            prev_code = self.segmentation_code_map[prev_y, prev_x]
            if prev_code < self._SC_AFTER_BOUNDARY and (prev_code & index_mask) == section_index:
                break
        else:
            # radiusが1未満になるのは特殊な地形のみで、そのときもlen(sections)==1となるため、
            # .build_highways() の冒頭でカットされ通常フローではここにはこない。
            # この分岐はsectionが1点からなる場合のためのフェイルセーフ
            # 開始位置と終了位置を含むために2点以上yieldする必要がある。
            yield (y, x), None, None
            yield (y, x), None, None
            return

        # 一周して (y, x) に戻ってきたら終了
        # Note: 一本道の場合、進行方向によって区別する。隣接するsectionの扱いも、
        #        例えば上方向に向かう場合は右側のみ、下方向に向かう場合は左側のみを
        #        隣接領域として扱う。
        # 反時計回りに一周するとdirection_indexは8増える
        #
        #      [+1] ← [0]=[+8]
        #    ↙︎              ↖︎
        # [+2]               [+7]
        #  ↓                 ↑
        # [+3]               [+6]
        #    ↘︎             ↗︎
        #      [+4] → [+5]
        final_state = y, x, (prev_direction_index + 8)
        hist = []  # for fail-safe warning
        while True:
            # 隣接セクションの同一性をチェックしながら次の進行方向を決定
            # neighbour_index:
            #     - -1: 隣接sectionがみつかっていない
            #     - -2: 複数のsection、又はunreachableと隣接している
            #     - 0以上: みつかっている単一の隣接sectionのindex
            neighbour_index = -1
            next_direction_index = None

            # 境界線を反時計回りにトレースするため、右135°への転進を優先し、以下左方向に探索
            #
            #    3rd      2nd      1st
            #     ↖︎       ↑      ↗︎
            #       (-1) (-2) (-3)
            # 4th←  (0)  CUR  .. ← PRV(8th)
            #       (+1) (+2) (+3)
            #     ↙︎       ↓     ↘︎
            #   5th       6th     7th
            for direction_index in range(prev_direction_index - 3, prev_direction_index + 5):
                dy, dx = directions[direction_index & 7]
                next_y = y + dy
                next_x = x + dx
                next_code = self.segmentation_code_map[next_y, next_x]
                if next_code >= self._SC_AFTER_BOUNDARY:
                    neighbour_index = -2
                    continue
                next_section_index = next_code & index_mask
                if next_section_index == section_index:
                    next_direction_index = direction_index
                    break
                if neighbour_index == -1:
                    neighbour_index = next_section_index
                elif neighbour_index != next_section_index:
                    neighbour_index = -2
            else:
                # Sectionが1点しか持たない場合（最初のフェイルセーフで処理されているはず）
                assert False

            # 同一国境を検知した場合はneighbour_pointを計算
            if neighbour_index >= 0:
                neighbour_direction_index = (
                    prev_direction_index - 4 + next_direction_index) // 2
                dy, dx = directions[neighbour_direction_index & 7]
                neighbour_point = y + dy, x + dx
            else:
                neighbour_index = None
                neighbour_point = None

            yield (y, x), neighbour_point, neighbour_index
            if (y, x) == final_state[:2]:
                if prev_direction_index == final_state[2]:
                    break
                # フェイルセーフ
                if ((prev_direction_index ^ final_state[2]) & 7) == 0 and hist:
                    logger.warning("unexpected cycle found: "
                                   f"d {final_state[2] - 8} -> {prev_direction_index} @ {(y, x)}")
                    break
            hist.append((y, x, prev_direction_index))
            if len(hist) == 20000:  # 境界長さが標準resolutionで1kmを超えた
                logger.warning("abandon tracing unexpectedly long border:"
                               f" section[{section_index}] @ {section.center} - {hist[:10]}...")
                break

            prev_direction_index = next_direction_index
            y = next_y
            x = next_x

    def calc_cost_from_hubs(self):
        """cost_from_hubsの計算

        各Sectionについて、そのSectionに属する各Hubから、
        Sectionの到達可能エリア内の各点への到達コストを計算。
        Dijkstra-likeに未到のセルを探索している。

        cost_from_hubsは、所属するSectionごとにつけられたサブインデックス(Hub.subindex)ごとに
        マップを共有する。
        例えば Section[0] の 2 番目のHubからの到達コストと、Section[1] の 2 番目のHubからの到達コストは、
        同じ cost_form_hubs[2, :, :] の2d-map上に記録される。
        到達可能エリア外や、そのsubindexを持つHubが存在しないSectionに対しては NaN になる。
        """
        max_subindex = max(map(len, self._sectionwise_hubs))
        self.cost_from_hubs = np.full((max_subindex, *self.segmentation_code_map.shape), np.nan)

        regular_cost = self.costmap.data.astype(np.float64)
        regular_cost += self._SC_COST_OFFSET / self._SC_BOUNDARY_SCALE

        for section_index, section_hubs in enumerate(self._sectionwise_hubs):
            outer_positions = set(zip(*np.where(
                ((self.segmentation_code_map & self._SC_INDEX_MASK) != section_index)
                | (self.segmentation_code_map >= self._SC_AFTER_BOUNDARY))))
            for subindex, hub in enumerate(section_hubs):
                costs = self.cost_from_hubs[subindex]
                open_nodes = [hub.position]
                node_scores = [0]
                done = outer_positions.copy()

                while open_nodes:
                    y, x = open_nodes.pop(0)
                    score = node_scores.pop(0)
                    done.add((y, x))
                    if costs[y, x] <= score:
                        continue
                    assert (y, x) in done
                    costs[y, x] = score
                    step_cost = regular_cost[y, x]
                    regular_score = score + step_cost
                    diagonal_score = score + step_cost * 2 ** .5
                    for nx, ny, new_score in (
                            (x - 1, y, regular_score), (x + 1, y, regular_score),
                            (x, y - 1, regular_score), (x, y + 1, regular_score),
                            (x - 1, y - 1, diagonal_score), (x + 1, y + 1, diagonal_score),
                            (x + 1, y - 1, diagonal_score), (x - 1, y + 1, diagonal_score)):
                        if (ny, nx) in done:
                            continue
                        assert np.isnan(costs[ny, nx])
                        node_index = bisect(node_scores, new_score)
                        open_nodes.insert(node_index, (ny, nx))
                        node_scores.insert(node_index, new_score)

    def build_highways(self):
        """Highwayの構築

        - 同じSectionに属する2つのHub
        - Section境界で対になっているHub
        どうしの間の経路を事前計画し、Highwayを登録する。
        経路はend側のHubからのcost_from_hubsの最小をトレースすることで計算される。
        """
        assert not self.highways

        # Section境界で対になっているHubを2点のみからなるHighwayでつなぐ
        for major_hub_index in self._major_hub_indices:
            major_hub = self.hubs[major_hub_index]
            start_position = major_hub.position
            end_position = self.hubs[major_hub_index + 1].position
            cost = float(self.costmap.data[start_position] + self.costmap.data[end_position]) / 2 \
                + self._SC_COST_OFFSET / self._SC_BOUNDARY_SCALE
            self.highways.append(
                Highway(major_hub_index, major_hub_index + 1, (start_position, end_position), cost))
        # Section内のHubをHighwayでつなぐ
        for section_index, section_hubs in enumerate(self._sectionwise_hubs):
            walkable_points = set(zip(*np.where(
                ((self.segmentation_code_map & self._SC_INDEX_MASK) == section_index)
                & (self.segmentation_code_map < self._SC_AFTER_BOUNDARY))))
            for start_hub, end_hub in itertools.combinations(section_hubs, 2):
                points, cost = _trace_to_origin(
                    self.cost_from_hubs[end_hub.subindex], start_hub.position, walkable_points)
                if points[-1] != end_hub.position:
                    logger.warning(f"skipped malformed highway: {start_hub.position} -> {end_hub.position}")
                self.highways.append(
                    Highway(start_hub.hub_index, end_hub.hub_index, points, cost))

    def build_precomputed_paths(self):
        """各hub間を移動するルートを構築

        Hubをノード、Highwayをそのコストを重みとするエッジとしたグラフについて、
        各Hub間を最短で繋ぐHighwayのパスが事前に計算される。
        start/goal Hubの対に対して、最初に使用すべきHighwayと、goalへの最短到達コストが記憶される。

        各 start Hub に対して、Dijkstra-likeに探索を行う。
        既に計算されたパスの部分パスとなる部分を端折って計算する方が早いが、
        この部分はボトルネックではない(cost_from_hubsの計算が圧倒的に重い)ため、実装の簡明さを優先。
        """
        assert not self.precomputed_paths

        # hub_index -> tuple of
        #    - [0] そこからのhighway_index(逆にたどる場合は ~highway_index)
        #    - [1] 行き先のhub_index
        #    - [2] エッジのコスト
        edges: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
        for highway_index, highway in enumerate(self.highways):
            edges[highway.start_hub_index].append((highway_index, highway.end_hub_index, highway.cost))
            edges[highway.end_hub_index].append((~highway_index, highway.start_hub_index, highway.cost))

        for start_hub_index in range(len(self.hubs)):
            edges_from_start = edges[start_hub_index]
            edges_from_start.sort(key=lambda e: e[2])
            open_paths = [(next_hub_index, highway_code)
                          for (highway_code, next_hub_index, _) in edges_from_start]
            total_costs = [step_cost for (_r, _n, step_cost) in edges_from_start]
            while open_paths:
                current_hub_index, first_highway_code = open_paths.pop(0)
                current_total_cost = total_costs.pop(0)
                key = int(start_hub_index), int(current_hub_index)
                if key in self.precomputed_paths:
                    continue
                self.precomputed_paths[key] = int(first_highway_code), float(current_total_cost)
                for _, next_hub_index, step_cost in edges[current_hub_index]:
                    if next_hub_index == start_hub_index:
                        continue
                    next_total_cost = current_total_cost + step_cost
                    index = bisect(total_costs, next_total_cost)
                    open_paths.insert(index, (next_hub_index, first_highway_code))
                    total_costs.insert(index, next_total_cost)


def rebuild_segmentation(
        rosmap: Optional['RosMap'], unwelcomed_area: Optional[List['Shape']]
) -> Segmentation:
    """2dマップと来ないでエリアから segemntation(含obstacle_cost_map) を行う。

    rosmap が None であればマップがない状態のSegmentationのインスタンスを返す(Noneではない)
    """
    # TODO: sectionの名前を引き継げるようにしたい

    with _timeit('cost map calculation'):
        costmap = _calc_obstacleCostMap(rosmap, unwelcomed_area)

    with _timeit('segmentation'):
        segmentation = Segmentation(costmap)
        segmentation.run()

    if not segmentation.sections:
        logger.warning("no section in building segmentation")
        return segmentation

    with _timeit('hubs'):
        segmentation.build_hubs()

    with _timeit('cost_from_hubs'):
        segmentation.calc_cost_from_hubs()

    with _timeit('highway'):
        segmentation.build_highways()

    with _timeit('precomputed_paths'):
        segmentation.build_precomputed_paths()

    return segmentation


def validate_segmentation_metadata(
        segmentation_dir: Path, *,
        version: int,
        # map_id: str,
        # unwelcomed_area: str,
        include_png: bool,
        **other_metadata) -> Union[Dict[str, Any], str]:
    """validate saved segmentation data with requested metadata.

    :return: metadata of segmentation, or error message in str.
    """
    if other_metadata:
        logger.warning(f"unknown metadata request ignored: {other_metadata}")

    path = segmentation_dir / 'segmentation.json'
    try:
        with path.open() as fin:
            buf = fin.read()
        j_segmentation = json.loads(buf)
    except IOError:
        return f"couldn't load segmentation data: {path}"
    except json.decoder.JSONDecodeError:
        return f"segmentation data broken: {path}"

    loaded_version = j_segmentation.get('version')
    if loaded_version != version:
        return f"segmentation version {loaded_version} != {version}"

    png_exists = (segmentation_dir / 'segmentation.png').is_file()
    if include_png and not png_exists:
        return "required segmentation.png does not exist"

    return {'version': loaded_version, 'include_png': png_exists}
