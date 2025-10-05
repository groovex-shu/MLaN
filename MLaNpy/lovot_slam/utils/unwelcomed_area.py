from collections import defaultdict
from hashlib import md5
import json
from math import asin, ceil, inf
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import prometheus_client

_unwelcomed_area_metric = prometheus_client.Gauge(
    'localization_unwelcomed_area_total', 'Total size of unwelcomed areas in the map'
)

PI = np.pi
TWO_PI = PI + PI


def _is_point(p):
    return isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)


def _is_point_list(li):
    return isinstance(li, list) and all(_is_point(p) for p in li)


def _segment_pixels(point_from, point_to):
    """linearly interpolates two points with continual pixels

    Note: 'point_to' is not contained.
    """
    steps = ceil(np.abs(point_to - point_from).max())
    return np.linspace(point_from, point_to, steps, endpoint=False).round().astype(np.int32)


def _fit_rectangle_to_vertices(vertices: np.ndarray) -> np.ndarray:
    """takes 4 vertices, fits a rectangle to them, and returns its vertices"""
    assert vertices.shape == (4, 2)
    g = vertices.mean(axis=0)
    rel = vertices - g  # 重心からの位置ベクトル

    # 一旦 対角線の中点が重心に重なるように対頂点を平行移動して
    # 平行四辺形をフィッティング
    a = rel[0] - rel[2]
    b = rel[1] - rel[3]
    # a/2, b/2, -a/2, -b/2 がフィッティングした平行四辺形

    # a と b のなす角の二等分線と、線分 a/2 - b/2 の交点を、フィッティングする矩形の軸とする
    # a と -b に関しても同様。両軸は(浮動小数点数の丸め誤差を除けば)直交する
    len_a = np.linalg.norm(a)
    len_b = np.linalg.norm(b)
    den = len_a + len_b
    if den == 0:
        return vertices  # 元の四角形がつぶれている場合はそのまま返す
    den = 1 / den
    a *= len_b * den
    b *= len_a * den
    fit = np.vstack((a, b, -a, -b))
    fit += g

    return fit


class Polygon:
    """
    Convex Polygon
    """
    def __init__(self, vertices):
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        self._vertices = np.array(vertices)
        # Sort vertices in clockwise order for consistent area calculation
        self._sort_vertices_clockwise()

    @property
    def vertices(self):
        return self._vertices
    
    @property
    def area(self) -> float:
        """Calculate polygon area using the shoelace formula."""
        n = len(self._vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self._vertices[i][0] * self._vertices[j][1]
            area -= self._vertices[j][0] * self._vertices[i][1]
        return abs(area) / 2.0
    
    @property
    def center(self) -> np.ndarray:
        """Calculate the centroid (center) of the polygon."""
        return np.mean(self._vertices, axis=0)


    def _sort_vertices_clockwise(self):
        """Sort vertices in clockwise order based on angle from centroid."""
        center = self._vertices.mean(axis=0)
        relative = self._vertices - center
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        # Sort by descending angle for clockwise order
        indexes = (-angles).argsort()
        self._vertices = self._vertices[indexes]

    @classmethod
    def json_deserialize(cls, j):
        vertices = j.get('vertices')
        if not _is_point_list(vertices):
            return None
        return cls(vertices)

    def json_serialize(self) -> str:
        return {'shape': 'polygon', 'vertices': [[float(v) for v in p] for p in self._vertices]}

    def calc_projected_contour_pixels(self, mat):
        # convert vertices by mat
        if mat is None:
            projected_vertices = self._vertices
        else:
            projected_vertices = (np.hstack((self._vertices, np.full((len(self._vertices), 1), 1.0))) @ mat.T).round()
        indices = list(range(len(self._vertices))) + [0]
        # edge as array of pixels, for each edge (of shape (len(vertices), <line segment interp.>, 2)
        edge_pixels_collection = [_segment_pixels(projected_vertices[idx_from], projected_vertices[idx_to])
                 for (idx_from, idx_to) in zip(indices, indices[1:])]
        return np.vstack(edge_pixels_collection)

    def transform(self, transformer, fit_rectangle=False) -> 'Polygon':
        vertices = transformer(self._vertices)
        if fit_rectangle and len(vertices) == 4:
            vertices = _fit_rectangle_to_vertices(vertices)
        return Polygon(vertices)
    


def decode_unwelcomed_area(encoded_str) -> List[Polygon]:
    try:
        j = json.loads(encoded_str)
    except json.decoder.JSONDecodeError:
        return None
    if not isinstance(j, list):
        return None
    polygons = []
    for item in j:
        if isinstance(item, dict) and item.get('shape') == 'polygon':
            polygon = Polygon.json_deserialize(item)
            if polygon:
                polygons.append(polygon)
    return polygons


def encode_unwelcomed_area(unwelcomed_area: Iterable[Polygon]) -> str:
    return json.dumps([shape.json_serialize() for shape in unwelcomed_area])


def rasterize_unwelcomed_area_with_conversion(
        unwelcomed_area: Iterable[Polygon], conversion_matrix: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """rasterize shapes converted by given matrix.

    returns rasterized bool 2d-array in circum-rectangle, and its offsets;
        None if rasterized image is blank.
    """
    # circum-rectangle of rasterized unwelcomed area
    left = inf
    top = inf
    right = -inf
    bottom = -inf

    runs_collection = []  # 'runs'(see below) for each shape
    for shape in unwelcomed_area:
        # since each shape is assumed to be convex,
        # we regard points of the rasterized shape as 'runs'.
        # 'runs' forms a dict x -> (y_min, y_max), meaning that vertical segment slices the shape.
        contour_points = shape.calc_projected_contour_pixels(conversion_matrix)
        runs: Dict[np.int32, List[np.int32]] = defaultdict(lambda: [inf, -inf])
        for x, y in contour_points:
            # update circum-rectangle
            if x < left:
                left = x
            if right < x:
                right = x
            if y < top:
                top = y
            if bottom < y:
                bottom = y

            # update runs
            r = runs[x]  # maltyped default [inf, -inf] will be immediately overwritten to be [y, y]
            if y < r[0]:
                r[0] = y
            if r[1] < y:
                r[1] = y
        runs_collection.append(runs)

    if left == inf:
        return None

    offsets = np.floor(np.array((left, top))).astype(np.int32)
    suprema = np.ceil(np.array((right, bottom))).astype(np.int32)
    size = suprema - offsets + 1

    # rasterized unwelcomed area as boolean array, with offset (left, top)
    left, top = offsets
    raster = np.zeros(size, bool)
    for runs in runs_collection:
        for x, (y_bgn, y_end) in runs.items():
            raster[x - left, y_bgn - top:y_end - top] = True
    return raster, offsets


def rasterize_unwelcomed_area(
        unwelcomed_area: Iterable[Polygon], result_shape: Tuple[int, int], conversion_matrix: np.ndarray
) -> np.ndarray:
    """rasterizes the area into bool 2d-array of given shape"""
    result = np.zeros(result_shape, bool)
    res = rasterize_unwelcomed_area_with_conversion(unwelcomed_area, conversion_matrix)
    if not res:
        return result
    rasterized, offsets = res
    maxima = offsets + rasterized.shape

    # overlapping rectangle of the given canvas (0, 0, *result_shape) and offset raster (*offsets, *maxima)
    overlap_bgn = np.maximum(0, offsets)
    overlap_end = np.minimum(result_shape, maxima)

    if np.all(overlap_bgn < overlap_end):  # overlaps
        src_left, src_top = overlap_bgn - offsets
        src_right, src_bottom = overlap_end - offsets
        result[overlap_bgn[0]:overlap_end[0], overlap_bgn[1]:overlap_end[1]] = \
            rasterized[src_left:src_right, src_top:src_bottom]
    return result


def calc_unwelcomed_area_hash(unwelcomed_area: Iterable[Polygon]) -> str:
    return md5((encode_unwelcomed_area(unwelcomed_area) if unwelcomed_area else '').encode('ascii')).hexdigest()

def calc_unwelcomed_area_hash_from_str(encoded_unwelcomed_area: str) -> str:
    return calc_unwelcomed_area_hash(decode_unwelcomed_area(encoded_unwelcomed_area))


def set_unwelcomed_area_metric(area: float):
    """Set the unwelcomed area metric."""
    _unwelcomed_area_metric.set(area)