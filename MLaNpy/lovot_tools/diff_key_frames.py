import argparse
from typing import List

import cv2
import numpy as np

from lovot_slam.feature_map.feature_map_vertices import FeatureMapVertices, VerticesComparator
from grid_map_util.occupancy_grid import OccupancyGrid

RESOLUTION = 0.02
DOT_RADIUS = 0.04

COLOR_RANGE_METER = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description="make image of differences of key frames' pose")
    parser.add_argument("reference", help="reference map path")
    parser.add_argument("target", help="target map path")
    parser.add_argument("--output", "-o", help="output image")
    args = parser.parse_args()
    return args


def _obtain_boundary(vertices: FeatureMapVertices, mergin: int = 0, resolution: float = RESOLUTION) -> np.ndarray:
    right = int((np.max(vertices.vertices[:, 0]) + 2 * mergin) / resolution)
    left = int((np.min(vertices.vertices[:, 0]) - 2 * mergin) / resolution)
    top = int((np.max(vertices.vertices[:, 1]) + 2 * mergin) / resolution)
    bottom = int((np.min(vertices.vertices[:, 1]) - 2 * mergin) / resolution)
    return np.array([right, left, top, bottom])


def _create_grid_map(vertices_list: List[FeatureMapVertices]) -> OccupancyGrid:
    boundaries = np.empty((0, 4))
    for vertices in vertices_list:
        boundary = _obtain_boundary(vertices, mergin=DOT_RADIUS)
        boundaries = np.append(boundaries, boundary[np.newaxis, :], axis=0)
    boundary = np.max(boundaries, axis=0).astype(np.int32)

    width = boundary[0] - boundary[1]
    height = boundary[2] - boundary[3]
    print(f'size: {width} x {height}')
    assert width > 0 and height > 0

    new_img = np.full([height, width], 0, dtype=np.uint8)
    origin = np.array([boundary[1] * RESOLUTION, boundary[3] * RESOLUTION])
    print(f'origin: {origin}')

    return OccupancyGrid(new_img, RESOLUTION, origin)


def run():
    args = parse_args()

    reference = FeatureMapVertices.from_map_path(args.reference)
    target = FeatureMapVertices.from_map_path(args.target)
    assert reference and target

    grid_map = _create_grid_map([reference, target])

    comparator = VerticesComparator(reference, target)

    for mission_id in target.missions:
        if mission_id not in reference.missions:
            print(f"{mission_id} is not in the reference.")
            continue

        # translational errors
        distance_error = comparator.distance_diffs(mission_id)
        distance_ave = np.average(distance_error)
        distance_max = np.max(distance_error)

        # rotational errors (angle)
        angle_error = np.degrees(comparator.angle_diffs(mission_id))
        angle_ave = np.average(angle_error)
        angle_max = np.max(angle_error)

        print(f"{mission_id}: {distance_ave:.3f}, {distance_max:.3f}, {angle_ave:.3f}, {angle_max:.3f}")

        # plot
        vertices = target.vertices_of(mission_id)
        for i in range(vertices.shape[0]):
            gray = int(np.clip(distance_error[i] / COLOR_RANGE_METER * 255, 0, 255))
            grid_map.fill_circle(vertices[i, :2], DOT_RADIUS, gray)

    # save plot
    if args.output:
        img_jet = cv2.applyColorMap(grid_map.img, cv2.COLORMAP_JET)
        cv2.imwrite(args.output, img_jet)


if __name__ == '__main__':
    run()
