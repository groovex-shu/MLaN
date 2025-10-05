import argparse

import cv2
import numpy as np

from lovot_slam.feature_map.feature_map_vertices import FeatureMapVertices
from grid_map_util.occupancy_grid import OccupancyGrid

RESOLUTION = 0.05
DOT_RADIUS = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="show vertices of featreu map of a map")
    parser.add_argument("map", help="target map path")
    parser.add_argument("--output", "-o", help="output image")
    args = parser.parse_args()
    return args


def _obtain_boundary(vertices: FeatureMapVertices, mergin: int = 0, resolution: float = RESOLUTION) -> np.ndarray:
    right = int((np.max(vertices.vertices[:, 0]) + 2 * mergin) / resolution)
    left = int((np.min(vertices.vertices[:, 0]) - 2 * mergin) / resolution)
    top = int((np.max(vertices.vertices[:, 1]) + 2 * mergin) / resolution)
    bottom = int((np.min(vertices.vertices[:, 1]) - 2 * mergin) / resolution)
    return np.array([right, left, top, bottom])


def _create_grid_map(vertices: FeatureMapVertices) -> OccupancyGrid:
    boundary = _obtain_boundary(vertices, mergin=DOT_RADIUS)

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

    target = FeatureMapVertices.from_map_path(args.map)
    assert target

    grid_map = _create_grid_map(target)

    missions_count = len(target.missions)
    print(f'{missions_count=}')
    for count, mission_id in enumerate(target.missions):
        # plot
        vertices = target.vertices_of(mission_id)
        gray = int(np.clip((count + 1) / (missions_count + 1) * 255, 0, 255))
        for i in range(vertices.shape[0]):
            grid_map.fill_circle(vertices[i, :2], DOT_RADIUS, gray)

    # save plot
    img_jet = cv2.applyColorMap(grid_map.img, cv2.COLORMAP_TURBO)
    img_jet[grid_map.img == 0, :] = 0
    if args.output:
        cv2.imwrite(args.output, img_jet)
    # cv2.imshow("output", img_jet)
    # cv2.waitKey(0)


if __name__ == '__main__':
    run()
