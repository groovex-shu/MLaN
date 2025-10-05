import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

from grid_map_util.occupancy_grid import OccupancyGrid


def parse_args():
    parser = argparse.ArgumentParser(
        description='align two different transforms')
    parser.add_argument('map', help='path to input map.yaml')
    parser.add_argument('csv', help='path to input csv')
    parser.add_argument('tf_yaml', help='path to input transform matrix yaml')
    return parser.parse_args()


class TransformsPlotter:
    def __init__(self, oc_map: OccupancyGrid, transforms_csv: str, tf_yaml: str):
        self.fig, self.ax = plt.subplots()

        self.oc_map = oc_map

        self.data = np.loadtxt(transforms_csv,
                               delimiter=",",
                               skiprows=0,
                               usecols=(0, 1, 7, 8))

        with open(tf_yaml) as f:
            loaded = yaml.safe_load(f)
        self.transform_matrix = np.array(loaded)
        print(self.transform_matrix)
        self.inverse_transform_matrix = np.linalg.inv(self.transform_matrix)

    def plot(self, ratio=5):
        img = cv2.cvtColor(self.oc_map.img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=3)
        for i in range(self.data.shape[0]):
            # a: loop closure
            pos_a = np.append(self.data[i][0:2],
                              np.array([0, 1])).reshape((4, 1))
            pos_a = np.dot(self.inverse_transform_matrix, pos_a)
            pt_a = self.oc_map.realcoords_to_cvcoords(
                pos_a.reshape((4))[0:2]) * ratio

            # b: vicon
            pos_b = np.append(self.data[i][2:4],
                              np.array([0, 1])).reshape((4, 1))
            pos_b = np.dot(self.inverse_transform_matrix, pos_b)
            pt_b = self.oc_map.realcoords_to_cvcoords(
                pos_b.reshape((4))[0:2]) * ratio

            radius = int(np.linalg.norm(pt_a - pt_b))
            img = cv2.circle(img, (pt_b[0], pt_b[1]), radius, (0, 0, 255), 2)
            img = cv2.line(img, (pt_b[0], pt_b[1]),
                           (pt_a[0], pt_a[1]), (255, 0, 0), 2)
        plt.imshow(img)
        plt.show()
        cv2.imwrite('out.png', img)


def run():
    args = parse_args()

    map_ = OccupancyGrid.from_yaml_file(args.map)
    plotter = TransformsPlotter(map_, args.csv, args.tf_yaml)
    plotter.plot()


if __name__ == '__main__':
    run()
