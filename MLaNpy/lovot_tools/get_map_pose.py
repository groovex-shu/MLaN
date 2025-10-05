import argparse

from grid_map_util.occupancy_grid import OccupancyGrid

from localization_tools.pose_setter import PoseSetter


def parse_args():
    parser = argparse.ArgumentParser(description="get map pose of clicked point.")
    parser.add_argument("target", help="target map path")
    args = parser.parse_args()
    return args


def run():
    args = parse_args()
    grid_map = OccupancyGrid.from_yaml_file(args.target)
    pose_setter = PoseSetter(grid_map)
    pose_setter.set_callback(print)
    pose_setter.plot()


if __name__ == '__main__':
    run()
