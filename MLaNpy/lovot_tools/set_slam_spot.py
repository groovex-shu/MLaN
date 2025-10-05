import argparse
from functools import partial

from grid_map_util.occupancy_grid import OccupancyGrid

from localization_tools.pose_setter import PoseSetter
from localization_tools.lovotalk_client.simple_lt_client import SimpleLovotalkClient


def parse_args():
    parser = argparse.ArgumentParser(description="slam:spot:** setter")
    parser.add_argument('hostname', help='target address')
    parser.add_argument('mapname', help='map name')
    parser.add_argument('spotname', help='spot name to set (slam:spot:[name])')
    args = parser.parse_args()
    return args


def set_slam_spot(pose_str, spotname="", lt_client=None):
    if lt_client is None:
        return
    key = f'slam:spot:{spotname}'
    items = {
        "id": spotname,
        "name": spotname,
        "coordinate": pose_str
    }
    print(items)
    lt_client.hmset('LTM', key, items)


def run():
    args = parse_args()

    lt_client = SimpleLovotalkClient(args.hostname)
    spot_keys = lt_client.keys('LTM', 'slam:spot:*')
    print(spot_keys)
    spots = lt_client.hgetall('LTM', spot_keys)

    map_ = OccupancyGrid.from_yaml_file(args.mapname)
    pose_setter = PoseSetter(map_)
    pose_setter.set_callback(partial(set_slam_spot, spotname=args.spotname, lt_client=lt_client))

    for key in spot_keys:
        print(f'{key}: {spots[key]}')
        if 'coordinate' in spots[key]:
            pose_setter.update_arrow(key, spots[key]['coordinate'])

    pose_setter.plot()


if __name__ == '__main__':
    run()
