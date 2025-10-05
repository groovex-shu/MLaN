import json

from lovot_slam.env import data_directories
from lovot_slam.utils.map_utils import MapUtils


def main(args):
    map_utils = MapUtils(data_directories.maps, data_directories.bags)
    pairs = map_utils.get_mapname_missionid_pairs(args.map_name)
    print(json.dumps(pairs))
