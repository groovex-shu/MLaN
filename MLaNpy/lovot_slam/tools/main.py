import argparse

from lovot_slam.tools import (build_merged_dense_map, build_single_mission_feature_map, list_mapname_missionid_pairs,
                              merge_2d_maps, merge_feature_maps, optimize_scale, prepare_data_directory, scale_map)
from lovot_slam.utils.logging_util import setup_logging


def run():
    # register subcommand
    parser = argparse.ArgumentParser(description="lovot localization tool")
    parser.add_argument('--debug', action='store_true', help='output debug log')
    parser.add_argument('--journal', action='store_true', help='send log to the journal')
    subparsers = parser.add_subparsers()

    build_merged_dense_map_parser = subparsers.add_parser("build_merged_dense_map",
                                                          help="build merged dense map")
    build_merged_dense_map_parser.add_argument("map_name", help="map name")
    build_merged_dense_map_parser.add_argument("--source_maps", help="source maps", nargs="*")
    build_merged_dense_map_parser.add_argument("--mission_ids", help="mission ids", nargs="*")
    build_merged_dense_map_parser.add_argument("--machine-type", "-m",
                                               help="machine type (default: coro1)",
                                               default="coro1")
    build_merged_dense_map_parser.set_defaults(handler=build_merged_dense_map.main)

    build_single_mission_feature_map_parser = subparsers.add_parser("build_single_mission_feature_map",
                                                                    help="build single mission feature maps")
    build_single_mission_feature_map_parser.add_argument("--input", "-i",
                                                         help="input bag",
                                                         required=True)
    build_single_mission_feature_map_parser.add_argument("--output", "-o",
                                                         help="output map",
                                                         required=True)
    build_single_mission_feature_map_parser.add_argument("--config", "-c",
                                                         help="path to directory containing calibration yaml files",
                                                         default="/data/localization/calibration/")
    build_single_mission_feature_map_parser.set_defaults(
        handler=build_single_mission_feature_map.main)

    merge_feature_maps_parser = subparsers.add_parser("merge_feature_maps", help="merge feature maps")
    merge_feature_maps_parser.add_argument("--input", "-i",
                                           help="input map (a merged map or a single mission map)",
                                           required=True)
    merge_feature_maps_parser.add_argument("--append", "-a",
                                           help="append map(s) (one or multiple single mission maps)",
                                           nargs="*")
    merge_feature_maps_parser.add_argument("--output", "-o", help="output map", required=True)
    merge_feature_maps_parser.set_defaults(handler=merge_feature_maps.main)

    merge_2d_maps_parser = subparsers.add_parser("merge_2d_maps", help="merge feature maps")
    merge_2d_maps_parser.add_argument("maps", help="input map directories",
                                      nargs="*")
    merge_2d_maps_parser.add_argument("--output", "-o",
                                      help="output map directories",
                                      required=True)
    merge_2d_maps_parser.set_defaults(handler=merge_2d_maps.main)

    optimize_scale_parser = subparsers.add_parser("optimize_scale", help="optimize scale of a feature map")
    optimize_scale_parser.add_argument("target",
                                       help="target map (a merged map or a single mission map)")
    optimize_scale_parser.set_defaults(handler=optimize_scale.main)

    prepare_data_directory_parser = subparsers.add_parser("prepare_data_directory",
                                                          help="prepare data directory for the localization mapset")
    prepare_data_directory_parser.set_defaults(handler=prepare_data_directory.main)

    scale_map_parser = subparsers.add_parser("scale_map", help="scale a feature map")
    scale_map_parser.add_argument("map_name", help="map name")
    scale_map_parser.add_argument("--source_maps", help="source maps", nargs="*")
    scale_map_parser.add_argument("--mission_ids", help="mission ids", nargs="*")
    scale_map_parser.set_defaults(handler=scale_map.main)

    list_mapname_missionid_pairs_parser = subparsers.add_parser("list_mapname_missionid_pairs",
                                                                help="get map_name and mission_id pairs")
    list_mapname_missionid_pairs_parser.add_argument("map_name", help="map name")
    list_mapname_missionid_pairs_parser.set_defaults(handler=list_mapname_missionid_pairs.main)

    args = parser.parse_args()
    setup_logging('lovot-localization', args.debug, args.journal)

    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()
