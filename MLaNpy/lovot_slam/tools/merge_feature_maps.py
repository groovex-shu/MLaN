import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np

from lovot_slam.env import MAP_FEATUREMAP, MAP_SUMMARYMAP, MAP_STATISTICS_YAML, data_directories
from lovot_slam.feature_map.feature_map_vertices import FeatureMapVertices, VerticesComparator
from lovot_slam.feature_map.missions_filter import MissionsFilter
from lovot_slam.flags.debug_params import (PARAM_MERGE_MAPS_MINIMUM_MISSIONS_TO_MAINTAIN,
                                           PARAM_MERGE_MAPS_MINIMUM_MISSIONS_TO_OPTIMIZE)
from lovot_slam.tools.maplab_console_commands_generator import MaplabConsoleCommandsGenerator
from lovot_slam.tools.utils import run_batch_runner
from lovot_slam.flags.cloudconfig import CLOUDCONFIG_LC_MIN_NEIGHBORS

TMP_DIR = Path("/tmp/lovot_localization")

MAPS_ROOT = data_directories.maps

THRESHOLD_AVE_DISTANCE_ERROR_TO_FIX = 0.1
THRESHOLD_MAX_DISTANCE_ERROR_TO_FIX = 0.2

logger = logging.getLogger(__name__)


def _mission_id_from(feature_map_path: Path) -> List[str]:
    vertices = FeatureMapVertices.from_map_path(feature_map_path.parent)
    return vertices.missions


def _generate_merger_yaml(yaml_path: Path,
                          input_map: Path, output_map: Path, summary_map: Path,
                          maps_to_append: List[Path],
                          num_iterations: int = 30):
    with MaplabConsoleCommandsGenerator(yaml_path) as gen:
        gen.set_loop_closure_options(lc_num_ransac_iters=400,
                                     lc_ransac_pixel_sigma=4)

        cmd = gen.commands
        cmd.load(input_map)
        # if the all mission baseframes are already known, this is not effective.
        cmd.set_mission_baseframe_to_known()

        # load and merge all maps, and remove missions which cannot be anchored
        append_missions = []
        for map_to_append in maps_to_append:
            cmd.load_merge_map(map_to_append)
            cmd.anchor_all_missions()
            append_missions += _mission_id_from(map_to_append)
        cmd.anchor_all_missions()

        cmd.anchor_all_missions(lc_min_num_neighbors=CLOUDCONFIG_LC_MIN_NEIGHBORS)

        cmd.remove_unknown_missions()

        # pose graph optimization
        cmd.relax()
        cmd.retriangulate_landmarks()
        # find loop closures between the new missions and the existing missions.
        cmd.loopclosure_missions_to_all(append_missions)
        cmd.optimize_visual_inertial(num_iterations)

        # save
        cmd.save(output_map, overwrite=True)
        mission_ids_yaml = output_map / 'vi_map' / 'missions.yaml'
        cmd.export_mission_ids(mission_ids_yaml)
        csv_path = output_map / 'vertices'
        cmd.csv_export_vertices_only(csv_path)
        csv_path = output_map / 'adjacency.csv'
        cmd.create_missions_adjacency_matrix(csv_path)

        # summary map
        cmd.generate_summary_map_and_save_to_disk(summary_map)


def _generate_optimizer_yaml(yaml_path: Path,
                             input_map: Path, output_map: Path,
                             missions_to_remove: Optional[List[str]] = [],
                             missions_to_fix: Optional[List[str]] = [],
                             num_iterations: int = 30,
                             fix_ncamera_intrinsics: bool = True):
    with MaplabConsoleCommandsGenerator(yaml_path) as gen:
        gen.set_loop_closure_options(lc_num_ransac_iters=400,
                                     lc_ransac_pixel_sigma=4)

        cmd = gen.commands
        cmd.load(input_map)

        # remove missions
        for mission_id in missions_to_remove:
            cmd.remove_mission(mission_id)

        # fix missions (set as constant) during optimization
        for mission_id in missions_to_fix:
            cmd.set_mission_vertices_fixed(mission_id)

        # optimization
        cmd.relax()
        cmd.retriangulate_landmarks()
        # cmd.loopclosure_all_missions()  # seems no need to loop close twice
        cmd.optimize_visual_inertial(num_iterations,
                                     ba_fix_ncamera_intrinsics=fix_ncamera_intrinsics)

        # save
        cmd.save(output_map, overwrite=True)
        mission_ids_yaml = output_map / 'vi_map' / 'missions.yaml'
        cmd.export_mission_ids(mission_ids_yaml)
        csv_path = output_map / 'vertices'
        cmd.csv_export_vertices_only(csv_path)
        csv_path = output_map / 'adjacency.csv'
        cmd.create_missions_adjacency_matrix(csv_path)
        stats_yaml_path = output_map / MAP_STATISTICS_YAML
        cmd.export_map_stats_yaml(stats_yaml_path)

def _merge(input_map: Path, output_map: Path, summary_map: Path,
           maps_to_append: List[Path], num_iterations: int) -> bool:
    batch_control_file = TMP_DIR / "batch_merge.yaml"
    if batch_control_file.exists():
        batch_control_file.unlink()

    _generate_merger_yaml(batch_control_file, input_map, output_map, summary_map, maps_to_append,
                          num_iterations=num_iterations)
    log_file = TMP_DIR / "batch_merge.log"
    return run_batch_runner(batch_control_file, log_file)


def _optimize(input_map: Path, output_map: Path,
              missions_to_remove: Optional[List[str]] = [],
              missions_to_fix: Optional[List[str]] = [],
              num_iterations: int = 30,
              fix_ncamera_intrinsics: bool = True) -> int:
    batch_control_file = TMP_DIR / "batch_merge_optimize.yaml"
    if batch_control_file.exists():
        batch_control_file.unlink()

    _generate_optimizer_yaml(batch_control_file,
                             input_map, output_map,
                             missions_to_remove=missions_to_remove,
                             missions_to_fix=missions_to_fix,
                             num_iterations=num_iterations,
                             fix_ncamera_intrinsics=fix_ncamera_intrinsics)
    log_file = TMP_DIR / "batch_merge_optimize.log"
    return run_batch_runner(batch_control_file, log_file)


def _find_missions_to_fix(reference_map: Path, target_map: Path,
                          minimum_to_unfix: int = 0) -> List[str]:
    """Not fix missions which have keyframes with large difference, or new missions.
    = Fix missions with small diff keyframes.
    """
    reference = FeatureMapVertices.from_map_path(reference_map)
    target = FeatureMapVertices.from_map_path(target_map)
    if not reference or not target:
        raise RuntimeError(
            f"Failed to create FeatureMapVertices from {reference_map} or {target_map}")
    comparator = VerticesComparator(reference, target)

    missions_to_fix = OrderedDict()

    for mission_id in target.missions:
        if mission_id not in reference.missions:
            logger.info(f"{mission_id} is not in the reference.")
            continue

        # translational errors
        distance_error = comparator.distance_diffs(mission_id)
        distance_ave = np.average(distance_error)
        distance_max = np.max(distance_error)

        # rotational errors (angle)
        angle_error = np.degrees(comparator.angle_diffs(mission_id))
        angle_ave = np.average(angle_error)
        angle_max = np.max(angle_error)

        logger.info(f"{mission_id}: {distance_ave:.3f}, {distance_max:.3f}, {angle_ave:.3f}, {angle_max:.3f}")

        if distance_ave < THRESHOLD_AVE_DISTANCE_ERROR_TO_FIX and \
                distance_max < THRESHOLD_MAX_DISTANCE_ERROR_TO_FIX:
            missions_to_fix[mission_id] = distance_ave

    # limit the number of missions to fix
    maximum_to_fix = max(0, len(target.missions) - minimum_to_unfix)
    if maximum_to_fix < len(missions_to_fix):
        # sort in order of less difference (likely to fix)
        missions_to_fix = OrderedDict(
            sorted(missions_to_fix.items(), key=lambda x: x[1]))
        for _ in range(len(missions_to_fix) - maximum_to_fix):
            missions_to_fix.popitem()

    missions_to_fix = list(missions_to_fix.keys())
    logger.info(f'fixed / total: {len(missions_to_fix)} / {len(target.missions)}')

    return missions_to_fix


def _find_missions_to_remove(map_path: Path, max_count: Optional[int] = None) -> List[str]:
    missions_filter = MissionsFilter.create_from_map_path(map_path)
    if not missions_filter:
        return []

    missions_to_remove = missions_filter.filter_by_overlapping()

    # limit missions to remove, to maintain minimum number of missions
    max_missions_to_remove = max(len(missions_filter.sorted_mission_ids) -
                                 PARAM_MERGE_MAPS_MINIMUM_MISSIONS_TO_MAINTAIN, 0)
    if len(missions_to_remove) > max_missions_to_remove:
        missions_to_remove = missions_to_remove[:max_missions_to_remove]

    # filter by mission count
    missions_to_remove = missions_filter.filter_by_count(missions_to_remove, max_count)
    return missions_to_remove


def main(args):
    start = time.time()

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    input_map = MAPS_ROOT / str(args.input)
    maps_to_append = [MAPS_ROOT / str(map_name) for map_name in args.append] if args.append else []
    outupt_map = MAPS_ROOT / str(args.output)

    logger.info(f"merge maps input : {input_map}")
    logger.info(f"merge maps append: {maps_to_append}")
    logger.info(f"merge maps output: {outupt_map}")

    # TODO: check whether the input map conatins any of append maps
    # if so, remove them

    # merge maps and optimize with a few iterations
    logger.info('merging maps and optimizing with a few iterations...')
    _merge(input_map / MAP_FEATUREMAP,
           outupt_map / MAP_FEATUREMAP,
           outupt_map / MAP_SUMMARYMAP,
           [map_path / MAP_FEATUREMAP for map_path in maps_to_append],
           num_iterations=5)

    try:
        # compare the input map and the optimized map
        # to find missions with small changes (which can be fixed during optimization)
        missions_to_fix = _find_missions_to_fix(input_map, outupt_map,
                                                minimum_to_unfix=PARAM_MERGE_MAPS_MINIMUM_MISSIONS_TO_OPTIMIZE)
    except RuntimeError as e:
        logger.error(e)
        sys.exit(1)

    # find missions which can be removed
    missions_to_remove = _find_missions_to_remove(Path(outupt_map))
    missions_to_fix = [mission for mission in missions_to_fix
                       if mission not in missions_to_remove]
    if missions_to_remove:
        logger.info(f'removing {len(missions_to_remove)} missions: '
                    f'[{[id[:7] for id in missions_to_remove]}]')

    # optimize
    logger.info('optimizing the merged map...')
    _optimize(outupt_map / MAP_FEATUREMAP,
              outupt_map / MAP_FEATUREMAP,
              missions_to_remove=missions_to_remove,
              missions_to_fix=missions_to_fix,
              num_iterations=30,
              fix_ncamera_intrinsics=True)

    logger.info(f"merge maps duration: {int(time.time() - start)} sec")
