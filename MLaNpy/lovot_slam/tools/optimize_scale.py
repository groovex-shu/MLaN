import logging
import time
from pathlib import Path

from lovot_slam.env import MAP_FEATUREMAP, MAP_SUMMARYMAP, data_directories
from lovot_slam.tools.maplab_console_commands_generator import MaplabConsoleCommandsGenerator
from lovot_slam.tools.utils import run_batch_runner

MAPS_ROOT = data_directories.maps
TMP_DIR = Path("/tmp/lovot_localization")

logger = logging.getLogger(__name__)


def _generate_optimizer_yaml(yaml_path: Path,
                             input_map: Path, output_map: Path, summary_map: Path,
                             distance_edge_filename: Path,
                             num_iterations: int = 30):
    with MaplabConsoleCommandsGenerator(yaml_path) as gen:
        gen.set_loop_closure_options(lc_num_ransac_iters=400,
                                     lc_ransac_pixel_sigma=4)

        cmd = gen.commands
        cmd.load(input_map)

        # optimization with distance edges
        cmd.optimize_visual_inertial(num_iterations,
                                     ba_distance_edge_filename = distance_edge_filename,
                                     ba_fix_ncamera_intrinsics=False)


        # save
        cmd.save(output_map, overwrite=True)
        mission_ids_yaml = output_map / 'vi_map' / 'missions.yaml'
        cmd.export_mission_ids(mission_ids_yaml)
        csv_path = output_map / 'vertices'
        cmd.csv_export_vertices_only(csv_path)

        # summary map
        cmd.generate_summary_map_and_save_to_disk(summary_map, overwrite=True)


def _optimize(input_map: Path, 
              output_map: Path, 
              summary_map: Path,
              distance_edge_filename: Path,
              num_iterations: int = 30) -> bool:
    batch_control_file = TMP_DIR / "batch_optimize_scale.yaml"
    if batch_control_file.exists():
        batch_control_file.unlink()

    _generate_optimizer_yaml(batch_control_file,
                             input_map, output_map, summary_map,
                             distance_edge_filename,
                             num_iterations=num_iterations)
    log_file = TMP_DIR / 'optimize_scale.log'
    return run_batch_runner(batch_control_file, log_file)


def optimize_scale(map_name: str) -> bool:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    target_map = MAPS_ROOT / map_name
    logger.info(f"target: {target_map}")

    # optimize
    # TODO: revisit num_iterations which seems to be too short
    logger.info('optimizing scale of the map...')
    it = 30
    logger.info(f'iteration: {it}')
    return _optimize(target_map / MAP_FEATUREMAP,
                     target_map / MAP_FEATUREMAP,
                     target_map / MAP_SUMMARYMAP,
                     target_map / MAP_FEATUREMAP / "distance_edges.csv", 
                     num_iterations=it)


def main(args):
    start = time.time()

    optimize_scale(args.target)

    logger.info(f"scale optimization duration: {int(time.time() - start)} sec")
