import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from lovot_slam.env import data_directories
from lovot_slam.flags.debug_params import PARAM_BUILD_MAP_RATE
from lovot_slam.tools.maplab_console_commands_generator import MaplabConsoleCommandsGenerator
from lovot_slam.tools.utils import SubprocessRunner, run_batch_runner

TMP_DIR = Path("/tmp/lovot_localization")

MAPS_ROOT = data_directories.maps
BAGS_ROOT = data_directories.bags

logger = logging.getLogger(__name__)


def _generate_optimizer_yaml(yaml_path: Path,
                             input_map: Path, output_map: Path,
                             fix_ncamera_intrinsics: bool = True) -> None:
    with MaplabConsoleCommandsGenerator(yaml_path) as gen:
        gen.set_loop_closure_options(lc_num_ransac_iters=400,
                                     lc_ransac_pixel_sigma=4)

        cmd = gen.commands
        cmd.load(input_map)
        # output original map -- debug
        # os.mkdir(output_map / 'vertices_original')
        # cmd.csv_export_vertices_only(output_map / 'vertices_original')

        # optimization before keyframing (rtl, lc, optvi: twice)
        num_iterations_before_keyframing = 5
        cmd.retriangulate_landmarks()

        # select keyframes using heuristic
        cmd.keyframe_heuristic(kf_distance_threshold_m=0.2,
                               kf_every_nth_vertex=15)

        cmd.loopclosure_all_missions()
        cmd.optimize_visual_inertial(
            num_iterations_before_keyframing,
            ba_fix_ncamera_intrinsics=fix_ncamera_intrinsics)
        cmd.retriangulate_landmarks()
        cmd.loopclosure_all_missions()
        cmd.optimize_visual_inertial(
            num_iterations_before_keyframing,
            ba_fix_ncamera_intrinsics=fix_ncamera_intrinsics)

        # optimization after keyframing (rtl, lc, optvi)
        num_iterations_before_keyframing = 300
        cmd.retriangulate_landmarks()
        cmd.loopclosure_all_missions()
        cmd.optimize_visual_inertial(
            num_iterations_before_keyframing,
            ba_fix_ncamera_intrinsics=fix_ncamera_intrinsics)


        # save
        logger.info(f"Optimization yaml: {str(yaml_path)}")
        commands_str = ", ".join(cmd._commands)
        logger.info(f"Optimization commands: {commands_str}")

        cmd.save(output_map, overwrite=True)
        mission_ids_yaml = output_map / 'vi_map' / 'missions.yaml'
        cmd.export_mission_ids(mission_ids_yaml)
        csv_path = output_map / 'vertices'
        cmd.csv_export_vertices_only(csv_path)


def _build(input_bag: Path, output_map: Path, config_yaml: Path) -> bool:
    cmd = ['roslaunch',
           'lovot_mapping',
           'build_map.launch',
           f'rosbag:={input_bag}',
           f'feature_map:={output_map}',
           f'config_yaml:={config_yaml}',
           f'rosbag_playback_rate:={PARAM_BUILD_MAP_RATE}',
           'rovio_enable_frame_visualization:="false"']

    log_path = TMP_DIR / "build_map.log"
    with log_path.open('w') as f:
        return SubprocessRunner(cmd).run(stdout=f) == 0


def _extract_iteration_count(log_file: Path) -> Optional[int]:
    with log_file.open('r') as f:
        content = f.read()

    pattern = r'outlier-rejection-solver\.h:\d+\]\s+(\d+)'
    matches = re.findall(pattern, content)

    if not matches:
        return None

    last_match = matches[-1]
    it = int(last_match.split(':')[-1])
    return it


def _optimize(input_map: Path, output_map: Path) -> bool:
    batch_control_file = TMP_DIR / "batch_optimize.yaml"
    batch_control_file.unlink(missing_ok=True)

    log_path = TMP_DIR / "batch_optimize.log"
    _generate_optimizer_yaml(batch_control_file,
                             input_map, output_map)
    if not run_batch_runner(batch_control_file, log_path):
        return False

    # always use optimized map, no matter if optimization converged
    it = _extract_iteration_count(log_path)
    logger.info(f"optimization was finished with {it} iteration")
    return True


def _backup_map(map_path: Path) -> None:
    if not map_path.exists():
        return
    backup_path = data_directories.tmp / 'last_failed_map'
    if backup_path.exists():
        shutil.rmtree(backup_path)
    map_path.rename(backup_path)


def main(args):
    start = time.time()

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    input_bag = Path(args.input)
    if not input_bag.exists():
        input_bag = BAGS_ROOT / str(args.input)
    outupt_map = MAPS_ROOT / str(args.output)
    feature_map_path = outupt_map / "feature_map"

    config_yaml = Path(args.config)

    logger.info(f"input : {input_bag}")
    logger.info(f"output: {outupt_map}")

    # build single mission map by rovioli
    ret = _build(input_bag, feature_map_path, config_yaml)
    if not feature_map_path.exists():
        logger.error("map build finished without map")
        sys.exit(1)
    if not ret:
        logger.error("map build failed")
        _backup_map(feature_map_path)
        sys.exit(1)

    # optimize
    ret = _optimize(feature_map_path, feature_map_path)
    if not feature_map_path.exists():
        logger.error("map optimization finished without map")
        sys.exit(1)
    if not ret:
        logger.error("map optimization failed")
        _backup_map(feature_map_path)
        sys.exit(1)

    logger.info(
        f"single mission feature map build duration: {int(time.time() - start)} sec")
