import pathlib
import signal
import subprocess
from logging import getLogger
from typing import AsyncIterator, List, Optional

import anyio

from lovot_slam.env import ENV_PATH, OMNI_CONVERSION_YAML
from lovot_slam.flags.cloudconfig import (
    CLOUDCONFIG_ENABLE_MARKER_LOCALIZATION_REGISTER,
    CLOUDCONFIG_ENABLE_MARKER_LOCALIZATION_RELOCALIZATION,
    CLOUDCONFIG_DISABLE_MAP_UPDATE,
)
from lovot_slam.flags.debug_params import PARAM_BUILD_DENSE_MAP_RATE, PARAM_FRONT_CAMERA_FRAMERATE
from lovot_slam.model import HardwareVariants, LovotModel, Model, NestModel
from lovot_slam.utils.exceptions import SlamProcessError
from lovot_slam.utils.file_util import tail

logger = getLogger(__name__)

# NOTE: taskset is used to prevent cpu spike on Jetson
PREFIX_COMMAND_TASKSET = ['taskset', '-c', '0']


def _get_ros_package_name(model: LovotModel) -> str:
    if model >= LovotModel.LV110:
        return 'lovot_localization_coro2'
    return 'lovot_localization_coro1'


def _dump_log(file_path: pathlib.Path, num_lines: int):
    if not file_path.exists():
        return
    try:
        log_messages = tail(file_path, num_lines)
        logger.info(f'dump {file_path}')
        for message in log_messages:
            logger.info(message)
    except EnvironmentError:
        pass


class SubprocessBase:
    _TMP_LOG_DIR = pathlib.Path('/tmp/lovot_localization')

    def __init__(self, output_to_console: bool = False) -> None:
        self._name = None
        self._process = None
        self._output_to_console = output_to_console
        self._stdout = None
        self._stderr = None

    def _create_temp_dir(self) -> None:
        self._TMP_LOG_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def _stdout_dump_file(self) -> pathlib.Path:
        # NOTE: this file is intended to be overwritten.
        return self._TMP_LOG_DIR / f'{self._name}_stdout.log'

    @property
    def _stderr_dump_file(self) -> pathlib.Path:
        # NOTE: this file is intended to be overwritten.
        return self._TMP_LOG_DIR / f'{self._name}_stderr.log'

    @property
    def pid(self) -> Optional[int]:
        if self._process is None:
            return None
        return self._process.pid

    def _start_process(self, cmd, parser=None, name='default'):
        if self.is_running():
            raise RuntimeError("Current process is running.")
        try:
            # if DEVNULL is specified as PIPE, dump stdout/stderr to a file.
            if not self._output_to_console:
                self._create_temp_dir()
                self._stdout = open(self._stdout_dump_file, "w")
                self._stderr = open(self._stderr_dump_file, "w")
            logger.debug(f'starting subprocess {cmd}')
            self._process = subprocess.Popen(cmd,
                                             stdout=self._stdout,
                                             stderr=self._stderr,
                                             bufsize=1)
        except OSError as e:
            logger.error(f'error starting subprocess: {e}')
            raise SlamProcessError

    async def stop_process_and_wait(self, timeout=25.0):
        """stop process started by subprocess.
        terminate process and wait for shutdown.
        when timeout [sec] passed without shutdown, kill process.
        timeout should be longer than 15 + 2 sec, when stopping roslaunch process.
        since the timeout for sigint and siterm are 15.0 sec and 2.0 sec respectively.
        https://github.com/ros/ros_comm/blob/noetic-devel/tools/roslaunch/src/roslaunch/nodeprocess.py#L58:L59
        """
        if self._process is None:
            await anyio.sleep(0)
            return
        if self._process.poll() is not None:
            await anyio.sleep(0)
            # already terminated
            self.clear()
            return

        async def wait_for_termination() -> bool:
            with anyio.move_on_after(timeout, shield=True):
                # shield cancel scope, in order to ensure to terminate the process
                # especially for the case when sigint/sigterm is called.
                while True:
                    if self._process.poll() is not None:
                        logger.info(f'{self._name} process terminated')
                        return True
                    await anyio.sleep(0.1)
            return False

        try:
            logger.info(f'terminating {self._name} process...')
            self._process.send_signal(signal.SIGINT)
            if await wait_for_termination():
                return
            logger.error(f'terminate {self._name} process failed. kill process')
            self._process.kill()
            await wait_for_termination()
        except subprocess.SubprocessError:
            raise SlamProcessError
        finally:
            self.clear()

    def is_running(self):
        return self._process and self._process.poll() is None

    def is_terminated(self):
        return self._process and self._process.poll() is not None

    def is_stopped(self):
        """is the process stopped?
        different from is_terminated, this method returns True only when the process is cleared explicitly.
        when the process is terminated by itself, is_terminated returns True while is_stopped returns False.
        """
        return not self._process

    def dump_stdout(self, num_lines: int = 20):
        _dump_log(self._stdout_dump_file, num_lines)

    def dump_stderr(self, num_lines: int = 20):
        _dump_log(self._stderr_dump_file, num_lines)

    def clear(self):
        if self.is_terminated():
            self._process = None
            if self._stdout:
                self._stdout.close()
                self._stdout = None
            if self._stderr:
                self._stderr.close()
                self._stderr = None
        else:
            raise SlamProcessError

    async def wait_for_termination(self):
        while True:
            if not self.is_running():
                return
            await anyio.sleep(0.1)


class BaseSubprocess(SubprocessBase):
    def __init__(self, variants: HardwareVariants, output_to_console: bool = False) -> None:
        super().__init__(output_to_console)
        self._name = "base"
        self._variants = variants
        self._package_name = _get_ros_package_name(variants.model)

        self._prefix_cmd = None
        self._args = []
        if variants.model is LovotModel.LV110:
            self._prefix_cmd = PREFIX_COMMAND_TASKSET
            
            if CLOUDCONFIG_ENABLE_MARKER_LOCALIZATION_REGISTER:
                self._args.append('enable_marker_register:=true')

            if CLOUDCONFIG_ENABLE_MARKER_LOCALIZATION_RELOCALIZATION:
                self._args.append('enable_marker_relocalization:=true')
            
        if CLOUDCONFIG_DISABLE_MAP_UPDATE:
            self._args.append('enable_map_update:=false')

    def start(self, omni_yaml):
        cmd = ['roslaunch',
               self._package_name,
               'realtime_base.launch',
               f'omni_yaml:={omni_yaml}',
               f'depth_camera:={self._variants.depth_camera.value}',
               f'front_camera_framerate:={PARAM_FRONT_CAMERA_FRAMERATE}']
        if self._prefix_cmd:
            cmd = self._prefix_cmd + cmd
        if self._args:
            cmd = cmd + self._args
        self._start_process(cmd)


class LocalizationSubprocess(SubprocessBase):
    def __init__(self, variants: HardwareVariants, output_to_console: bool = False) -> None:
        super().__init__(output_to_console)
        self._name = "localization"
        self._package_name = _get_ros_package_name(variants.model)

        self._map_name = None

        self._prefix_cmd = None
        if variants.model is LovotModel.LV110:
            self._prefix_cmd = PREFIX_COMMAND_TASKSET

    @property
    def map_name(self):
        return self._map_name

    def start(self, map_name, vi_map_folder):
        if self.is_running():
            raise SlamProcessError

        self._map_name = map_name

        cmd = ['roslaunch',
               self._package_name,
               'realtime_lc.launch',
               'map_name:=' + map_name,
               'vi_map_folder:=' + vi_map_folder]
        if self._prefix_cmd:
            cmd = self._prefix_cmd + cmd
        self._start_process(cmd)


class RecordSubprocess(SubprocessBase):
    def __init__(self, variants: HardwareVariants, output_to_console: bool = False) -> None:
        super().__init__(output_to_console)
        self._name = "record"
        self._variants = variants
        self._package_name = _get_ros_package_name(variants.model)

        self._prefix_cmd = None
        if variants.model is LovotModel.LV110:
            self._prefix_cmd = PREFIX_COMMAND_TASKSET

    def start(self, bag_file):
        cmd = ['roslaunch',
               self._package_name,
               'realtime_record.launch',
               f'rosbag_filename:={bag_file}',
               f'depth_camera:={self._variants.depth_camera.value}']
        if self._prefix_cmd:
            cmd = self._prefix_cmd + cmd
        self._start_process(cmd)


class BuildMapSubprocess(SubprocessBase):
    def __init__(self, model: Model, output_to_console: bool = False, journal: bool = False) -> None:
        super().__init__(output_to_console)
        self._model = model
        self._name = ""
        self._journal = journal

    def start_bag_conversion(self, original_bag: str, converted_bag: str):
        self._name = "bag_conversion"
        cmd = ['roslaunch',
               'lovot_mapping',
               'bag_converter.launch',
               f'original_bag:={original_bag}',
               f'converted_bag:={converted_bag}',
               'mode:=0',
               f'conversion_yaml:={OMNI_CONVERSION_YAML}']
        self._start_process(cmd)

    def start_bag_diminish(self, original_bag: str, topics: str, vertices_csv: str, converted_bag: str):
        self._name = "bag_diminish"
        cmd = ['roslaunch',
               'lovot_mapping',
               'bag_diminish.launch',
               f'original_bag:={original_bag}',
               f'topics:={topics}',
               f'vertices_csv:={vertices_csv}',
               f'converted_bag:={converted_bag}']
        self._start_process(cmd)

    def start_bag_prune(self, original_bag: str, topics: list, converted_bag: str):
        self._name = "bag_prune"
        topic_syn = ' or '.join([f"topic == '{topic}'" for topic in topics])
        cmd = ['rosbag', 'filter', original_bag, converted_bag, topic_syn]
        self._start_process(cmd)

    def start_build_feature_map(self, converted_bag: str, map_dir: str, config_dir: str):
        self._name = "build_feature_map"
        cmd = [ENV_PATH / 'bin' / 'lovot_slam_tools']
        if self._journal:
            cmd += ['--journal']
        cmd += ['build_single_mission_feature_map',
                '--input', converted_bag,
                '--output', map_dir,
                '--config', config_dir]
        self._start_process(cmd)

    def start_scale_map(self, map_name: str, source_maps: List[str], mission_ids: List[str]):
        self._name = "scale_map"
        cmd = [ENV_PATH / 'bin' / 'lovot_slam_tools']
        if self._journal:
            cmd += ['--journal']
        cmd += ['scale_map']
        cmd += [map_name]
        cmd += ['--source_maps'] + source_maps
        cmd += ['--mission_ids'] + mission_ids
        self._start_process(cmd)

    def start_build_dense_map(self, maps_root: str, bags_root: str, config_yaml: str,
                              map_name: str, mission_id: str, publish_camera_info: bool):
        self.name = "build_dense_map"
        machine_type = "coro1" if isinstance(self._model, NestModel) else "coro2"
        # NOTE: filter_ground (octomap option) is set to true for coro2
        filter_ground = "true" if machine_type == "coro2" else "false"
        cmd = ['roslaunch',
               'lovot_mapping',
               'octomap.launch',
               f'maps_root:={maps_root}',
               f'map_name:={map_name}',
               f'mission_id:={mission_id}',
               f'config_yaml:={config_yaml}',
               f'rosbag:={bags_root}/{map_name}_converted.bag',
               f'rosbag_playback_rate:={PARAM_BUILD_DENSE_MAP_RATE}',
               f'machine_type:={machine_type}',
               f'publish_camera_info:={"true" if publish_camera_info else "false"}',
               f'filter_ground:={filter_ground}']
        self._start_process(cmd)

    def start_merge_dense_maps(self, map_name: str, source_maps: List[str], mission_ids: List[str]):
        self._name = "merge_dense_map"
        machine_type = "coro1" if isinstance(self._model, NestModel) else "coro2"
        cmd = [ENV_PATH / 'bin' / 'lovot_slam_tools']
        if self._journal:
            cmd += ['--journal']
        cmd += ['build_merged_dense_map']
        cmd += [map_name]
        cmd += ['--source_maps'] + source_maps
        cmd += ['--mission_ids'] + mission_ids
        cmd += ['--machine-type', machine_type]
        self._start_process(cmd)

    def start_merge_feature_maps(self, input_map: str, output_map: str, maps_to_append: List[str]) -> None:
        self.name = "merge_feature_maps"
        cmd = [ENV_PATH / 'bin' / 'lovot_slam_tools']
        if self._journal:
            cmd += ['--journal']
        cmd += ['merge_feature_maps',
                '-i', input_map,
                '-o', output_map]
        if maps_to_append:
            cmd += ['-a'] + maps_to_append
        self._start_process(cmd)
