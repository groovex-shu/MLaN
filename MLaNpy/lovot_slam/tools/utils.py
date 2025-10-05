import logging
import signal
import subprocess
from pathlib import Path
from typing import IO, List, Optional

from lovot_slam.flags.debug_params import PARAM_MAP_OPTIMIZATION_THREAD_NUM

_logger = logging.getLogger(__name__)


class SubprocessRunner:
    def __init__(self, cmd: List[str]) -> None:
        self._cmd = cmd

    def run(self, stdout: Optional[IO] = None, stderr: Optional[IO] = None) -> Optional[int]:
        process = None
        try:
            process = subprocess.Popen(self._cmd, stdout=stdout, stderr=stderr)
            process.wait()
            retruncode = process.returncode
            process = None
            return retruncode
        finally:
            if process is not None:
                _logger.info('Terminating process...')
                process.send_signal(signal.SIGINT)
                process.wait()


def run_batch_runner(batch_control_file: Path, log_path: Path) -> bool:
    """Run maplab batch runner.
    This blocks until the batch runner finishes.
    :param batch_control_file: Path to the batch control yaml file.
    :param log_path: Path to the log file. (NOTE: log stderr not stdout)
    :return: True if the batch runner finished successfully.
    """
    assert batch_control_file.exists()

    cmd = ['roslaunch',
           'lovot_mapping',
           'maplab_batch_runner.launch',
           f'batch_control_file:={batch_control_file}',
           f'num_hardware_threads:={PARAM_MAP_OPTIMIZATION_THREAD_NUM}']
    with log_path.open('w') as f:
        return SubprocessRunner(cmd).run(stderr=f) == 0
