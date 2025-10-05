import subprocess
from logging import getLogger

import trio
import yaml

_logger = getLogger(__name__)


async def is_camera_info_recorded(bag_file: str) -> bool:
    """Parse `rosbag info -y` and check if
    rostopic named `/depth/camera_info` with type of `sensor_msgs/CameraInfo` is recorded.
    """
    cmd = ['rosbag', 'info', '-y', str(bag_file)]
    proc = await trio.run_process(cmd,
                                  capture_stdout=True,
                                  capture_stderr=subprocess.DEVNULL)
    try:
        info = yaml.safe_load(proc.stdout.decode())
    except yaml.YAMLError:
        _logger.error('failed to parse rosbag info.')
        return False

    found = any([True for t in info.get('topics', [])
                 if t.get('topic') == '/depth/camera_info'
                 and t.get('type') == 'sensor_msgs/CameraInfo'])
    if not found:
        _logger.warning('CameraInfo is not recorded in %s', bag_file)
    return found
