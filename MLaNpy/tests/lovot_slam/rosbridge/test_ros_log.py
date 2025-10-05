import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from lovot_slam.rosbridge.ros_log import RosLog

NO_SHUTDOWN = """
[roslaunch][INFO] 2022-06-24 03:49:28,286: process[odom_nest_tf_publisher-10]: cwd will be [/var/log/ros]
[roslaunch][INFO] 2022-06-24 03:49:28,290: process[odom_nest_tf_publisher-10]: started with pid [5030]
[roslaunch][INFO] 2022-06-24 03:49:28,290: ... successfully launched [odom_nest_tf_publisher-10]
[roslaunch][INFO] 2022-06-24 03:49:28,290: ... launch_nodes complete
"""

SHUTDOWN_TWIST_PUBLISHER = """
[roslaunch][INFO] 2022-06-24 03:47:19,472: spin
[roslaunch][ERROR] 2022-06-24 03:49:01,544: \
================================================================================\
REQUIRED process [twist_publisher_node-7] has died!
process has died [pid 1920, exit code -15, cmd /opt/ros/noetic/lib/twist_publisher/twist_publisher_node __name\
:=twist_publisher_node __log:=/var/log/ros/log/585222ae-f370-11ec-ab85-7085c2e3f79c/twist_publisher_node-7.log].
log file: /var/log/ros/log/585222ae-f370-11ec-ab85-7085c2e3f79c/twist_publisher_node-7*.log
Initiating shutdown!
================================================================================
"""

SHUTDOWN_OMNI_STREMAER = """
[roslaunch.parent][INFO] 2022-06-24 03:49:28,290: ... roslaunch parent running, waiting for process exit
[roslaunch][INFO] 2022-06-24 03:49:28,290: spin
[roslaunch][ERROR] 2022-06-24 03:51:16,613: \
================================================================================\
REQUIRED process [omni_streamer-8] has died!
process has died [pid 5027, exit code -15, cmd /opt/ros/noetic/lib/lovot_shm_bridge/omni_streamer_node \
--config_yaml /opt/lovot/share/lovot-localization/configs/top_camera_realtime.yaml \
__name:=omni_streamer __log:=/var/log/ros/log/a5369276-f370-11ec-8190-7085c2e3f79c/omni_streamer-8.log].
log file: /var/log/ros/log/a5369276-f370-11ec-8190-7085c2e3f79c/omni_streamer-8*.log
Initiating shutdown!
================================================================================
[roslaunch.pmon][INFO] 2022-06-24 03:51:16,714: ProcessMonitor._post_run \
<ProcessMonitor(ProcessMonitor-1, started daemon 140557705488128)>
"""


@pytest.mark.parametrize('logs,expected_datetime,expected_proc', [
    ([], None, None),
    ([NO_SHUTDOWN], None, None),
    ([SHUTDOWN_TWIST_PUBLISHER], datetime(2022, 6, 24, 3, 49, 1), 'twist_publisher_node'),
    ([SHUTDOWN_OMNI_STREMAER], datetime(2022, 6, 24, 3, 51, 16), 'omni_streamer'),
    # should find the latest shutdown log message
    ([SHUTDOWN_TWIST_PUBLISHER, SHUTDOWN_OMNI_STREMAER], datetime(2022, 6, 24, 3, 51, 16), 'omni_streamer'),
])
def test_get_launch_shutdown_process(logs, expected_datetime, expected_proc):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        latest_log_dir = tmpdir / 'latest'
        latest_log_dir.mkdir()
        with open(latest_log_dir / 'roslaunch-tom-1234.log', 'w') as f:
            for log in logs:
                f.write(log)

        ros_log = RosLog(Path(tmpdir), 'tom')
        shutdown_datetime, shutdown_proc = ros_log.get_launch_shutdown_process(1234)

        assert shutdown_datetime == expected_datetime
        assert shutdown_proc == expected_proc


def test_get_launch_shutdown_process_without_logfile():
    with tempfile.TemporaryDirectory() as tmpdir:
        ros_log = RosLog(Path(tmpdir), 'tom')
        shutdown_datetime, shutdown_proc = ros_log.get_launch_shutdown_process(1234)

        assert shutdown_datetime is None
        assert shutdown_proc is None
