import datetime
import re
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple

_logger = getLogger(__name__)


def _grep(file: str, pattern: str) -> List[str]:
    matched = []
    try:
        with open(file, 'r') as f:
            matched = [line for line in f if re.search(pattern, line)]
    except EnvironmentError:
        _logger.warn(f"Cannot open file: {file}.")
    return matched


def _get_line_datetime(line: str) -> datetime.datetime:
    m = re.search(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}", line)
    return datetime.datetime.strptime(line[m.start():m.end()], '%Y-%m-%d %H:%M:%S')


def _get_latest_message(lines: List[str]) -> Tuple[Optional[datetime.datetime], Optional[str]]:
    latest_datetime = None
    latest_log = None
    for line in lines:
        try:
            t = _get_line_datetime(line)
        except (AttributeError, ValueError):
            continue
        if not latest_datetime or latest_datetime < t:
            latest_datetime = t
            latest_log = line
    return latest_datetime, latest_log


class RosLog:
    def __init__(self, log_root: Path, hostname: str) -> None:
        self._log_root = log_root
        self._hostname = hostname

    @property
    def latest_log_dir(self) -> Path:
        return self._log_root / "latest"

    def get_launch_shutdown_process(self, pid: int) -> Tuple[Optional[datetime.datetime], Optional[str]]:
        """find the latest roslaunch shutdown process by extracting lines like:
            roslaunch-tom-100.log:[roslaunch][ERROR] 2020-09-08 01:54:06,119: ={80}REQUIRED process [amcl-2] has died!
        return timestamp, process_name
        """
        log_file = self.latest_log_dir / f"roslaunch-{self._hostname}-{pid}.log"
        if not log_file.exists():
            return None, None

        # grep shutdown reason messages
        messages = _grep(log_file, r"={70,90}REQUIRED process \[.*\] has died!")
        if not messages:
            return None, None

        # find the latest
        latest_datetime, latest_message = _get_latest_message(messages)
        if not latest_datetime or not latest_message:
            return None, None

        # extract process name
        m = re.search(r"={70,90}REQUIRED process \[(.*)-[0-9]+\] has died!", latest_message)
        if not m:
            return None, None
        return latest_datetime, m.group(1)
