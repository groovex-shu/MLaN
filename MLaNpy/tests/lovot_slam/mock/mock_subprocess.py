from typing import Optional

import trio

from lovot_slam.utils.exceptions import SlamProcessError


class MockRosmaster:
    def __init__(self) -> None:
        self._holder = None

    def is_master_online(self):
        return bool(self._holder)

    def start(self, name):
        self._holder = name

    def stop(self, name, force=False):
        if self._holder == name or force:
            self._holder = None


mock_rosmaster = MockRosmaster()


class MockSubprocessBase:
    def __init__(self, output_to_console: bool = False) -> None:
        self._name = None
        self._process = None
        self._running = False

    @property
    def pid(self) -> Optional[int]:
        return self._process

    def _start_process(self, cmd, parser=None, name='default'):
        global mock_rosmaster
        mock_rosmaster.start(self._name)
        self._process = 1
        self._running = True

    async def stop_process_and_wait(self, timeout=25.0):
        global mock_rosmaster
        mock_rosmaster.stop(self._name)
        if self._process is None:
            return
        await trio.sleep(1.0)

    def is_running(self):
        return self._process and self._running

    def is_terminated(self):
        return self._process and not self._running

    def is_stopped(self):
        return not self._process

    def dump_stdout(self, num_lines: int = 20):
        pass

    def dump_stderr(self, num_lines: int = 20):
        pass

    def clear(self):
        if self.is_terminated():
            self._process = None
        else:
            raise SlamProcessError

    async def wait_for_termination(self):
        await trio.sleep(1.0)
        self._running = False


class BaseSubprocess(MockSubprocessBase):
    def __init__(self, variants, output_to_console: bool = False) -> None:
        super().__init__(output_to_console)
        self._name = "base"
        self._variants = variants

    def start(self, omni_yaml):
        cmd = []
        self._start_process(cmd)

    def is_terminated(self):
        # Emulate roslaunch kills itself when rosmaster is down
        if not mock_rosmaster.is_master_online():
            self._running = False
        return super().is_terminated()


class LocalizationSubprocess(MockSubprocessBase):
    def __init__(self, variants, output_to_console: bool = False) -> None:
        super().__init__(output_to_console)
        self._name = "localization"
        self._map_name = None
        self._variants = variants

    @property
    def map_name(self):
        return self._map_name

    def start(self, map_name, vi_map_folder):
        if self.is_running():
            raise SlamProcessError

        self._map_name = map_name

        cmd = []
        self._start_process(cmd)


class RecordSubprocess(MockSubprocessBase):
    def __init__(self, variants, output_to_console: bool = False) -> None:
        super().__init__(output_to_console)
        self._name = "record"
        self._variants = variants

    def start(self, bag_file):
        cmd = []
        self._start_process(cmd)
