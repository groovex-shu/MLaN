import copy
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import UnixStreamServer
from typing import Optional

import pytest
import redis
import trio
from trio_util import periodic

from lovot_slam.redis import create_stm_client
from lovot_slam.utils.omni_camera_mode import (
    CONVERTER_MODE_KEY,
    IMAGE_MODE_KEY,
    CameraMode,
    CameraOperationMode,
    OmniCameraMode,
    OmniCameraMode2,
)

# lovot-camera settings (for coro2)
LOVOT_CAMERA_COMMON_SETTINGS = {
    "blocksize": 4096, "num-buffers": -1, "timeout": 0, "wbmode": "auto",
    "saturation": 1.0, "sensor-id": 0, "sensor-mode": 0, "exposuretimerange": "28000 1000000000",
    "gainrange": "1 72", "ispdigitalgainrange": "1 1", "tnr-strength": -0.1,
    "tnr-mode": "NoiseReduction_Fast", "ee-mode": "EdgeEnhancement_Fast",
    "ee-strength": 0.1, "aeantibanding": "AeAntibandingMode_Auto",
    "exposurecompensation": 0.1, "aelock": False, "aeregion": None, "awblock": False,
    "event-wait": 3000000000, "acquire-wait": 5000000000
}
LOVOT_CAMERA_SETTINGS = {
    CameraOperationMode.NORMAL_MODE: {
        **LOVOT_CAMERA_COMMON_SETTINGS,
        "exposuretimerange": "28000 1000000000",
        "gainrange": "1 72",
        "ispdigitalgainrange": "1 1",
    },
    CameraOperationMode.RECORD_MODE: {
        **LOVOT_CAMERA_COMMON_SETTINGS,
        "exposuretimerange": "1000000 33333333",
        "gainrange": "1 16",
        "ispdigitalgainrange": "1 2",
    },
}


class UnixSocketCameraServer(SimpleHTTPRequestHandler):
    """Mock of lovot-camera server (coro2)
    This mock is used to simulate the unix socket interface of lovot-camera.
    """

    # storage should be managed as a class variable
    _settings = LOVOT_CAMERA_SETTINGS[CameraOperationMode.NORMAL_MODE]

    # simulate the delay of lovot-camera (but it's not likely happen)
    # 1 means that 1 additional retry with the same settings is required to apply change
    DELAY_COUNT = 0
    _call_count = 0

    @classmethod
    def increment_call_count(cls) -> int:
        cls._call_count += 1
        return cls._call_count

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_settings(self):
        return self._settings

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        if self.path == '/elements/horn-top/settings':
            self.wfile.write(json.dumps(self.get_settings()).encode())

    def do_POST(self):
        if self.path == '/horn-top/reset-settings':
            self._settings.update(LOVOT_CAMERA_SETTINGS[CameraOperationMode.NORMAL_MODE])
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_PATCH(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        update = json.loads(post_data)

        # Update camera state based on the settings
        current_count = self.increment_call_count()
        # if the same settings are called multiple times, delay the update
        if current_count > self.DELAY_COUNT:
            self._settings.update(update)

        self.send_response(204)
        self.end_headers()

    def address_string(self):
        return 'unix_socket_client'


@pytest.fixture
def mock_lovot_camera_server():
    sock_file = "/tmp/lovot-camera.sock"
    if os.path.exists(sock_file):
        os.remove(sock_file)

    server = UnixStreamServer(sock_file, UnixSocketCameraServer)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    yield server

    server.shutdown()
    server_thread.join()

    if os.path.exists(sock_file):
        os.remove(sock_file)


class MockOmniConverter:
    """Mock of lovot-omni-converter
    overview specification of lovot-omni-converter
    - client softwares (e.g. localization) publish target mode (0~4) to STM `omni:image_mode`
    - lovot-omni-converter (and lovot-omni-server@jerry) switches mode (takes a few seconds)
    - lovot-omni-server sets current mode to STM `omni:converter_mode`

    This Mock simulates the part that sets the mode to STM key after delay time for STM publish
    """
    def __init__(self, initial_mode: Optional[CameraMode], delay: int) -> None:
        self._delay = delay

        self._stm_client = create_stm_client()
        if initial_mode is not None:
            self._stm_client.set(CONVERTER_MODE_KEY, str(initial_mode.value))

        self._pubsub = self._stm_client.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe([IMAGE_MODE_KEY])

    async def run(self):
        try:
            async for _ in periodic(0.1):
                message = self._pubsub.get_message()
                if message is None:
                    continue
                print(f'{message=}')
                if message['channel'] == IMAGE_MODE_KEY:
                    print(f'{message["data"]=}')
                    await trio.sleep(self._delay)
                    self._stm_client.set(CONVERTER_MODE_KEY, str(message['data']))
        finally:
            self._pubsub.unsubscribe()


@pytest.fixture
def setup_stm():
    stm_client = create_stm_client()
    stm_client.delete(CONVERTER_MODE_KEY)
    yield stm_client
    stm_client.delete(CONVERTER_MODE_KEY)


@pytest.mark.parametrize("value,mode", [
    ("0", CameraMode.MODE_0),
    ("1", CameraMode.MODE_1),
    ("2", CameraMode.MODE_2),
    ("3", CameraMode.MODE_3),
    ("4", CameraMode.MODE_4),
    ("", None),
    (None, None),
])
def test_get_current_mode(setup_stm: redis.StrictRedis, value, mode):
    stm_client = setup_stm
    if value is not None:
        stm_client.set(CONVERTER_MODE_KEY, value)

    mode_setter = OmniCameraMode()
    assert mode_setter.get_current_mode() == mode


@pytest.mark.parametrize("initial_mode,target_mode,delay,retry_count,success", [
    (CameraMode.MODE_0, CameraMode.MODE_2, 1.0, 1, True),
    (CameraMode.MODE_0, CameraMode.MODE_2, 20.0, 1, False),  # fails because timeout is 15 sec
    (CameraMode.MODE_0, CameraMode.MODE_2, 20.0, 2, True),  # success on second attempt
    (CameraMode.MODE_0, CameraMode.MODE_0, 20.0, 1, True),  # success because the initial mode is the same as target
    (None, CameraMode.MODE_0, 1.0, 1, True),
])
async def test_change_mode_and_wait(setup_stm: redis.StrictRedis,
                                    initial_mode, target_mode, delay, retry_count, success,
                                    nursery, autojump_clock):
    converter = MockOmniConverter(initial_mode, delay)
    nursery.start_soon(converter.run)

    mode_setter = OmniCameraMode()
    assert await mode_setter.change_mode_and_wait(target_mode, retry_count) == success


@pytest.mark.parametrize("initial_mode,target_mode", [
    (CameraOperationMode.NORMAL_MODE, CameraOperationMode.RECORD_MODE),
    (CameraOperationMode.NORMAL_MODE, CameraOperationMode.NORMAL_MODE),
    (CameraOperationMode.RECORD_MODE, CameraOperationMode.NORMAL_MODE),
    (CameraOperationMode.RECORD_MODE, CameraOperationMode.RECORD_MODE),
])
async def test_change_mode_and_wait2(mock_lovot_camera_server, initial_mode, target_mode, autojump_clock):
    UnixSocketCameraServer._settings = copy.deepcopy(LOVOT_CAMERA_SETTINGS[initial_mode])

    mode_setter = OmniCameraMode2()
    result = await mode_setter.change_mode_and_wait(target_mode, 1)
    assert result

    assert UnixSocketCameraServer._settings == LOVOT_CAMERA_SETTINGS[target_mode]


async def test_change_mode_and_wait2_failure(mock_lovot_camera_server, autojump_clock):
    UnixSocketCameraServer._settings = copy.deepcopy(LOVOT_CAMERA_SETTINGS[CameraOperationMode.NORMAL_MODE])
    UnixSocketCameraServer.DELAY_COUNT = 1

    mode_setter = OmniCameraMode2()
    result = await mode_setter.change_mode_and_wait(CameraOperationMode.RECORD_MODE, 0)
    assert not result


async def test_change_mode_and_wait2_retry(mock_lovot_camera_server, autojump_clock):
    UnixSocketCameraServer._settings = copy.deepcopy(LOVOT_CAMERA_SETTINGS[CameraOperationMode.NORMAL_MODE])
    UnixSocketCameraServer.DELAY_COUNT = 2

    mode_setter = OmniCameraMode2()
    result = await mode_setter.change_mode_and_wait(CameraOperationMode.RECORD_MODE, 3)
    assert result

    assert UnixSocketCameraServer._settings == LOVOT_CAMERA_SETTINGS[CameraOperationMode.RECORD_MODE]


async def test_reset_settings(mock_lovot_camera_server, autojump_clock):
    # Set initial settings to record mode
    UnixSocketCameraServer._settings = copy.deepcopy(LOVOT_CAMERA_SETTINGS[CameraOperationMode.RECORD_MODE])
    
    mode_setter = OmniCameraMode2()
    result = mode_setter._reset_settings()
    assert result
    
    # Verify settings were reset to normal mode defaults
    assert UnixSocketCameraServer._settings == LOVOT_CAMERA_SETTINGS[CameraOperationMode.NORMAL_MODE]
