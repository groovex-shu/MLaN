import json
from contextlib import contextmanager
from enum import Enum
from io import BytesIO
from logging import getLogger
from typing import Optional

import pycurl
import redis
import trio
from trio_util import periodic

from lovot_slam.redis import create_stm_client

logger = getLogger(__name__)

IMAGE_MODE_KEY = 'omni:image_mode'
CONVERTER_MODE_KEY = 'omni:converter_mode'


class CameraOperationMode(Enum):
    """Enum class representing software-defined operation modes for the camera.
    This class is used on both coro1 and coro2.
        NORMAL_MODE: Mode used during localization
        RECORD_MODE: Mode used during recording (explore)
    """
    NORMAL_MODE = 0
    RECORD_MODE = 1


# coro1
class CameraMode(Enum):
    """These modes directly correspond to capture modes of Coro1's omni camera.
    """
    MODE_0 = 0      #record
    MODE_1 = 1
    MODE_2 = 2      #default
    MODE_3 = 3
    MODE_4 = 4
    COUNT = 5

    @classmethod
    def from_value(cls, value: int) -> 'CameraMode':
        if value not in range(CameraMode.COUNT.value):
            raise ValueError(f'{value} is not a valid omni camera mode.')
        for e in CameraMode:
            if e.value == value:
                return e

# coro1
class OmniCameraMode:
    def __init__(self) -> None:
        self._redis_stm = create_stm_client()

    def get_current_mode(self) -> Optional[CameraMode]:
        """Get current omni camera mode.
        :return: CameraMode if a valid mode is read by STM, else None
        """
        try:
            return CameraMode.from_value(int(self._redis_stm.get(CONVERTER_MODE_KEY)))
        except (redis.RedisError, TypeError, ValueError) as e:
            logger.error(f'failed to get omni camera mode: {e}')
            return None

    def _change_mode(self, mode: CameraMode) -> bool:
        """Publish mode to the image_mode key.
        """
        try:
            self._redis_stm.publish(IMAGE_MODE_KEY, str(mode.value))
        except redis.RedisError as e:
            logger.warning(f'failed to publish {str(mode.value)} to {IMAGE_MODE_KEY}: {e}')
            return False
        return True

    async def change_mode_and_wait(self, mode: CameraOperationMode, retry_count: int) -> bool:
        """Change regulus camera mode and wait.
        :param mode: mode to change
        :param retry_count: maximum retry count (it takes up to 15 seconds for each try)
        :returns True if successfully changed, else False
        """
        if mode == CameraOperationMode.NORMAL_MODE:
            mode = CameraMode.MODE_2
        elif mode == CameraOperationMode.RECORD_MODE:
            mode = CameraMode.MODE_0

        current_mode = self.get_current_mode()
        if current_mode is mode:
            logger.info(f'omni camera mode is already {current_mode}')
            return True

        for _ in range(retry_count):
            logger.info(f'changing omni camera mode from {current_mode} to {mode}')
            if not self._change_mode(mode):
                await trio.sleep(1)
                continue

            with trio.move_on_after(15):
                async for _ in periodic(1):
                    current_mode = self.get_current_mode()
                    if mode is current_mode:
                        return True
            logger.warning(f'failed to change regulus mode to {mode} and the current mode is {current_mode}')
        return False


# coro2
class OmniCameraMode2:
    SETTINGS_ENDPOINT = "elements/horn-top/settings"
    RESET_SETTINGS_ENDPOINT = "horn-top/reset-settings"

    # Camera settings for mapping (explore),
    # which is mainly to prevent abnormal vibration of AE (auto exposure)
    # while reducing motion blur even in the low light environment.
    # https://groove-x.atlassian.net/wiki/spaces/GXREC/pages/2602172684/Camera+settings+top+cam
    RECORD_SETTINGS = {
        "exposuretimerange": "1000000 33333333",
        "gainrange": "1 16",
        "ispdigitalgainrange": "1 2"
    }

    def __init__(self,
                 host: str = "localhost",
                 unix_socket: str = "/tmp/lovot-camera.sock"):
        self._host = host
        self._unix_socket = unix_socket

    @contextmanager
    def _curl_context(self, url: str):
        """Context manager for handling pycurl resources.

        Yields:
            tuple: (pycurl.Curl, BytesIO) objects for use in the context
        """
        b = BytesIO()
        c = pycurl.Curl()
        try:
            c.setopt(c.URL, url)
            c.setopt(c.WRITEDATA, b)
            c.setopt(c.HTTPHEADER, ['Content-Type: application/json'])
            c.setopt(c.UNIX_SOCKET_PATH, self._unix_socket)
            yield c, b
        finally:
            c.close()
            b.close()

    def _get_settings(self) -> dict:
        '''
        Returns:
            data(dict): full dict from json response
        '''
        url = f"http://{self._host}/{self.SETTINGS_ENDPOINT}"
        with self._curl_context(url) as (c, b):
            try:
                c.perform()
                http_code = c.getinfo(pycurl.HTTP_CODE)
                if 200 <= http_code < 300:
                    data = json.loads(b.getvalue())
                    return data

                logger.error(f'Curl error: {http_code}')
            except (pycurl.error, IOError) as e:
                logger.error(f'failed to get camera settings: {e}')
            return {}

    def _set_settings(self, data: dict) -> bool:
        url = f"http://{self._host}/{self.SETTINGS_ENDPOINT}"
        with self._curl_context(url) as (c, b):
            try:
                c.setopt(c.CUSTOMREQUEST, "PATCH")
                c.setopt(c.POSTFIELDS, json.dumps(data))
                c.perform()
                http_code = c.getinfo(pycurl.HTTP_CODE)
                if 200 <= http_code < 300:
                    return True

                logger.error(f'Curl error: {http_code}')
            except (pycurl.error, IOError) as e:
                logger.error(f'failed to set camera settings: {e}')
            return False

    def _reset_settings(self) -> bool:
        """Reset camera settings to default values.
        Returns:
            bool: True if reset was successful, False otherwise
        """
        url = f"http://{self._host}/{self.RESET_SETTINGS_ENDPOINT}"
        with self._curl_context(url) as (c, b):
            try:
                c.setopt(c.CUSTOMREQUEST, "POST")
                c.perform()
                http_code = c.getinfo(pycurl.HTTP_CODE)
                if 200 <= http_code < 300:
                    return True

                logger.error(f'Curl error: {http_code}')
            except (pycurl.error, IOError) as e:
                logger.error(f'failed to reset camera settings: {e}')
            return False

    async def change_mode_and_wait(self, mode: CameraOperationMode, retry_count: int) -> bool:
        '''
        Args:
            mode: int -> mode code
            retry_count: int -> maximum retry count
        '''
        await trio.sleep(0)

        if mode == CameraOperationMode.NORMAL_MODE:
            while retry_count:
                if self._reset_settings():
                    return True
                await trio.sleep(1)
                retry_count -= 1

            logger.error('failed to reset camera mode')
            return False

        elif mode == CameraOperationMode.RECORD_MODE:
            # status check
            current_setting = self._get_settings()
            if self.RECORD_SETTINGS.items() <= current_setting.items():
                logger.info(f'camera mode is already {mode}')
                return True

            # update mode with limited retry
            while retry_count:
                if self._set_settings(self.RECORD_SETTINGS):
                    # check if the setting is updated
                    await trio.sleep(1)
                    new_setting = self._get_settings()
                    if self.RECORD_SETTINGS.items() <= new_setting.items():
                        return True

                await trio.sleep(1)
                retry_count -= 1

            # all failed
            logger.error('failed to set camera mode')
            return False
