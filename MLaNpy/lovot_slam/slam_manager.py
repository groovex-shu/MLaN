import platform
from contextlib import AsyncExitStack
from logging import getLogger
import datetime

import anyio

from lovot_slam.env import ROS_LOG_ROOT, data_directories, redis_keys
from lovot_slam.redis.clients import (create_ltm_client, 
                                      create_stm_client,
                                      create_async_stm_client,
                                      create_async_ltm_client)
from lovot_slam.redis.keys import GHOST_ID_KEY, RedisKeyRepository
# from lovot_slam.redis_listener import RedisListener
from lovot_slam.rosbridge.ros_log import RosLog
from lovot_map.utils.map_utils import (BagUtils, 
                                       MapUtils, 
                                       SpotUtils)

logger = getLogger(__name__)

CLEANUP_TIMEOUT_SEC = 30.0


class SlamManager:
    def __init__(self, debug=False, journal=False):
        logger.info('initialize SlamManager')
        
        # redis client
        self.redis_stm = create_stm_client()
        self.redis_ltm = create_ltm_client()
        self.aioredis_stm = create_async_stm_client()
        self.aioredis_ltm = create_async_ltm_client()

        # async pubsub listener
        self.listener = self.aioredis_stm.pubsub()

        # read ghost
        self.ghost_id = self.redis_ltm.get(GHOST_ID_KEY)
        self._model = None

        # helper class to access bags
        self.bag_utils = BagUtils(data_directories.bags)
        self.bag_utils.create_directory()

        # helper class to access maps
        self.map_utils = MapUtils(data_directories.maps, data_directories.bags)
        self.map_utils.create_directory()

        self._monitor_root = data_directories.monitor

        # helper class to access spots
        self.spot_utils = SpotUtils(self.map_utils)

        # ros log collector (collect from ${ROS_HOME}/log/*)
        hostname = platform.node()
        self._ros_log = RosLog(ROS_LOG_ROOT, hostname)

        # subprocess output to console when debug mode, otherwise to files
        self._subprocess_output_to_console = debug

        self._journal = journal

        # TODO: refactor using async
        self._last_time_execute: datetime.datetime = datetime.datetime.min

    def parse_request(self, req):
        req_list = req.split(' ')
        if len(req_list) < 2:
            return '', '', []
        req_cmd = req_list[0]
        req_id = req_list[1]
        req_args = req_list[2:]
        return req_cmd, req_id, req_args

    def publish_response(self, resp_cmd, resp_id, resp_res, err=None):
        response = resp_cmd + ' ' + resp_id + ' ' + resp_res
        if err is not None:
            response += (' ' + str(err))
        response = response.strip()
        self.redis_stm.publish(redis_keys.response, response)

    async def _monitor_command(self):
        await self.listener.subscribe(redis_keys.command)

        while True:
            message = await self.listener.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if message and message['type'] == 'message':
                command = message['data']
                if isinstance(command, bytes):
                    command = command.decode()
                if isinstance(command, str):
                    # logger.debug(f"command: {command}")
                    await self.process_command(command)
            await anyio.sleep(0.01)

    async def run(self):
        try:
            async with AsyncExitStack() as stack:
                await self._setup_context(stack)

                async with anyio.create_task_group() as tg:
                    tg.start_soon(self._run_main)
                    tg.start_soon(self._monitor_command)
        finally:
            with anyio.fail_after(CLEANUP_TIMEOUT_SEC, shield=True):
                await self._stop()
                await self.listener.unsubscribe()
                await self.listener.close()


    # --- below methods should be implemented in subclass ---
    async def process_command(self, req: str):
        raise NotImplementedError

    async def _run_main(self):
        raise NotImplementedError

    async def _stop(self):
        raise NotImplementedError

    async def _setup_context(self, stack: AsyncExitStack) -> None:
        raise NotImplementedError

    