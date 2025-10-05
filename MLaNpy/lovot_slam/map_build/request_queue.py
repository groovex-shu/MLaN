import enum
import json
import queue
from logging import getLogger
from typing import List, Tuple, Union

import redis
from attr import define, field
from cattr import structure, unstructure

_logger = getLogger(__name__)


class RequestTypes(enum.Enum):
    """Types of requests that can be put into the request queue
    only BuildMap is used in production while the others are used for testing
    """
    BuildMap = 0  # build a single mission map and merge it with the latest merged map
    BuildSingleMissionMap = 1  # just build a single mission map
    MergeMaps = 2  # just merge maps


@define(frozen=True)
class BuildSingleMapOption:
    map_name: str


@define(frozen=True)
class MergeMapsOption:
    input_map: str
    maps_to_append: List[str] = field(factory=list)


BuildOption = Union[BuildSingleMapOption, MergeMapsOption]

_CORRESPONDING_REQUEST_TYPE = {
    RequestTypes.BuildMap: BuildSingleMapOption,
    RequestTypes.BuildSingleMissionMap: BuildSingleMapOption,
    RequestTypes.MergeMaps: MergeMapsOption,
}


class RequestQueue:
    _PERSIST_REDIS_KEY = "slam:request_queue"
    _REQUEST_TYPES_TO_PERSIST = [RequestTypes.BuildMap]

    def __init__(self, ltm_client: redis.StrictRedis) -> None:
        self._queue = queue.Queue()

        self._ltm_client = ltm_client

    def __len__(self) -> int:
        return self._queue.qsize()

    def push(self, request_type: RequestTypes, request_option: BuildOption) -> None:
        assert isinstance(request_option, _CORRESPONDING_REQUEST_TYPE[request_type])
        self._queue.put((request_type, request_option))
        # Persist the request to redis
        self.store_to_redis()

    def pop(self) -> Tuple[RequestTypes, BuildOption]:
        item = self._queue.get()
        self.store_to_redis()
        return item

    def clear(self) -> None:
        _logger.info("Clearing request queue")
        with self._queue.mutex:
            self._queue.queue.clear()
        self._ltm_client.delete(self._PERSIST_REDIS_KEY)

    def empty(self) -> bool:
        return self._queue.empty()

    def get_map_names_in_requests(self) -> List[str]:
        with self._queue.mutex:
            return [option.map_name for _, option in self._queue.queue
                    if isinstance(option, BuildSingleMapOption)]

    def _serialize(self, include_request_types: List[RequestTypes]) -> str:
        with self._queue.mutex:
            ser = unstructure([(request_type, option)
                            for request_type, option in self._queue.queue
                            if request_type in include_request_types])
        return json.dumps(ser)

    @staticmethod
    def _deserialize(serialized: str) -> List[Tuple[RequestTypes, BuildOption]]:
        ser = json.loads(serialized)
        return structure(ser, List[Tuple[RequestTypes, BuildOption]])

    def store_to_redis(self) -> None:
        """ Store the request queue to redis """
        serialized = self._serialize(self._REQUEST_TYPES_TO_PERSIST)
        self._ltm_client.set(self._PERSIST_REDIS_KEY, serialized)

    def load_from_redis(self) -> None:
        """ Load the request queue from redis
        This will clear the current queue and replace it with the one from redis
        """
        _logger.info("Loading request queue from redis")
        serialized = self._ltm_client.get(self._PERSIST_REDIS_KEY)
        if not serialized:
            _logger.info("No request queue found in redis")
            return
        try:
            request_list = self._deserialize(serialized)
        except Exception as e:
            _logger.error(f"Failed to deserialize request queue: {e}")
            self._ltm_client.delete(self._PERSIST_REDIS_KEY)
            return

        self.clear()
        # Push the requests to the queue
        # this will also persist the requests to redis
        for request_type, request_option in request_list:
            self.push(request_type, request_option)
        _logger.info(f"Loaded {len(request_list)} requests from redis")
