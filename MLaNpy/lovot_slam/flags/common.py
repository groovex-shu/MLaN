from logging import getLogger
from typing import Any, Callable

import redis

from lovot_slam.redis.clients import create_device_client, create_ltm_client

logger = getLogger(__name__)


def get_param(key: str, default: Any, create_client_func: Callable[[None], redis.Redis]) -> Any:
    try:
        value = create_client_func().get(key)
        if value is None:
            return default
    except redis.RedisError:
        return default

    value = type(default)(value)
    logger.warning(f'Param: {key} of type {type(default)} is set to {value}')
    return value


def get_param_from_ltm(key, default):
    return get_param(key, default, create_ltm_client)


def get_param_from_device(key, default):
    return get_param(key, default, create_device_client)
