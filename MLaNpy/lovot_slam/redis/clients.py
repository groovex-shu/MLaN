import os

import redis
import redis.asyncio as aioredis

LOVOT_REDIS_STM = os.getenv('LOVOT_REDIS_STM', '127.0.0.1:6379:1')
LOVOT_REDIS_LTM = os.getenv('LOVOT_REDIS_LTM', '127.0.0.1:6379:2')
LOVOT_REDIS_DEVICE = os.getenv('LOVOT_REDIS_DEVICE', '127.0.0.1:6379:3')

REDIS_HOST_STM, REDIS_PORT_STM, REDIS_DB_STM = LOVOT_REDIS_STM.split(':')
REDIS_HOST_LTM, REDIS_PORT_LTM, REDIS_DB_LTM = LOVOT_REDIS_LTM.split(':')
REDIS_HOST_DEVICE, REDIS_PORT_DEVICE, REDIS_DB_DEVICE = LOVOT_REDIS_DEVICE.split(':')


def create_stm_client(decode_responses: bool = True) -> redis.StrictRedis:
    return redis.StrictRedis(host=REDIS_HOST_STM, port=REDIS_PORT_STM, db=REDIS_DB_STM,
                             socket_connect_timeout=1, decode_responses=decode_responses)


def create_ltm_client(decode_responses: bool = True) -> redis.StrictRedis:
    return redis.StrictRedis(host=REDIS_HOST_LTM, port=REDIS_PORT_LTM, db=REDIS_DB_LTM,
                             socket_connect_timeout=1, decode_responses=decode_responses)


def create_device_client(decode_responses: bool = True) -> redis.StrictRedis:
    return redis.StrictRedis(host=REDIS_HOST_DEVICE, port=REDIS_PORT_DEVICE, db=REDIS_DB_DEVICE,
                             socket_connect_timeout=1, decode_responses=decode_responses)


def create_async_stm_client(decode_responses: bool = True) -> aioredis.Redis:
    return aioredis.Redis(host=REDIS_HOST_STM, port=REDIS_PORT_STM, db=REDIS_DB_STM,
                          socket_connect_timeout=1, decode_responses=decode_responses)

def create_async_ltm_client(decode_responses: bool = True) -> aioredis.Redis:
    return aioredis.Redis(host=REDIS_HOST_LTM, port=REDIS_PORT_LTM, db=REDIS_DB_LTM,
                          socket_connect_timeout=1, decode_responses=decode_responses)