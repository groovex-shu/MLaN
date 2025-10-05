import os
from logging import getLogger
from typing import Optional

import redis

from lovot_slam.wifi.type import Covariance, Reliablity, Localizer

_logger = getLogger(__name__)

_LOVOT_REDIS_STM = os.getenv('LOVOT_REDIS_STM', 'localhost:6379:1').split(':')


class LocalizationClient:
    def __init__(self, redis_host: str, redis_port: int, redis_db: int) -> None:
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)


    def get_localizer(self) -> Optional[Localizer]:
        try:
            res = self.redis.get('slam:pose:localizer')
            if res is None:
                return None
            
            return Localizer.value_of(res)
        
        except redis.RedisError as exc:
            _logger.debug(f"Redis error: {exc}")
            return None
        
        except ValueError:
            raise RuntimeError

    def get_covariance(self) -> Optional[Covariance]:
        try:
            result = self.redis.hgetall('slam:pose:covariance')
            if not result or 'covariance' not in result or 'timestamp' not in result:
                return None
            return Covariance.from_csv_string(result['covariance'], float(result['timestamp']))
        
        except redis.RedisError as exc:
            _logger.debug(f"Redis error: {exc}")
            return None
        
        except (ValueError, TypeError):
            raise RuntimeError

    def get_failure_detection_results(self) -> Optional[Reliablity]:

        require_set = {'timestamp', 'reliability', 'detection', 'likelihood'}
        try:
            result = self.redis.hgetall('slam:failure_detection:result')
            if not result or not require_set.issubset(result.keys()):
                return None

            return Reliablity(
                float(result['timestamp']),
                float(result['reliability']),
                float(result['detection']),
                float(result['likelihood']),
            )

        except redis.RedisError as exc:
            _logger.debug(f"Redis error: {exc}")
            return None

        except (ValueError, TypeError):
            raise RuntimeError

    def get_map_name(self) -> Optional[str]:
        try:
            result = self.redis.hget('slam:map', 'name')
            if not result:
                return None
            return result
        
        except redis.RedisError as exc:
            _logger.debug(f"Redis error: {exc}")
            return None


def open_localization_client() -> LocalizationClient:
    """
    Creates a LocalizationClient instance.
    
    Example:
        client = open_localization_client()
        localizer = client.get_localizer()
        print(localizer)
    """
    redis_host, redis_port, redis_db = _LOVOT_REDIS_STM
    client = LocalizationClient(redis_host, int(redis_port), int(redis_db))
    
    return client
