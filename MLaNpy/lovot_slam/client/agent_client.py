from logging import getLogger

import httpx
import requests

from lovot_slam.env import CLOUD_UPLOAD_PORT, LOCALHOST

_logger = getLogger(__name__)

_AGENT_PROXY_HOST = f"http://{LOCALHOST}:{CLOUD_UPLOAD_PORT}/"


async def upload_data_to_cloud(end_point: str, data: str) -> bool:
    _logger.info(f'lovot-agent client uploading to {end_point}')
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                _AGENT_PROXY_HOST + end_point,
                timeout=30,
                data=data,
                headers={'Content-Type': 'application/protobuf'}
            )
    except (OSError, httpx.HTTPError) as e:
        _logger.warning('lovot-agent connection failed: %s: %s', type(e).__name__, e)
        return False
    
    if r.status_code != requests.codes.created:
        _logger.error(
            f"failed to upload data to {end_point}. status_code: {r.status_code}")
        return False
    
    _logger.info(f"succeeded to upload to {end_point}")
    
    return True
