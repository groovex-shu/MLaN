import numpy as np
import pytest
import redis

from lovot_slam.client.localization_client import _LOVOT_REDIS_STM, open_localization_client
from lovot_slam.wifi.type import Covariance, Reliablity, Localizer


@pytest.fixture
def redis_keys_setter(request):
    redis_host, redis_port, redis_db = _LOVOT_REDIS_STM
    stm_client = redis.Redis(host=redis_host, port=int(redis_port), db=int(redis_db), decode_responses=True)

    # set keys to redis
    keys = request.param
    for key, value in keys.items():
        if isinstance(value, dict):
            stm_client.hset(key, mapping=value)
        elif isinstance(value, str):
            stm_client.set(key, value)

    yield keys

    # delete keys from redis
    for key in keys:
        stm_client.delete(key)



@pytest.mark.parametrize(
    "redis_keys_setter,localizer,covariance,reliability,map_name",
    [(
        {
            'slam:pose:localizer': 'visual',
            'slam:pose:covariance': {
                'covariance': '1,0,0,0,1,0,0,0,1',
                'timestamp': '123456789.123456789',
            },
            'slam:failure_detection:result': {
                'timestamp': '123456789.123456789',
                'reliability': '0.1',
                'detection': '0.2',
                'likelihood': '0.3',
            },
            'slam:map': {'name': '20230113_184300'},
        },
        Localizer.VISUAL,
        Covariance(123456789.123456789, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        Reliablity(123456789.123456789, 0.1, 0.2, 0.3),
        '20230113_184300',
    ),
    ],
    indirect=["redis_keys_setter"])
def test_localization_client_getter(redis_keys_setter, localizer, covariance, reliability, map_name):
    client = open_localization_client()
    ret = client.get_localizer()
    assert ret == localizer

    ret = client.get_covariance()
    assert ret == covariance

    ret = client.get_failure_detection_results()
    assert ret == reliability

    ret = client.get_map_name()
    assert ret == map_name



def test_localization_client_getter_with_empty_keys():
    """Test with empty Redis - should return None"""
    redis_host, redis_port, redis_db = _LOVOT_REDIS_STM
    stm_client = redis.Redis(host=redis_host, port=int(redis_port), db=int(redis_db), decode_responses=True)
    
    client = open_localization_client()
    
    result = client.get_localizer()
    assert result is None

    result = client.get_covariance()
    assert result is None

    result = client.get_failure_detection_results()
    assert result is None

    result = client.get_map_name()
    assert result is None


def test_localization_client_getter_with_invalid_enum_values():
    """Test with invalid enum values - should raise RuntimeError"""
    redis_host, redis_port, redis_db = _LOVOT_REDIS_STM
    stm_client = redis.Redis(host=redis_host, port=int(redis_port), db=int(redis_db), decode_responses=True)
    
    # Set invalid enum value
    stm_client.set('slam:pose:localizer', 'invalid')
    
    try:
        client = open_localization_client()
        
        with pytest.raises(RuntimeError):
            _ = client.get_localizer()
    finally:
        # Cleanup
        stm_client.delete('slam:pose:localizer')


def test_localization_client_getter_with_invalid_data_format():
    """Test with invalid data format - get_map_name should return None"""
    redis_host, redis_port, redis_db = _LOVOT_REDIS_STM  
    stm_client = redis.Redis(host=redis_host, port=int(redis_port), db=int(redis_db), decode_responses=True)
    
    # Set invalid data format for map (wrong type - should be hash)
    stm_client.set('slam:map', 'invalid')
    
    try:
        client = open_localization_client()
        
        # get_map_name should return None for wrong data type
        result = client.get_map_name()
        assert result is None
        
    finally:
        # Cleanup
        stm_client.delete('slam:map')


def test_localization_client_getter_with_parsing_errors():
    """Test with data that causes parsing errors - should raise RuntimeError"""
    redis_host, redis_port, redis_db = _LOVOT_REDIS_STM  
    stm_client = redis.Redis(host=redis_host, port=int(redis_port), db=int(redis_db), decode_responses=True)
    
    try:
        client = open_localization_client()
        
        # Test covariance with invalid CSV format (ValueError)
        stm_client.hset('slam:pose:covariance', mapping={
            'covariance': 'invalid_csv_format',  # Invalid CSV → ValueError
            'timestamp': '123456789.123456789',
        })
        with pytest.raises(RuntimeError):
            _ = client.get_covariance()
        stm_client.delete('slam:pose:covariance')
        
        # Test failure_detection with non-numeric values (ValueError)  
        stm_client.hset('slam:failure_detection:result', mapping={
            'timestamp': '123456789.123456789',
            'reliability': 'not_a_number',  # Invalid float → ValueError
            'detection': '0.2',
            'likelihood': '0.3',
        })
        with pytest.raises(RuntimeError):
            _ = client.get_failure_detection_results()
        stm_client.delete('slam:failure_detection:result')
        
        # Test missing fields behavior
        stm_client.hset('slam:pose:covariance', mapping={
            'timestamp': '123456789.123456789',
            # Missing 'covariance' field → explicit check returns None
        })
        result = client.get_covariance()
        assert result is None
        
        stm_client.hset('slam:failure_detection:result', mapping={
            'timestamp': '123456789.123456789',
            # Missing 'reliability' field → explicit check returns None
        })
        result = client.get_failure_detection_results()
        assert result is None
        
    finally:
        # Cleanup
        stm_client.delete('slam:pose:covariance')
        stm_client.delete('slam:failure_detection:result')
