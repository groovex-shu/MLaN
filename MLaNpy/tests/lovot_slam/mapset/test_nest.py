import pytest
import fakeredis
from unittest.mock import patch

from lovot_slam.mapset.nest import NestMapsetPair



@pytest.fixture
def mock_redis():
    return fakeredis.FakeStrictRedis(decode_responses=True)


def test_query_nest_mapset_pair(mock_redis):
    # Setup mock data
    mock_redis.hset(NestMapsetPair._KEY, "nest_2", "mapset_234")
    
    # Test get_mapset_id with a specific nest_id
    with patch.object(NestMapsetPair, '_ltm', mock_redis):
        mapset_id = NestMapsetPair.query_nest_mapset_pair("nest_1")
        assert mapset_id is None

        mapset_id = NestMapsetPair.query_nest_mapset_pair("nest_2")
        assert mapset_id == "mapset_234"


def test_add_nest_mapset_pair(mock_redis):
    # Test add_nest_mapset_pair
    with patch.object(NestMapsetPair, '_ltm', mock_redis):
        NestMapsetPair.add_nest_mapset_pair("nest_3", "mapset_345")
        mapset_id = mock_redis.hget(NestMapsetPair._KEY, "nest_3")
        assert mapset_id == "mapset_345"


def test_remove_nest_mapset_pair(mock_redis):
    # Setup mock data
    mock_redis.hset(NestMapsetPair._KEY, "nest_4", "mapset_456")
    
    # Test remove_nest_mapset_pair
    with patch.object(NestMapsetPair, '_ltm', mock_redis):
        NestMapsetPair.remove_nest_mapset_pair("nest_4")
        mapset_id = mock_redis.hget(NestMapsetPair._KEY, "nest_4")
        assert mapset_id is None


def test_remove_all(mock_redis):
    # Setup mock data
    mock_redis.hset(NestMapsetPair._KEY, "nest_5", "mapset_567")
    
    # Test remove_all
    with patch.object(NestMapsetPair, '_ltm', mock_redis):
        NestMapsetPair.remove_all()
        check_hash = mock_redis.exists(NestMapsetPair._KEY)
        assert check_hash == 0