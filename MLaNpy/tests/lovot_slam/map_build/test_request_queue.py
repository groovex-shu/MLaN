import pytest

from lovot_slam.map_build.request_queue import BuildSingleMapOption, MergeMapsOption, RequestQueue, RequestTypes
from lovot_slam.redis.clients import create_ltm_client


@pytest.fixture(name="ltm_client")
def ltm_client_with_cleanup():
    ltm_client = create_ltm_client()

    ltm_client.delete(RequestQueue._PERSIST_REDIS_KEY)
    yield ltm_client
    ltm_client.delete(RequestQueue._PERSIST_REDIS_KEY)


def test_push_pop(ltm_client):
    queue = RequestQueue(ltm_client)
    assert queue.empty()

    queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name="test1"))
    assert not queue.empty()
    assert len(queue) == 1

    request_type, option = queue.pop()
    assert request_type == RequestTypes.BuildMap
    assert option.map_name == "test1"

    assert queue.empty()
    assert len(queue) == 0


def test_clear(ltm_client):
    queue = RequestQueue(ltm_client)
    queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name="test1"))
    queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name="test2"))
    queue.push(RequestTypes.MergeMaps, MergeMapsOption(input_map="test3", maps_to_append=["test4", "test5"]))
    queue.push(RequestTypes.BuildSingleMissionMap, BuildSingleMapOption(map_name="test6"))

    assert not queue.empty()
    assert len(queue) == 4

    queue.clear()
    assert queue.empty()
    assert len(queue) == 0
    assert ltm_client.get(RequestQueue._PERSIST_REDIS_KEY) is None


def test_map_names_in_request(ltm_client):
    queue = RequestQueue(ltm_client)
    queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name="test1"))
    queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name="test2"))
    queue.push(RequestTypes.MergeMaps, MergeMapsOption(input_map="test3", maps_to_append=["test4", "test5"]))
    queue.push(RequestTypes.BuildSingleMissionMap, BuildSingleMapOption(map_name="test6"))

    assert set(queue.get_map_names_in_requests()) == set(["test1", "test2", "test6"])


def test_store_load(ltm_client):
    queue = RequestQueue(ltm_client)
    queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name="test1"))
    queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name="test2"))
    queue.push(RequestTypes.MergeMaps, MergeMapsOption(input_map="test3", maps_to_append=["test4", "test5"]))
    queue.push(RequestTypes.BuildSingleMissionMap, BuildSingleMapOption(map_name="test6"))
    queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name="test7"))

    req_type, req_option = queue.pop()
    assert req_type == RequestTypes.BuildMap
    assert req_option.map_name == "test1"

    # Re-create a new queue and load from redis
    queue = RequestQueue(ltm_client)
    queue.load_from_redis()

    assert len(queue) == 2
    assert queue.pop() == (RequestTypes.BuildMap, BuildSingleMapOption(map_name="test2"))
    assert queue.pop() == (RequestTypes.BuildMap, BuildSingleMapOption(map_name="test7"))

    # Re-create a new queue and load from redis
    queue = RequestQueue(ltm_client)
    queue.load_from_redis()

    assert queue.empty()


@pytest.mark.parametrize("invalid_data", [
    '',  # empty string
    '[]',  # empty list
])
def test_load_empty(ltm_client, invalid_data):
    ltm_client.set(RequestQueue._PERSIST_REDIS_KEY, invalid_data)
    queue = RequestQueue(ltm_client)
    queue.load_from_redis()
    assert queue.empty()


@pytest.mark.parametrize("invalid_data", [
    'invalid_data',  # invalid string
    '[0, {"map_name": "test1"}]',  # invalid depth
    '[[0, {"map_name": "test1"}], [0, {"map_name": "test2"}]',  # incomplete data
    '[[0, {"map_name": "test1"}], [3, {"map_name": "test2"}]]',  # invalid type (3)
    '[[0, {"file_name": "test1"}], [0, {"map_name": "test2"}]]',  # invalid option
])
def test_load_invalid_data(ltm_client, invalid_data):
    ltm_client.set(RequestQueue._PERSIST_REDIS_KEY, invalid_data)
    queue = RequestQueue(ltm_client)
    queue.load_from_redis()
    assert queue.empty()

    assert ltm_client.get(RequestQueue._PERSIST_REDIS_KEY) is None
