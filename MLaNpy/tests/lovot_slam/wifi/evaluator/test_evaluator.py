import copy
from contextlib import AsyncExitStack

import pytest
import trio
from trio_util import AsyncValue

from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import RadioMap as RadioMap_pb
from lovot_apis.lovot_tf.tf.tf_pb2 import SetTransformResponse

from lovot_slam import Context, context
from lovot_slam.client.lovot_tf_client import open_lovot_tf_client
from lovot_slam.utils.protobuf_util import unix_time_to_pb_timestamp
from lovot_slam.wifi.evaluator.evaluator import InferenceEvaluator
from lovot_slam.wifi.mapping.mapping import RadioMap
from lovot_slam.wifi.type import Fingerprint

from ...client.test_lovot_tf_client import MockTfServicer
from ...client.util import open_servicer_and_client
from ..mapping.test_mapping import DUMMY_FINGERPRINT


class MockFingerprintSync:
    def __init__(self):
        self.fingerprint_event = AsyncValue(None)

    def set_fingerprint(self, fingerprint):
        self.fingerprint_event.value = fingerprint


@pytest.fixture
async def mock_context(nursery):
    async with AsyncExitStack() as stack:
        mock_tf_service, lovot_tf_client = \
            await stack.enter_async_context(open_servicer_and_client(MockTfServicer, open_lovot_tf_client))
        mock_tf_service.response = SetTransformResponse()

        context.set(Context(
            slam_servicer_client=None,
            wifi_client=None,
            lovot_tf_client=lovot_tf_client,
            localization_client=None,
            fingerprint_sync=MockFingerprintSync(),
            wifi_scan=None,
            radio_map=RadioMap(),
        ))

        yield


async def _generate_radio_map(n: int) -> RadioMap:
    fingerprints = []
    for i in range(n):
        fingerprint = copy.deepcopy(DUMMY_FINGERPRINT)
        fingerprint.stamp.CopyFrom(unix_time_to_pb_timestamp(i))
        fingerprints.append(fingerprint)
    return await RadioMap.from_proto(RadioMap_pb(fingerprints=fingerprints))


async def test_update_target_map(mock_context, nursery, autojump_clock):
    evaluator = InferenceEvaluator()
    nursery.start_soon(evaluator.run)

    # Not updated, because the map is empty
    await trio.sleep(InferenceEvaluator._TARGET_MAP_UPDATE_PERIOD * 2)
    assert evaluator._target_map is None

    # Not updated, because the map is too small
    context.get().radio_map = await _generate_radio_map(RadioMap._MINIMUM_FINGERPRINTS_TO_PREDICT-1)
    await trio.sleep(InferenceEvaluator._TARGET_MAP_UPDATE_PERIOD * 2)
    assert evaluator._target_map is None

    # Updated
    context.get().radio_map = await _generate_radio_map(RadioMap._MINIMUM_FINGERPRINTS_TO_PREDICT+1)
    await trio.sleep(InferenceEvaluator._TARGET_MAP_UPDATE_PERIOD * 2)
    assert evaluator._target_map == context.get().radio_map

    # Updated
    context.get().radio_map = await _generate_radio_map(RadioMap._MINIMUM_FINGERPRINTS_TO_PREDICT+2)
    await trio.sleep(InferenceEvaluator._TARGET_MAP_UPDATE_PERIOD * 2)
    assert evaluator._target_map == context.get().radio_map

    # Not updated, becase of no change
    previous_id = id(evaluator._target_map)
    context.get().radio_map = await _generate_radio_map(RadioMap._MINIMUM_FINGERPRINTS_TO_PREDICT+2)
    await trio.sleep(InferenceEvaluator._TARGET_MAP_UPDATE_PERIOD * 2)
    assert id(evaluator._target_map) == previous_id  # no copy was made
    assert evaluator._target_map == context.get().radio_map

    # Not updated, because the map becomes too small
    context.get().radio_map = await _generate_radio_map(RadioMap._MINIMUM_FINGERPRINTS_TO_PREDICT-1)
    await trio.sleep(InferenceEvaluator._TARGET_MAP_UPDATE_PERIOD * 2)
    assert id(evaluator._target_map) == previous_id


async def test_evaluate(mock_context, nursery, autojump_clock):
    mock_fingerprint_sync = context.get().fingerprint_sync
    evaluator = InferenceEvaluator()
    nursery.start_soon(evaluator.run)

    def copy_fingerprint():
        stamp_message = unix_time_to_pb_timestamp(trio.current_time())
        fingerprint = copy.deepcopy(DUMMY_FINGERPRINT)
        fingerprint.stamp.CopyFrom(stamp_message)
        return fingerprint

    # Not evaluated, because the map is empty
    for _ in range(10):
        fingerprint = copy_fingerprint()
        mock_fingerprint_sync.set_fingerprint(Fingerprint.from_proto(fingerprint))

        await trio.sleep(1.0)
        # The error is not updated
        assert evaluator.average_error() is None

    # Set target map
    context.get().radio_map = await _generate_radio_map(InferenceEvaluator._MINIMUM_FINGERPRINT_SIZE+1)
    await trio.sleep(InferenceEvaluator._TARGET_MAP_UPDATE_PERIOD * 2)
    assert evaluator._target_map == context.get().radio_map

    # Evaluate the tareget map with fingerprints with some positional errors
    expected_error = 1.0
    for _ in range(10):
        fingerprint = copy_fingerprint()
        fingerprint.transform.transform.translation.x = \
            fingerprint.transform.transform.translation.x + expected_error  # with error
        mock_fingerprint_sync.set_fingerprint(Fingerprint.from_proto(fingerprint))

        await trio.sleep(1.0)
        # The error is updated with the expected error
        assert evaluator.average_error() == pytest.approx(expected_error, 0.01)

    # Not evaluated, because the covariance is too large
    for _ in range(10):
        fingerprint = copy_fingerprint()
        fingerprint.transform.covariance.matrix[0] = 3.0  # Too large covariance
        fingerprint.transform.transform.translation.x = \
            fingerprint.transform.transform.translation.x + 100.0  # Large error
        mock_fingerprint_sync.set_fingerprint(Fingerprint.from_proto(fingerprint))

        await trio.sleep(1.0)
        # The error is not updated (the same as the previous one)
        assert evaluator.average_error() == pytest.approx(expected_error, 0.01)
