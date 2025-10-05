import prometheus_client
import pytest

from lovot_slam.env import NestSlamState
from lovot_slam.map_build.map_build_metrics import MapBuildAttemptResultsMetric
from lovot_slam.redis import create_ltm_client

_LTM_CONTINUOUS_FAIL_COUNTS_KEY = 'localization:test:status:counter'


def _unregister_prometheus_collectors():
    collectors = list(prometheus_client.REGISTRY._collector_to_names.keys())
    for collector in collectors:
        prometheus_client.REGISTRY.unregister(collector)


@pytest.fixture
def setup():
    ltm_client = create_ltm_client()
    ltm_client.delete(_LTM_CONTINUOUS_FAIL_COUNTS_KEY)
    yield
    ltm_client.delete(_LTM_CONTINUOUS_FAIL_COUNTS_KEY)

    _unregister_prometheus_collectors()


def test_map_build_attempt_result_metric_fail_count(monkeypatch, setup):
    monkeypatch.setattr(MapBuildAttemptResultsMetric,
                        '_LTM_CONTINUOUS_FAIL_COUNTS_KEY', _LTM_CONTINUOUS_FAIL_COUNTS_KEY)

    metric = MapBuildAttemptResultsMetric()

    assert metric.get_continuous_fail_count(True, NestSlamState.BAG_CONVERSION) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.BUILD_FEATURE_MAP) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.SCALE_MAP) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.BUILD_DENSE_MAP) == 0
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_FEATURE_MAP) == 0
    assert metric.get_continuous_fail_count(False, NestSlamState.SCALE_MAP) == 0
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_DENSE_MAP) == 0
    assert metric.get_total_continuous_fail_count() == 0

    metric.fail(False, NestSlamState.BUILD_FEATURE_MAP)

    assert metric.get_continuous_fail_count(True, NestSlamState.BAG_CONVERSION) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.BUILD_FEATURE_MAP) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.SCALE_MAP) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.BUILD_DENSE_MAP) == 0
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_FEATURE_MAP) == 1
    assert metric.get_continuous_fail_count(False, NestSlamState.SCALE_MAP) == 0
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_DENSE_MAP) == 0
    assert metric.get_total_continuous_fail_count() == 1

    metric.fail(True, NestSlamState.BUILD_FEATURE_MAP)

    assert metric.get_continuous_fail_count(True, NestSlamState.BAG_CONVERSION) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.BUILD_FEATURE_MAP) == 1
    assert metric.get_continuous_fail_count(True, NestSlamState.SCALE_MAP) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.BUILD_DENSE_MAP) == 0
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_FEATURE_MAP) == 1
    assert metric.get_continuous_fail_count(False, NestSlamState.SCALE_MAP) == 0
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_DENSE_MAP) == 0
    assert metric.get_total_continuous_fail_count() == 2

    metric.fail(True, NestSlamState.BUILD_FEATURE_MAP)

    assert metric.get_continuous_fail_count(True, NestSlamState.BAG_CONVERSION) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.BUILD_FEATURE_MAP) == 2
    assert metric.get_continuous_fail_count(True, NestSlamState.SCALE_MAP) == 0
    assert metric.get_continuous_fail_count(True, NestSlamState.BUILD_DENSE_MAP) == 0
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_FEATURE_MAP) == 1
    assert metric.get_continuous_fail_count(False, NestSlamState.SCALE_MAP) == 0
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_DENSE_MAP) == 0
    assert metric.get_total_continuous_fail_count() == 3


def test_map_build_attempt_result_metric_clear_counter(monkeypatch, setup):
    monkeypatch.setattr(MapBuildAttemptResultsMetric, '_LTM_CONTINUOUS_FAIL_COUNTS_KEY',
                        _LTM_CONTINUOUS_FAIL_COUNTS_KEY)

    metric = MapBuildAttemptResultsMetric()

    # clear continuous fail count on reset
    metric.fail(False, NestSlamState.BUILD_FEATURE_MAP)
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_FEATURE_MAP) == 1
    assert metric.get_total_continuous_fail_count() == 1
    metric.reset()
    assert metric.get_total_continuous_fail_count() == 0

    # clear continuous fail count on build success
    metric.fail(True, NestSlamState.BUILD_FEATURE_MAP)
    assert metric.get_continuous_fail_count(True, NestSlamState.BUILD_FEATURE_MAP) == 1
    assert metric.get_total_continuous_fail_count() == 1
    metric.success()
    assert metric.get_total_continuous_fail_count() == 0


def test_map_build_attempt_result_metric_store_and_load(monkeypatch, setup):
    monkeypatch.setattr(MapBuildAttemptResultsMetric, '_LTM_CONTINUOUS_FAIL_COUNTS_KEY',
                        _LTM_CONTINUOUS_FAIL_COUNTS_KEY)

    metric = MapBuildAttemptResultsMetric()

    metric.fail(False, NestSlamState.BUILD_FEATURE_MAP)
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_FEATURE_MAP) == 1
    assert metric.get_total_continuous_fail_count() == 1

    _unregister_prometheus_collectors()

    # re-create instance so that the values are reloaded from LTM
    metric = MapBuildAttemptResultsMetric()
    assert metric.get_continuous_fail_count(False, NestSlamState.BUILD_FEATURE_MAP) == 1
    assert metric.get_total_continuous_fail_count() == 1
