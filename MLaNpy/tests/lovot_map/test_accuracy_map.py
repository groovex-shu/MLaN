import numpy as np
import pytest

from grid_map_util.accuracy_map import AccuracyMap, CostMap, logarithm_probability, unlogarithm_probability

RESOLUTION = 0.05


@pytest.mark.parametrize("unlog,log", [
    (0.5, 0.0),
    (0.0, -13.815509557963773),
    (1.0, 13.815509557963773),
    (np.full((10, 10), 0.5), np.full((10, 10), 0.0)),
])
def test_logarithm_probability(unlog, log):
    obtained = logarithm_probability(unlog)
    assert np.all(np.isclose(obtained, log))


@pytest.mark.parametrize("unlog,log", [
    (0.5, 0.0),
    (0.0, -13.815509557963773),
    (1.0, 13.815509557963773),
    (np.full((10, 10), 0.5), np.full((10, 10), 0.0)),
])
def test_unlogarithm_probability(unlog, log):
    obtained = unlogarithm_probability(log)
    assert np.all(np.isclose(obtained, unlog, atol=1e-5))


@pytest.mark.parametrize("data,origin,position,point", [
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     np.array((0, 0)), np.array((5, 5), dtype=np.int32)),
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     np.array((0.05, 0.05)), np.array((6, 6), dtype=np.int32)),
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     np.array((0.1, 0.1)), np.array((7, 7), dtype=np.int32)),
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     np.array((-0.05, 0.1)), np.array((7, 4), dtype=np.int32)),
])
def test_world_to_map(data, origin, position, point):
    cost_map = CostMap(data, origin, RESOLUTION)
    obtained_point = cost_map.world_to_map(position)
    assert np.all(obtained_point == point)


@pytest.mark.parametrize("data,origin,position,point", [
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     np.array((0, 0)), np.array((5, 5), dtype=np.int32)),
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     np.array((0.05, 0.05)), np.array((6, 6), dtype=np.int32)),
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     np.array((0.1, 0.1)), np.array((7, 7), dtype=np.int32)),
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     np.array((-0.05, 0.1)), np.array((7, 4), dtype=np.int32)),
])
def test_map_to_world(data, origin, position, point):
    cost_map = CostMap(data, origin, RESOLUTION)
    obtained_position = cost_map.map_to_world(point)
    assert np.all(np.isclose(obtained_position, position))


@pytest.mark.parametrize("data,origin,boundary", [
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)), (-.25, -.25, .25, .25)),
    (np.zeros((11, 12)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)), (-.25, -.25, .35, .30)),
    (np.zeros((11, 12)), np.array((-RESOLUTION * 7, -RESOLUTION * 6)), (-.35, -.30, .25, .25)),
])
def test_get_boundary(data, origin, boundary):
    cost_map = CostMap(data, origin, RESOLUTION)
    obtained_boundary = cost_map.get_boundary()
    assert np.all(np.isclose(obtained_boundary, boundary))


@pytest.mark.parametrize("data,origin,new_boundary,new_shape,new_origin", [
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     (-.25, -.25, .25, .25), (10, 10), np.array((-RESOLUTION * 5, -RESOLUTION * 5))),
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     (-.25, -.25, .35, .30), (11, 12), np.array((-RESOLUTION * 5, -RESOLUTION * 5))),
    (np.zeros((10, 10)), np.array((-RESOLUTION * 5, -RESOLUTION * 5)),
     (-.35, -.30, .25, .25), (11, 12), np.array((-RESOLUTION * 7, -RESOLUTION * 6))),
])
def test_extend_including(data, origin, new_boundary, new_shape, new_origin):
    cost_map = CostMap(data, origin, RESOLUTION)
    cost_map.extend_including(*new_boundary)
    assert cost_map._data.shape[0] == new_shape[0]
    assert cost_map._data.shape[1] == new_shape[1]
    assert np.all(np.isclose(cost_map._origin, new_origin))


@pytest.mark.parametrize("probabilities,resulted_value", [
    ([0.5], 0.0),
    ([0.8], np.log(0.8 / (1 - 0.8))),
    ([0.8, 0.2], 0.0),
    ([0.8, 0.8], 2 * np.log(0.8 / (1 - 0.8))),
    ([0.05], 0.0),  # ignore probability lower than limit
    ([0.999], np.log(0.99 / (1 - 0.99))),  # clipped at upper limit
])
def test_accuracy_map_update(probabilities, resulted_value):
    data = np.zeros((100, 100))
    origin = np.array((-2.5, -2.5))
    resolution = 0.05
    accuracy_map = AccuracyMap(data, origin, resolution)

    position = np.array((0., 0.))
    radius = 0.1
    for probability in probabilities:
        accuracy_map.update(position, probability, radius)

    assert np.isclose(accuracy_map.data[50, 50], resulted_value)


@pytest.mark.parametrize("initial_value,time_elapsed,resulted_value", [
    (0.0, 24 * 60 * 60, 0.0),
    (1.0, 24 * 60 * 60, 0.5),  # decayed by half in 24 hours
    (-1.0, 24 * 60 * 60, -0.5),  # decayed by half in 24 hours
])
def test_accuracy_map_decay(initial_value, time_elapsed, resulted_value):
    data = np.full((100, 100), initial_value)
    origin = np.array((-2.5, -2.5))
    resolution = 0.05
    decay_half_life = 24 * 60 * 60  # decayed by half in 24 hours
    accuracy_map = AccuracyMap(data, origin, resolution,
                               decay_half_life=decay_half_life)

    timestamp = 100
    accuracy_map.decay(timestamp)
    accuracy_map.decay(timestamp + time_elapsed)

    assert np.isclose(accuracy_map.data[50, 50], resulted_value)


@pytest.mark.parametrize("value,average", [
    (0.0, 0.5),
    (-1.0, np.exp(-1.0) / (1 + np.exp(-1.0))),
    (1.0, np.exp(1.0) / (1 + np.exp(1.0))),
])
def test_accuracy_map_average(value, average):
    data = np.full((100, 100), value)
    origin = np.array((-2.5, -2.5))
    resolution = 0.05
    accuracy_map = AccuracyMap(data, origin, resolution)

    assert np.isclose(accuracy_map.average(), average)


@pytest.mark.parametrize("value,correct_hist", [
    (0.0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (-1.0, [0, 0, 10000, 0, 0, 0, 0, 0, 0, 0]),
    (1.0, [0, 0, 0, 0, 0, 0, 0, 10000, 0, 0]),
])
def test_accuracy_map_histogram(value, correct_hist):
    data = np.full((100, 100), value)
    origin = np.array((-2.5, -2.5))
    resolution = 0.05
    accuracy_map = AccuracyMap(data, origin, resolution)

    bucket, hist = accuracy_map.histogram(11)

    assert np.all(np.isclose(bucket, [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]))
    assert np.all(np.isclose(hist, correct_hist))


@pytest.mark.parametrize("value_a,value_b,sum_value", [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (-1.0, 1.0, 0.0),
])
def test_accuracy_map_add(value_a, value_b, sum_value):
    data_a = np.full((100, 100), value_a)
    data_b = np.full((100, 100), value_b)
    origin = np.array((-2.5, -2.5))
    resolution = 0.05
    accuracy_map_a = AccuracyMap(data_a, origin, resolution)
    accuracy_map_b = AccuracyMap(data_b, origin, resolution)

    added_map = accuracy_map_a + accuracy_map_b

    assert np.isclose(added_map.data[0, 0], sum_value)
