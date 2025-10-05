import copy
import os
import pathlib
import shutil
import time

import numpy as np
import pytest

from grid_map_util.accuracy_map import AccuracyMap, logarithm_probability, unlogarithm_probability
from grid_map_util.occupancy_grid import OccupancyGrid

from lovot_slam.env import DataDirectories, data_directories
from lovot_slam.exploration.accuracy_map_util import load_accuracy_map, load_accuracy_maps_and_merge, match_map_size

DataDirectories.DATA_ROOT = pathlib.Path('/tmp/lovot-localization/test')


def _discretize_data(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """data (dtype == np.float64)を8bit or 16bitで量子化後にnp.float64に戻す処理
    ファイルにsaveしたarrayを再度読み込んで、オリジナルと比較するテストがあるが、ファイル化する際の量子化誤差を吸収するために使う
    :param data: 元のarray (np.float64)
    :param dtype: 量子化のターゲット (np.uint8 or np.uint16)
    :return: 量子化した後再度np.float化したarray
    """
    if dtype == np.uint8:
        data = np.round(unlogarithm_probability(data) * 254).astype(np.uint8)
        return logarithm_probability(data.astype(np.float64) / 254)
    elif dtype == np.uint16:
        data = np.round(unlogarithm_probability(data) * 65534).astype(np.uint16)
        return logarithm_probability(data.astype(np.float64) / 65534)
    raise RuntimeError(f'Invalid dtype: {dtype}')


def _decode_text_accuracy_map(text_map: str, width: int, height: int,
                              origin: np.ndarray, resolution: float, dtype: np.dtype) -> AccuracyMap:
    data = [int(ch) / 10. for ch in list(text_map)]
    data = np.array(data).reshape((height, width))
    data = np.flipud(data)
    data = logarithm_probability(data)
    data = _discretize_data(data, dtype)
    return AccuracyMap(data, origin, resolution)


# represents the first decimal place of the probability (0.0 ~ 0.9)
ACCURACY_MAPS = {
    'accuracy_map_a': _decode_text_accuracy_map(
        "33333"
        "33333"
        "33333"
        "33333"
        "33333",
        5,
        5,
        np.array((-0.15, -0.15)),
        0.05,
        np.uint16
    ),
    'accuracy_map_b': _decode_text_accuracy_map(
        "77777"
        "77777"
        "77777"
        "77777"
        "77777",
        5,
        5,
        np.array((0., 0.)),
        0.05,
        np.uint16
    ),
    # this map doesn't affect to sum
    'zero': _decode_text_accuracy_map(
        "55555"
        "55555"
        "55555"
        "55555"
        "55555",
        5,
        5,
        np.array((-0.1, -0.1)),
        0.05,
        np.uint16
    ),
    'sum': _decode_text_accuracy_map(
        # upper right corner: accuracy_map_b
        # lower left corner: accuracy_map_a
        # center 2x2 pixels: average of a and b = 5
        "55577777"
        "55577777"
        "55577777"
        "33355777"
        "33355777"
        "33333555"
        "33333555"
        "33333555",
        8,
        8,
        np.array((-0.15, -0.15)),
        0.05,
        np.uint16
    ),
}
ACUURACY_MAP_8BIT_DESCRETIZED = _decode_text_accuracy_map(
    "77777"
    "77777"
    "77777"
    "77777"
    "77777",
    5,
    5,
    np.array((0., 0.)),
    0.05,
    np.uint8
)


@pytest.fixture
def mock_accuracy_maps(monkeypatch):
    monkeypatch.setattr('lovot_slam.env.DataDirectories.DATA_ROOT', data_directories.data_root)
    yaml_files = []
    for ghost_id, accuracy_map in ACCURACY_MAPS.items():
        grid_map = accuracy_map.to_occupancy_grid()
        if ghost_id == 'sum':
            map_yaml = data_directories.monitor / 'accuracy_map' / 'map.yaml'
        else:
            map_yaml = data_directories.monitor / ghost_id / 'accuracy_map' / 'map.yaml'
        grid_map.save(map_yaml)
        yaml_files.append(map_yaml)
    yield yaml_files
    shutil.rmtree(data_directories.data_root)


@pytest.fixture
def mock_8bit_accuracy_map(monkeypatch):
    monkeypatch.setattr('lovot_slam.env.DataDirectories.DATA_ROOT', data_directories.data_root)

    data = ACUURACY_MAP_8BIT_DESCRETIZED.data
    img = np.round((unlogarithm_probability(data) - 0.5) * 254).astype(np.uint8) + 127
    img = np.flipud(img)
    grid_map = OccupancyGrid(img, ACUURACY_MAP_8BIT_DESCRETIZED.resolution, ACUURACY_MAP_8BIT_DESCRETIZED.origin)

    map_yaml = data_directories.monitor / 'accuracy_map' / 'map.yaml'
    grid_map.save(map_yaml)
    yield
    shutil.rmtree(data_directories.data_root)


def test_match_map_size():
    map_a = copy.copy(ACCURACY_MAPS['accuracy_map_a'])
    map_b = copy.copy(ACCURACY_MAPS['accuracy_map_b'])
    match_map_size(map_a, map_b)

    map_c = copy.copy(ACCURACY_MAPS['sum'])
    assert map_a.data.shape == map_b.data.shape == map_c.data.shape
    assert np.all(np.isclose(map_a.origin, map_c.origin))
    assert np.all(np.isclose(map_b.origin, map_c.origin))


@pytest.mark.parametrize('ghost_id,accuracy_map', [
    ('accuracy_map_a', ACCURACY_MAPS['accuracy_map_a']),
    ('accuracy_map_b', ACCURACY_MAPS['accuracy_map_b']),
    ('zero', ACCURACY_MAPS['zero']),
    ('', ACCURACY_MAPS['sum']),
])
def test_load_accuracy_map(mock_accuracy_maps, ghost_id, accuracy_map):
    if ghost_id:
        loaded_accuracy_map = load_accuracy_map(ghost_id)
    else:
        loaded_accuracy_map = load_accuracy_map()

    assert loaded_accuracy_map
    assert np.all(np.isclose(loaded_accuracy_map.data, accuracy_map.data))
    assert np.all(loaded_accuracy_map.origin == accuracy_map.origin)
    assert loaded_accuracy_map.resolution == accuracy_map.resolution

    if ghost_id == 'zero':
        assert np.all(loaded_accuracy_map.data == 0.0)


def test_load_8bit_accuracy_map(mock_8bit_accuracy_map):
    """8bitで保存されたAccuracyMapを読み込めるかのテスト
    16bit化する前のバージョン(<=1.0.0)で保存したデータの互換性担保のため
    """
    loaded_accuracy_map = load_accuracy_map()
    assert loaded_accuracy_map
    assert np.all(np.isclose(loaded_accuracy_map.data, ACUURACY_MAP_8BIT_DESCRETIZED.data))
    assert np.all(loaded_accuracy_map.origin == ACUURACY_MAP_8BIT_DESCRETIZED.origin)
    assert loaded_accuracy_map.resolution == ACUURACY_MAP_8BIT_DESCRETIZED.resolution


@pytest.mark.parametrize('ghost_id,accuracy_map,elapsed_time,loaded', [
    ('accuracy_map_a', ACCURACY_MAPS['accuracy_map_a'], 23*60*60, True),
    ('accuracy_map_a', ACCURACY_MAPS['accuracy_map_a'], 25*60*60, False),
])
def test_load_accuracy_map_ignore_old_file(mock_accuracy_maps, ghost_id, accuracy_map, elapsed_time, loaded):
    # modify modification timestamp of the map yaml files
    yaml_files = mock_accuracy_maps
    mod_time = time.time() - elapsed_time
    for yaml_file in yaml_files:
        os.utime(yaml_file, (mod_time, mod_time))

    loaded_accuracy_map = load_accuracy_map(ghost_id)

    if loaded:
        assert loaded_accuracy_map
        assert np.all(np.isclose(loaded_accuracy_map.data, accuracy_map.data))
        assert np.all(loaded_accuracy_map.origin == accuracy_map.origin)
        assert loaded_accuracy_map.resolution == accuracy_map.resolution
    else:
        assert loaded_accuracy_map is None


@pytest.mark.parametrize('ghost_ids,accuracy_map', [
    (['accuracy_map_a'], ACCURACY_MAPS['accuracy_map_a']),
    (['accuracy_map_b'], ACCURACY_MAPS['accuracy_map_b']),
    (['accuracy_map_a', 'accuracy_map_b'], ACCURACY_MAPS['sum']),
])
def test_load_accuracy_maps_and_merge(mock_accuracy_maps, ghost_ids, accuracy_map):
    loaded_accuracy_map = load_accuracy_maps_and_merge(ghost_ids)

    assert loaded_accuracy_map
    assert np.all(np.isclose(loaded_accuracy_map.data, accuracy_map.data, atol=1./127))
    assert np.all(loaded_accuracy_map.origin == accuracy_map.origin)
    assert loaded_accuracy_map.resolution == accuracy_map.resolution
