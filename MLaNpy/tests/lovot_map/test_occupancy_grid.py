import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from grid_map_util.occupancy_grid import OccupancyGrid


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_YAML = os.path.join(BASE_DIR, '2d_map', 'map.yaml')


def test_open_map_yaml():
    oc_grid = OccupancyGrid.from_yaml_file(MAP_YAML)
    assert np.all(oc_grid.origin == np.array([-4.25, -3.55]))
    assert oc_grid.origin_yaw == 0.0
    assert oc_grid.resolution == 0.05


@pytest.mark.parametrize('contents, ftype', [
    ('', ''),  # 'NoneType' object has no attribute 'get'
    ('image: map.pgm\norigin: [0, 0, 0]\nresolution: 0.05', ''),  # Failed to read image file.
    ('image: map.pgm\norigin: []\nresolution: 0.05', ''),  # list index out of range
    ('image: map.pgm\norigin: [0, 0, 0]\nresolution: 0.0', ''),  # Invalid map data with resolution=0.0.
    (bytes([0, 0, 0, 0, 0, 0]), 'b'),  # unacceptable character #x0000: special characters are not allowed
])
def test_open_broken_map_yaml(contents, ftype):
    """不正なファイルの時にOccupancyGrid.from_yaml_fileがRuntimeErrorを投げるかどうかのテスト
    (RuntimeErrorは呼び出し元で処理する前提)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_file = Path(tmpdir) / 'map.yaml'
        with open(yaml_file, f'w{ftype}') as f:
            f.write(contents)

        with pytest.raises(RuntimeError) as e_info:
            _ = OccupancyGrid.from_yaml_file(yaml_file)
        print(e_info)


def test_save_map(tmp_path):
    dest_yaml = os.path.join(tmp_path, 'output', 'map.yaml')
    oc_grid = OccupancyGrid.from_yaml_file(MAP_YAML)
    oc_grid.save(dest_yaml)

    # compare yaml
    with open(MAP_YAML, 'r') as f:
        conf_orig = yaml.safe_load(f)
    with open(dest_yaml, 'r') as f:
        conf_dest = yaml.safe_load(f)
    assert conf_orig == conf_dest

    # compare pgm image
    img_orig = cv2.imread(os.path.join(os.path.dirname(
        MAP_YAML), conf_orig['image']), cv2.IMREAD_GRAYSCALE)
    img_dest = cv2.imread(os.path.join(os.path.dirname(
        MAP_YAML), conf_orig['image']), cv2.IMREAD_GRAYSCALE)
    assert (img_orig == img_dest).all()


def test_area():
    oc_grid = OccupancyGrid.from_yaml_file(MAP_YAML)

    assert oc_grid.get_area_meter_square() == (147 * 205 * 0.05 * 0.05)
    assert oc_grid.get_area_pixel_square() == (147 * 205)
    assert oc_grid.get_free_area() == 6756 * 0.05 * 0.05
    assert oc_grid.get_occupied_area() == 3872 * 0.05 * 0.05
    assert oc_grid.get_unknown_area() == 19507 * 0.05 * 0.05


def test_area_zero():
    oc_grid = OccupancyGrid(np.empty((0, 0)), 0.05, np.array([0., 0.]))

    assert oc_grid.get_area_meter_square() == 0
    assert oc_grid.get_area_pixel_square() == 0


def test_area_invalid_map():
    oc_grid = OccupancyGrid(np.zeros((0)), 0.05, np.array([0., 0.]))

    with pytest.raises(RuntimeError, match="invalid map image"):
        oc_grid.get_area_meter_square()
    with pytest.raises(RuntimeError, match="invalid map image"):
        oc_grid.get_area_pixel_square()


def test_get_nearest_free_cell():
    img = np.full([50, 50], OccupancyGrid.OCCUPIED_CODE)
    # 40th rows and columns respectively from left/bottom corner (40 * 0.05 = 2.0)
    img[50 - 40 - 1, 40] = OccupancyGrid.FREE_CODE
    oc_grid = OccupancyGrid(img, 0.05, np.array([0., 0.]))

    start = np.array([0.1, 0.1])
    assert np.all(np.isclose(oc_grid.get_nearest_free_cell(start), np.array([2., 2.])))


@pytest.mark.parametrize('realcoords,cvcoords', [
    (np.array([0.0, 0.0]), np.array([2, 2]).astype(np.int32)),
    (np.array([-0.10, -0.10]), np.array([0, 4]).astype(np.int32)),
    (np.array([-0.124, -0.124]), np.array([0, 4]).astype(np.int32)),
    (np.array([-0.076, -0.076]), np.array([0, 4]).astype(np.int32)),
])
def test_realcoords_to_cvcoords(realcoords, cvcoords):
    origin = np.array((-0.10, -0.10))
    resolution = 0.05
    grid_map = OccupancyGrid(np.zeros([5, 5]), resolution, origin)

    obtained_cvcoords = grid_map.realcoords_to_cvcoords(realcoords)
    assert np.array_equal(obtained_cvcoords, cvcoords)


@pytest.mark.parametrize('realcoords,npcoords', [
    (np.array([0.0, 0.0]), np.array([2, 2]).astype(np.int32)),
    (np.array([-0.10, -0.10]), np.array([4, 0]).astype(np.int32)),
    (np.array([-0.124, -0.124]), np.array([4, 0]).astype(np.int32)),
    (np.array([-0.076, -0.076]), np.array([4, 0]).astype(np.int32)),
])
def test_realcoords_to_npcoords(realcoords, npcoords):
    origin = np.array((-0.10, -0.10))
    resolution = 0.05
    grid_map = OccupancyGrid(np.zeros([5, 5]), resolution, origin)

    obtained_npcoords = grid_map.realcoords_to_npcoords(realcoords)
    assert np.array_equal(obtained_npcoords, npcoords)


@pytest.mark.parametrize('realcoords,npcoords', [
    (np.array([0.0, 0.0]), np.array([2, 2]).astype(np.int32)),
    (np.array([-0.10, -0.10]), np.array([4, 0]).astype(np.int32)),
])
def test_npcoords_to_realcoords(realcoords, npcoords):
    origin = np.array((-0.10, -0.10))
    resolution = 0.05
    grid_map = OccupancyGrid(np.zeros([5, 5]), resolution, origin)

    obtained_realcoords = grid_map.npcoords_to_realcoords(npcoords)
    assert np.all(np.isclose(obtained_realcoords, realcoords))


@pytest.mark.parametrize('radius,correct_kernel_str', [
    (0.05,
     "010"
     "111"
     "010"),
    (0.074,
     "010"
     "111"
     "010"),
    (0.10,
     "00100"
     "01110"
     "11111"
     "01110"
     "00100"),
])
def test_create_kernel(radius, correct_kernel_str):
    resolution = 0.05

    width = int(len(correct_kernel_str) ** 0.5 + 0.5)
    correct_kernel = [int(ch) for ch in list(correct_kernel_str)]
    correct_kernel = np.array(correct_kernel).astype(np.uint8).reshape((width, width))

    grid_map = OccupancyGrid(np.zeros([2, 2]), resolution, np.array([0.0, 0.0]))
    kernel = grid_map._create_kernel(radius)

    assert np.array_equal(kernel, correct_kernel)


def _generate_occupancy_grid(data_str, width, height, resolution, origin):
    """
    0: Free
    1: Unknown
    2: Occupied
    """
    dec = {'0': 254, '1': 205, '2': 0}
    data = [dec[ch] for ch in list(data_str)]
    data = np.array(data).astype(np.uint8).reshape((height, width))
    return OccupancyGrid(data, resolution, origin)


@pytest.mark.parametrize('original,radius,expected', [
    (
        # no change (all free)
        _generate_occupancy_grid(
            '000000'
            '000000'
            '000000'
            '000000'
            '000000'
            '000000',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        0.2,
        _generate_occupancy_grid(
            '000000'
            '000000'
            '000000'
            '000000'
            '000000'
            '000000',
            6, 6, 0.05, np.array([-0.10, -0.10])
        ),
    ),
    (
        # no change (all unknown)
        _generate_occupancy_grid(
            '111111'
            '111111'
            '111111'
            '111111'
            '111111'
            '111111',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        0.2,
        _generate_occupancy_grid(
            '111111'
            '111111'
            '111111'
            '111111'
            '111111'
            '111111',
            6, 6, 0.05, np.array([-0.10, -0.10])
        ),
    ),
    (
        # no change (all obstacle)
        _generate_occupancy_grid(
            '222222'
            '222222'
            '222222'
            '222222'
            '222222'
            '222222',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        0.2,
        _generate_occupancy_grid(
            '222222'
            '222222'
            '222222'
            '222222'
            '222222'
            '222222',
            6, 6, 0.05, np.array([-0.10, -0.10])
        ),
    ),
    (
        # center void will disappear
        _generate_occupancy_grid(
            '222222'
            '200002'
            '201102'
            '201102'
            '200002'
            '222222',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        0.05,
        _generate_occupancy_grid(
            '222222'
            '200002'
            '200002'
            '200002'
            '200002'
            '222222',
            6, 6, 0.05, np.array([-0.10, -0.10])
        ),
    ),
    (
        # center void will become smaller
        _generate_occupancy_grid(
            '000000'
            '011110'
            '011110'
            '011110'
            '011110'
            '000000',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        0.05,
        _generate_occupancy_grid(
            '000000'
            '001100'
            '011110'
            '011110'
            '001100'
            '000000',
            6, 6, 0.05, np.array([-0.10, -0.10])
        ),
    ),
    (
        # border will not change
        _generate_occupancy_grid(
            '000000'
            '000000'
            '000000'
            '111111'
            '111111'
            '111111',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        0.05,
        _generate_occupancy_grid(
            '000000'
            '000000'
            '000000'
            '111111'
            '111111'
            '111111',
            6, 6, 0.05, np.array([-0.10, -0.10])
        ),
    ),
    (
        # void at the edge will become smaller
        _generate_occupancy_grid(
            '000000'
            '000000'
            '000000'
            '001100'
            '001100'
            '001100',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        0.05,
        _generate_occupancy_grid(
            '000000'
            '000000'
            '000000'
            '000000'
            '000000'
            '000000',
            6, 6, 0.05, np.array([-0.10, -0.10])
        ),
    ),
])
def test_erode_unknown(original, radius, expected):
    original.erode_unknown(radius)
    assert np.array_equal(original.img, expected.img)


@pytest.mark.parametrize('original,point_of_interest,radius,expected', [
    (
        # no change (all free)
        _generate_occupancy_grid(
            '000000'
            '000000'
            '000000'
            '000000'
            '000000'
            '000000',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        np.array([0., 0.]),
        0.2,
        _generate_occupancy_grid(
            '000000'
            '000000'
            '000000'
            '000000'
            '000000'
            '000000',
            6, 6, 0.05, np.array([-0.10, -0.10])
        ),
    ),
    (
        # ROI is on the top-left corner
        # so only the free area on the top-left corner will be closed
        _generate_occupancy_grid(
            '0000111111'  # <- ROI (0, 0)
            '0110111111'
            '0110111111'
            '0000111111'
            '1111111111'
            '1111100000'
            '1111101110'
            '1111101110'
            '1111101110'
            '1111100000',
            10, 10, 0.05, np.array([0.0, -9 * 0.05])),
        np.array([0., 0.]),
        0.2,
        _generate_occupancy_grid(
            '0000111111'
            '0000111111'  # <- closed
            '0000111111'  # <- closed
            '0000111111'
            '1111111111'
            '1111100000'
            '1111101110'  # <- remain open
            '1111101110'  # <- remain open
            '1111101110'  # <- remain open
            '1111100000',
            10, 10, 0.05, np.array([0.0, -9 * 0.05])),
    ),
    (
        # ROI is on the top-left corner
        # so only the free area on the bottom-right corner will be closed
        _generate_occupancy_grid(
            '0000111111'
            '0110111111'
            '0110111111'
            '0000111111'
            '1111111111'
            '1111100000'
            '1111101110'
            '1111101110'
            '1111101110'
            '1111100000',  # <- ROI (9, 9)
            10, 10, 0.05, np.array([0.0, -9 * 0.05])),
        np.array([9 * 0.05, -9 * 0.05]),
        0.2,
        _generate_occupancy_grid(
            '0000111111'
            '0110111111'  # <- remain open
            '0110111111'  # <- remain open
            '0000111111'
            '1111111111'
            '1111100000'
            '1111100000'  # <- closed
            '1111100000'  # <- closed
            '1111100000'  # <- closed
            '1111100000',
            10, 10, 0.05, np.array([0.0, -9 * 0.05])),
    ),
])
def test_apply_closing_unknown(original, point_of_interest, radius, expected):
    ret = original.apply_closing_unknown(point_of_interest, radius)
    assert ret
    assert np.array_equal(original.img, expected.img)


@pytest.mark.parametrize('grid_map,point_of_interest', [
    (
        # no change (all unknown)
        _generate_occupancy_grid(
            '111111'
            '111111'
            '111111'
            '111111'
            '111111'
            '111111',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        np.array([0., 0.]),
    ),
    (
        # no change (all occupied)
        _generate_occupancy_grid(
            '222222'
            '222222'
            '222222'
            '222222'
            '222222'
            '222222',
            6, 6, 0.05, np.array([-0.10, -0.10])),
        np.array([0., 0.]),
    ),
])
def test_apply_closing_unknown_failure(grid_map, point_of_interest):
    original_img = grid_map.img.copy()
    radius = 0.2
    ret = grid_map.apply_closing_unknown(point_of_interest, radius)
    assert not ret
    assert np.array_equal(grid_map.img, original_img)
