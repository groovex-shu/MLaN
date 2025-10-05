import os
import shutil

import pytest

from lovot_slam.utils.file_util import (zip_archive, unzip_archive, _scan_dir, get_file_md5sum)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, '../../../../', 'dataset')
TEMP_DIR = '/tmp/lovot_slam'


def test_zip_archive():
    base_name = os.path.join(TEMP_DIR, 'test.zip')
    root_dir = os.path.join(DATASET_ROOT, 'dummy', 'maps', '20190714_051701_8d2552ddd47de903deb2b21ef533ea11')
    zip_name = zip_archive(base_name, root_dir)

    assert base_name == zip_name

    origin_items = [os.path.relpath(item, root_dir) for item in _scan_dir(root_dir)]

    target_dir = os.path.join(TEMP_DIR, 'test')
    unzip_archive(zip_name, target_dir)
    destination_items = [os.path.relpath(item, target_dir) for item in _scan_dir(target_dir)]
    assert set(destination_items) == set(origin_items)

    shutil.rmtree(TEMP_DIR)


def test_zip_archive_with_base_dir():
    base_name = os.path.join(TEMP_DIR, 'test.zip')
    root_dir = os.path.join(DATASET_ROOT, 'dummy', 'maps')
    base_dir = '20190714_051701_8d2552ddd47de903deb2b21ef533ea11'
    zip_name = zip_archive(base_name, root_dir, base_dir=base_dir)

    assert base_name == zip_name

    origin_items = [os.path.relpath(item, root_dir) for item in _scan_dir(os.path.join(root_dir, base_dir))]

    target_dir = os.path.join(TEMP_DIR, 'test')
    unzip_archive(zip_name, target_dir)
    destination_items = [os.path.relpath(item, target_dir) for item in _scan_dir(os.path.join(target_dir, base_dir))]
    assert set(destination_items) == set(origin_items)

    shutil.rmtree(TEMP_DIR)


@pytest.mark.parametrize("target_file,expected_md5",
                         [(f"{DATASET_ROOT}/dummy/maps/20190714_044330/2d_map/map.pgm", "9b6ac560cc2da527f0af0bf57bab0822")])
def test_get_md5(target_file, expected_md5):
    actual_md5 = get_file_md5sum(target_file)
    assert actual_md5 == expected_md5
