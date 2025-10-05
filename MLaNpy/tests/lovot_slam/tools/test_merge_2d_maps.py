import numpy as np
import pytest

from lovot_slam.tools.merge_2d_maps import (FREE_PROBABILITIES, OCCUPIED_PROBABILITY, MapType, CostMapsMerger,
                                            convert_to_cost_map, convert_to_occupancy_grid)
from MLaNpy.lovot_map.rosmap import RosMap

# 0: free, 1: unknown, 2: occupied
UNSHIFTED_MAP_IMAGES = {
    "unobserved": "11111"
                  "11111"
                  "11111"
                  "11111"
                  "11111",
    "close":      "11111"
                  "22221"
                  "20021"
                  "20021"
                  "22221",  # <- door close
    "open":       "11111"
                  "22221"
                  "20021"
                  "20021"
                  "20021",  # <- door open
    "path":       "11111"
                  "11111"
                  "10011"
                  "10011"
                  "10011",
}
UNSHIFTED_ORIGIN = "-0.15, -0.15, 0, 0, 0, 0, 1"

# 0: free, 1: unknown, 2: occupied
SHIFTED_MAP_IMAGES = {
    "unobserved": "11111"
                  "11111"
                  "11111"
                  "11111"
                  "11111",
    "close":      "12222"
                  "12002"
                  "12002"
                  "12222"  # <- door close
                  "11111",
    "open":       "12222"
                  "12002"
                  "12002"
                  "12002"  # <- door open
                  "11111",
}
SHIFTED_ORIGIN = "-0.2, -0.1, 0, 0, 0, 0, 1"

MERGED_WITH_SHIFTED_MAP = RosMap.from_hashed_values(
    ("6", "6", "0.05",  # width, height, resolution
     "111111"
     "122221"
     "120021"
     "120021"
     "120021"
     "111111",
     "-0.2, -0.15, 0, 0, 0, 0, 1",  # origin
     "name")
)


# パラメータ (FREE_PROBABILITIES, OCCUPIED_PROBABILITY)で結果が変わるため、
# パラメータ調整を行った際にテスト結果を修正する必要あり (ただし、結果が妥当かを確認した上で行うこと)
@pytest.mark.parametrize("unobserved_count,close_count,open_count,path_count,correct_image", [
    (1, 0, 0, 0, UNSHIFTED_MAP_IMAGES["unobserved"]),  # 観測なし
    (1, 1, 0, 0, UNSHIFTED_MAP_IMAGES["close"]),  # occupiedの観測のみ
    (1, 0, 1, 0, UNSHIFTED_MAP_IMAGES["open"]),  # freeの観測のみ
    (1, 1, 1, 0, UNSHIFTED_MAP_IMAGES["open"]),
    (1, 2, 1, 0, UNSHIFTED_MAP_IMAGES["open"]),
    (1, 3, 1, 0, UNSHIFTED_MAP_IMAGES["close"]),
    (1, 1, 0, 1, UNSHIFTED_MAP_IMAGES["close"]),
    (1, 1, 0, 2, UNSHIFTED_MAP_IMAGES["open"]),
    (1, 3, 1, 1, UNSHIFTED_MAP_IMAGES["open"]),
])
def test_merge_unshifted(unobserved_count, close_count, open_count, path_count, correct_image):
    width, height, resolution, origin, name = "5", "5", "0.05", UNSHIFTED_ORIGIN, "none"

    # create dense maps (aka 2d_map)
    images = [UNSHIFTED_MAP_IMAGES["unobserved"]] * unobserved_count \
        + [UNSHIFTED_MAP_IMAGES["close"]] * close_count \
        + [UNSHIFTED_MAP_IMAGES["open"]] * open_count
    grid_maps = [RosMap.from_hashed_values((width, height, resolution, image, origin, name)).as_occupancy_grid()
                 for image in images]
    dense_maps = [convert_to_cost_map(grid_map,
                                      occupied_probability=OCCUPIED_PROBABILITY,
                                      free_probability=FREE_PROBABILITIES[MapType.DENSE_MAP]) for grid_map in grid_maps]

    # create path maps
    images = [UNSHIFTED_MAP_IMAGES["path"]] * path_count
    grid_maps = [RosMap.from_hashed_values((width, height, resolution, image, origin, name)).as_occupancy_grid()
                 for image in images]
    path_maps = [convert_to_cost_map(grid_map,
                                     occupied_probability=OCCUPIED_PROBABILITY,
                                     free_probability=FREE_PROBABILITIES[MapType.PATH_MAP]) for grid_map in grid_maps]

    # merge
    merger = CostMapsMerger(dense_maps, path_maps)
    merged_map = convert_to_occupancy_grid(merger.merge())

    correct_map = RosMap.from_hashed_values(
        (width, height, resolution, correct_image, origin, name)).as_occupancy_grid()

    assert correct_map.resolution == merged_map.resolution
    assert np.all(np.isclose(correct_map.origin, merged_map.origin))
    assert np.array_equal(correct_map.img, merged_map.img)


def test_merge_with_shifted():
    # maps with unshifted origin
    images = [UNSHIFTED_MAP_IMAGES["unobserved"],
              UNSHIFTED_MAP_IMAGES["close"],
              UNSHIFTED_MAP_IMAGES["open"]]
    width, height, resolution, origin, name = "5", "5", "0.05", UNSHIFTED_ORIGIN, "none"
    grid_maps = [RosMap.from_hashed_values((width, height, resolution, image, origin, name)).as_occupancy_grid()
                 for image in images]

    # maps with shifted origin
    images = [SHIFTED_MAP_IMAGES["unobserved"],
              SHIFTED_MAP_IMAGES["close"],
              SHIFTED_MAP_IMAGES["open"]]
    width, height, resolution, origin, name = "5", "5", "0.05", SHIFTED_ORIGIN, "none"
    grid_maps += [RosMap.from_hashed_values((width, height, resolution, image, origin, name)).as_occupancy_grid()
                  for image in images]

    dense_maps = [convert_to_cost_map(grid_map,
                                      occupied_probability=OCCUPIED_PROBABILITY,
                                      free_probability=FREE_PROBABILITIES[MapType.DENSE_MAP]) for grid_map in grid_maps]

    # merge
    merger = CostMapsMerger(dense_maps, [])
    merged_map = convert_to_occupancy_grid(merger.merge())

    correct_map = MERGED_WITH_SHIFTED_MAP.as_occupancy_grid()

    assert correct_map.resolution == merged_map.resolution
    assert np.all(np.isclose(correct_map.origin, merged_map.origin))
    assert np.array_equal(correct_map.img, merged_map.img)
