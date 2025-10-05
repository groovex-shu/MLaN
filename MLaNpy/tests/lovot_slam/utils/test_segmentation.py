from logging import getLogger

from mock import Mock
import numpy as np
import pytest

from MLaNpy.lovot_map.rosmap import RosMap
from lovot_slam.utils.segmentation import \
    Section, Segmentation, NO_OBSTACLE, UNKNOWN, OBSTACLE, rebuild_segmentation


_logger = getLogger(__name__)


def _create_rosmap_mock(mapdata):
    name = "dummy rosmap"
    data = mapdata.flatten()
    height, width = mapdata.shape
    origin_pos_2d = (0, 0)
    origin_yaw = 0
    resolution = .05

    rosmap = Mock(RosMap, data=data, width=width, height=height,
                  origin_pos_2d=origin_pos_2d, origin_yaw=origin_yaw, resolution=resolution)
    rosmap.name = name
    return rosmap


def _make_random_map():
    mapdata = np.full((400, 300), UNKNOWN, dtype=np.uint8)

    for _ in range(np.random.randint(3, 6)):
        w = np.random.randint(100, 300)
        h = np.random.randint(100, 200)
        x = np.random.randint(mapdata.shape[0] - w)
        y = np.random.randint(mapdata.shape[1] - h)
        mapdata[x:(x + w), y:(y + h)] = NO_OBSTACLE

    for _ in range(np.random.randint(2, 9)):
        w = np.random.randint(50, 120)
        h = np.random.randint(50, 120)
        x = np.random.randint(mapdata.shape[0] - w)
        y = np.random.randint(mapdata.shape[1] - h)
        mapdata[x:(x + w), y:(y + h)] = OBSTACLE

    return mapdata


def _preview_mapdata(mapdata):
    rep = {NO_OBSTACLE: '.', UNKNOWN: '?', OBSTACLE: '@'}
    return '\n'.join([''.join([rep[cell] for cell in row]) for row in mapdata])


@pytest.mark.parametrize('fill_value, size, expected_nsection', [
    [NO_OBSTACLE, (60, 60), 1],
    [NO_OBSTACLE, (150, 150), 5],
    [UNKNOWN, (100, 100), 0],
    [OBSTACLE, (100, 100), 0],
])
def test_blank_map(fill_value, size, expected_nsection):
    mapdata = np.full(size, fill_value, dtype=np.uint8)
    segmentation = rebuild_segmentation(_create_rosmap_mock(mapdata), None)
    assert len(segmentation.sections) == expected_nsection


@pytest.mark.parametrize('seed', [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 51])
def test_random_map(seed):
    np.random.seed(seed)
    mapdata = _make_random_map()
    map_preview = _preview_mapdata(mapdata)
    _logger.debug(f"sample map:\n{map_preview}")
    segmentation = rebuild_segmentation(_create_rosmap_mock(mapdata), None)

    # cost_from_hubs validataion
    cost_from_hubs = segmentation.cost_from_hubs.copy()
    for hub in segmentation.hubs:
        assert cost_from_hubs[hub.subindex][hub.position] == 0
        cost_from_hubs[hub.subindex][hub.position] = np.nan
    assert not (cost_from_hubs == 0).any()

    # highways varidation
    for highway in segmentation.highways:
        assert segmentation.hubs[highway.start_hub_index].position == highway.points[0]
        assert segmentation.hubs[highway.end_hub_index].position == highway.points[-1]
        assert highway.cost > 0
        assert highway.cost <= 2 ** .5 or len(highway.points) > 2

    # precomputed_paths validation
    for ((start_hub_index, end_hub_index), (first_highway_code, total_cost)) \
            in segmentation.precomputed_paths.items():
        assert start_hub_index != end_hub_index
        assert total_cost > 0
        current_hub_index = start_hub_index
        next_highway_code = first_highway_code
        total_cost_rest = total_cost
        cycle_sentinel = [current_hub_index]
        _logger.debug(f"{start_hub_index} ... {end_hub_index} - <{first_highway_code}> {total_cost}")
        while True:
            if next_highway_code >= 0:
                next_highway = segmentation.highways[next_highway_code]
                assert next_highway.start_hub_index == current_hub_index
                next_hub_index = next_highway.end_hub_index
            else:
                next_highway = segmentation.highways[~next_highway_code]
                assert next_highway.end_hub_index == current_hub_index
                next_hub_index = next_highway.start_hub_index
            _logger.debug(f" ... {next_highway}")
            if next_hub_index == end_hub_index:
                assert next_highway.cost == total_cost_rest
                break
            current_hub_index = next_hub_index
            next_highway_code, next_total_cost = segmentation.precomputed_paths[current_hub_index, end_hub_index]
            cycle_sentinel.append(current_hub_index)
            _logger.debug(f"{current_hub_index} ... {end_hub_index} - <{next_highway_code}> {next_total_cost}")
            assert current_hub_index not in cycle_sentinel[:-1]
            assert next_total_cost + next_highway.cost == pytest.approx(total_cost_rest)
            total_cost_rest = next_total_cost


@pytest.mark.parametrize('section_map, expected_count', [
    ['''
        .........
        .@@@@@@@.
        ....@....
        ...@@@...
        ...@*@...
        ...@@@...
        .........
    ''', 21],
    ['''
        ....
        .*@.
        .@@.
        ....
    ''', 5],
    ['''
        .........
        .@@@*@@@.
        .........
    ''', 13],
    ['''
        ...
        .@.
        .@.
        .@.
        .*.
        .@.
        .@.
        .@.
        ...
    ''', 13],
    ['''
        .....
        ..@..
        .@*@.
        ..@..
        .....
     ''', 5],
    ['''
        ....
        .*@.
        .@..
        ....
     ''', 4],
    ['''
        ...
        .*.
        ...
     ''', 2],
    ['''
        .......
        .@...@.
        ..@.@..
        ...*...
        ..@.@..
        .@...@.
        .......
     ''', 17],
    ['''
        ...............
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@*@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        .@@@@@@@@@@@@@.
        ...............
     ''', 49],
    ['''
        ...............
        .@@@@@@@@@@@@@.
        .@...@...@...@.
        .@...@...@...@.
        .@...@...@...@.
        .@@@@@@@@@@@@@.
        .@...@@@@@...@.
        .@...@@*@@...@.
        .@...@@@@@...@.
        .@@@@@@@@@@@@@.
        .@...@...@...@.
        .@...@...@...@.
        .@...@...@...@.
        .@@@@@@@@@@@@@.
        ...............
     ''', 49],
])
def test_trace_section_border(section_map, expected_count):
    segmentation_code_map = []
    center_pos = None
    for y, line in enumerate(section_map.strip().split()):
        code_line = []
        for x, char in enumerate(line):
            code_line.append(Segmentation._SC_INIT if char == '.' else 0)
            if char == '*':
                center_pos = y, x
        segmentation_code_map.append(code_line)
    assert center_pos

    segmentation = Segmentation(costmap=None)
    segmentation.sections = [Mock(Section, center=center_pos, center_area_radius=1)]
    segmentation.segmentation_code_map = np.array(segmentation_code_map, Segmentation._SC_DTYPE)

    count = sum(1 for _ in segmentation._trace_section_border(0))
    assert count == expected_count


@pytest.mark.parametrize('section_map, section_index, expected_count', [
    ['''
        .................
        .---------------.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@*@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .-@@@@@@@@@@@@@-.
        .---------------.
        .................
     ''', 1, 49],
    ['''
        .................
        .---------------.
        .-@@@@@@@@@@@@@-.
        .-@---@---@---@-.
        .-@---@---@---@-.
        .-@---@---@---@-.
        .-@@@@@@@@@@@@@-.
        .-@---@@@@@---@-.
        .-@---@@*@@---@-.
        .-@---@@@@@---@-.
        .-@@@@@@@@@@@@@-.
        .-@---@---@---@-.
        .-@---@---@---@-.
        .-@---@---@---@-.
        .-@@@@@@@@@@@@@-.
        .---------------.
        .................
     ''', 2, 49],
])
def test_trace_section_border_boundary(section_map, section_index, expected_count):
    segmentation_code_map = []
    center_pos = None
    after_boundary = (Segmentation._SC_AFTER_BOUNDARY & ~Segmentation._SC_INDEX_MASK) | section_index
    before_boundary = ((Segmentation._SC_AFTER_BOUNDARY - 1) & ~Segmentation._SC_INDEX_MASK) | section_index
    for y, line in enumerate(section_map.strip().split()):
        code_line = []
        for x, char in enumerate(line):
            if char == '.':
                segmentation_code = Segmentation._SC_INIT
            elif char == '-':
                segmentation_code = after_boundary
            else:
                segmentation_code = before_boundary
            code_line.append(segmentation_code)
            if char == '*':
                center_pos = y, x
        segmentation_code_map.append(code_line)
    assert center_pos

    section = Mock(Section, center=center_pos, center_area_radius=1)
    sections = [None] * section_index + [section]
    segmentation = Segmentation(costmap=None)
    segmentation.sections = sections
    segmentation.segmentation_code_map = np.array(segmentation_code_map, Segmentation._SC_DTYPE)
    count = sum(1 for _ in segmentation._trace_section_border(section_index))
    assert count == expected_count
