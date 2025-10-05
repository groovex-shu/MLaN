import csv
import copy
import pathlib
from logging import getLogger
from typing import FrozenSet, List, Optional, Set

import cv2
import numpy as np
from attr import attrs, attrib

from lovot_map.accuracy_map import CostMap
from lovot_slam.feature_map.feature_map_vertices import FeatureMapVertices, create_cost_map_from_vertices

MAX_MERGE_MISSIONS_COUNT = 30

logger = getLogger(__name__)


def create_coverage_map(vertices_list: FeatureMapVertices, cost_map: CostMap,
                        vertex_radius_m: float = 0.2,
                        missions_to_ignore: List[str] = [], clipping_threshold: int = np.inf) -> np.ndarray:
    """Create coverage area map of the vertices.
    assuming that each vertex has a circle coverage with radius of vertex_radius_m.
    :vertices: vertices of n missions
    :cost_map: cost map which covers area of the vertices
    :vertex_radius_m: radius of the exclusive circle in meter
    :clipping_threshold: maximum missions count to clip
    :return: 2d array map with each pixel has a count of missions
    """
    radius = int(vertex_radius_m / cost_map.resolution)

    # create coverage maps for every missions
    coverage_maps = np.zeros((len(vertices_list.missions),
                              cost_map.data.shape[0], cost_map.data.shape[1]))
    for i, mission_id in enumerate(vertices_list.missions):
        if mission_id in missions_to_ignore:
            continue
        vertices = vertices_list.vertices_of(mission_id)
        vertex_indices = np.unique(cost_map.world_to_map(vertices[:, :2]), axis=0)
        for vertex_i in range(vertex_indices.shape[0]):
            cv2.circle(coverage_maps[i],
                       tuple(vertex_indices[vertex_i, ::-1].tolist()), radius, 1.0, thickness=-1)

    # merge maps by summing count and clip with given threshold
    coverage_map = np.sum(coverage_maps.astype(np.float64), axis=0)
    return np.where(coverage_map > clipping_threshold, clipping_threshold, coverage_map)


@attrs
class _MissionsAdjacency:
    sorted_mission_ids: List[str] = attrib()
    matrix: np.ndarray = attrib()

    @classmethod
    def from_map_path(cls, map_path: pathlib.Path) -> '_MissionsAdjacency':
        """Load adjacency matrix from csv file in feature map.
        raise RuntimeError when the file is invalid.
        :return: mission id list (which is originally soreted by timestamp), adjacency matrix (N x N diagonal matrix)
        """
        csv_file_name = map_path / 'feature_map' / 'adjacency.csv'
        if not csv_file_name.exists():
            logger.warning(f'adjacency matrix {csv_file_name} not found')
            return None

        # format (N rows, N+1 columns. the first column represents mission_id)
        # id_0, coeff_0_0, coeff_0_1, ..., coeff_0_N
        # id_1, coeff_1_0, coeff_1_1, ..., coeff_1_N
        # ...
        # id_N, coeff_N_0, coeff_N_1, ..., coeff_N_N
        adjacency = np.genfromtxt(csv_file_name, skip_header=0,
                                  dtype=np.float64,
                                  delimiter=", ")
        if len(adjacency.shape) == 1:
            # this should happen when the feature map contains only one mission
            adjacency = adjacency[np.newaxis, :]

        # remove the first column (mission_id as string)
        # then the matrix should be diagonal
        adjacency = adjacency[:, 1:]
        if np.any(np.isnan(adjacency)) or adjacency.shape[0] != adjacency.shape[1]:
            logger.warning('the adjacency matrix is invalid')
            return None

        # original data represents inlier ratio between ith and jth missions
        # so convert it to general adjacency matrix of non-directed graph
        adjacency[adjacency > 0] = 1
        np.fill_diagonal(adjacency, 0)

        mission_ids = []
        with open(csv_file_name) as f:
            reader = csv.reader(f)
            mission_ids = [row[0] for row in reader]

        if not (adjacency.shape[0] == adjacency.shape[1] == len(mission_ids)):
            logger.warning(f'adjacency matrix size {adjacency.shape} is invalid'
                           f'with the missions count {len(mission_ids)}')

        return cls(mission_ids, adjacency)

    def _calculate_n_squared_adjacency(self) -> np.ndarray:
        """Calculate n squared adjacency matrix, where n represents number of missions.
        i, j element of (A + I)^k represents connectivity between vertex i and j with walking distance of k.
        :return: n squared adjacency matrix
        """
        adjacency = np.copy(self.matrix)
        np.fill_diagonal(adjacency, 1)
        adjacency_k = np.copy(adjacency)
        for i in range(1, self.matrix.shape[0]):
            adjacency_k[adjacency_k > 0] = 1
            if np.min(adjacency_k) > 0:
                return adjacency_k  # all elements are ones
            adjacency_k = np.dot(adjacency_k, adjacency)
        return adjacency_k

    def is_fully_connected(self) -> bool:
        """Calculate graph connectivity using adjacency matrix A.
        :return: True if missions are fully connected, else False
        """
        adjacency_n = self._calculate_n_squared_adjacency()
        return np.min(adjacency_n) > 0

    def obtain_connected_groups(self) -> Set[FrozenSet[str]]:
        """Obtain groups in which missions are connected together.
        :return: list of lists of mission_ids
        """
        adjacency_n = self._calculate_n_squared_adjacency()
        indices_to_check = set(i for i in range(adjacency_n.shape[0]))
        groups = set()
        while indices_to_check:
            index = indices_to_check.pop()
            group_members = set()
            for i in indices_to_check:
                if adjacency_n[index, i]:
                    group_members.add(i)
            indices_to_check -= group_members
            group_members.add(index)
            groups.add(frozenset(self.sorted_mission_ids[i] for i in group_members))
        return groups

    def remove(self, mission_id: str) -> None:
        idx = self.sorted_mission_ids.index(mission_id)
        del(self.sorted_mission_ids[idx])
        self.matrix = np.delete(self.matrix, idx, axis=0)
        self.matrix = np.delete(self.matrix, idx, axis=1)

    def copy(self) -> '_MissionsAdjacency':
        return _MissionsAdjacency(copy.copy(self.sorted_mission_ids),
                                  np.copy(self.matrix))


def _check_connectivity_degradation_on_missions(
        original_groups: Set[FrozenSet[str]], adjacency: _MissionsAdjacency, missions_to_remove: List[str]) -> bool:
    """Check connectivity degradation on missions.
    If connected groups of the filtered missions are the same with the original groups,
    it's considered that the map is not degraded by the filtering.
    When connectivity is originally lost on the original missions, it's ignored.
    :return: True if the missions don't degraded, else False
    """
    # prune missions from the original groups
    # this does not take the new adjacency in account
    _original_groups = set(frozenset(mission for mission in group
                                     if mission not in missions_to_remove)
                           for group in original_groups)
    _original_groups.discard(frozenset())

    # obtain groups from the filtered missions adjacency
    filtered_adjacency = adjacency.copy()
    for mission_id in missions_to_remove:
        filtered_adjacency.remove(mission_id)
    filtered_groups = filtered_adjacency.obtain_connected_groups()

    # if the group structure is the same, the map is not affected by the filtering
    return _original_groups == filtered_groups


class MissionsFilter:
    AREA_THRESHOLD_TO_REMOVE_MISSION = 0.5
    MAX_OVERLAPPED_MISSIONS = 2
    RESOLUTION = 0.05

    def __init__(self, adjacency: _MissionsAdjacency, vertices: FeatureMapVertices) -> None:
        self._adjacency = adjacency
        self._vertices = vertices

        # create empty (filled with zeros) map whose size is the same as the coverage of the vertices
        self._empty_map = create_cost_map_from_vertices([self._vertices],
                                                        margin=0.5,
                                                        resolution=self.RESOLUTION)

        self._coverage = create_coverage_map(self._vertices, self._empty_map)

    @property
    def sorted_mission_ids(self) -> List[str]:
        return self._adjacency.sorted_mission_ids

    @classmethod
    def create_from_map_path(cls, map_path: pathlib.Path) -> Optional['MissionsFilter']:
        """Create MissionsFilter from a feature map.
        """
        adjacency = _MissionsAdjacency.from_map_path(map_path)
        if not adjacency:
            logger.warning(f'failed to load adjacency matrix from feature_map: {map_path.name}')
            return None

        vertices = FeatureMapVertices.from_map_path(map_path)
        if not vertices:
            logger.warning(f'failed to read vertices from feature map: {map_path.name}')
            return None

        return cls(adjacency, vertices)

    def _calculate_coverage_area(self, coverage_map: np.ndarray) -> float:
        """
        :return: coverage area [m^2]
        """
        return np.sum(np.where(coverage_map > 0, 1, 0)) * (self._empty_map.resolution ** 2)

    def _find_overlapped_old_missions(self, area_threshold, max_overlapped_missions):
        """Find overlapped old missions, which can be removed from the map.
        :area_threshold: maximum acceptable area change [m^2]
            it is only used to decide to remove single mission,
            so that the total area change could be exceed this threshold.
        :max_overlapped_missions: maximum number of missions covering the same area
        :return: candidate of the removable missions
        """
        # create coverage map from the vertices
        original_coverage = create_coverage_map(self._vertices, self._empty_map,
                                                clipping_threshold=max_overlapped_missions)

        # extract overlapped old missions which have less effect on the total area
        overlapped_old_missions = []
        for mission_id in self.sorted_mission_ids:
            assert(mission_id in self._vertices.missions)

            missions_to_ignore = overlapped_old_missions + [mission_id]
            reduced_coverage = create_coverage_map(self._vertices, self._empty_map,
                                                   missions_to_ignore=missions_to_ignore,
                                                   clipping_threshold=max_overlapped_missions)

            diff = original_coverage - reduced_coverage
            area_to_be_reduced = self._calculate_coverage_area(diff)
            if area_to_be_reduced < area_threshold:
                overlapped_old_missions.append(mission_id)
                logger.info(f'{mission_id[:7]}: {area_to_be_reduced:.3f} [m^2] can be removed')
            else:
                logger.info(f'{mission_id[:7]}: {area_to_be_reduced:.3f} [m^2]')

        return overlapped_old_missions

    def filter_by_overlapping(self) -> List[str]:
        """
        :return: list of mission id to remove
        """
        original_groups = self._adjacency.obtain_connected_groups()

        missions_to_remove = self._find_overlapped_old_missions(self.AREA_THRESHOLD_TO_REMOVE_MISSION,
                                                                self.MAX_OVERLAPPED_MISSIONS)
        logger.info(f'found {len(missions_to_remove)} overlapped missions: '
                    f'[{[id[:7] for id in missions_to_remove]}]')

        # TODO: make this smarter
        # find missions which have least (or most) significant to keep the connectivity,
        # and remove (or keep) them
        while missions_to_remove:
            if _check_connectivity_degradation_on_missions(original_groups, self._adjacency, missions_to_remove):
                break
            # if all the missions cannot be removed due to the connectivity,
            # remain the newest one (remove it from the remove-list) and recheck
            del(missions_to_remove[-1])
        logger.info(f'{len(missions_to_remove)} missions can be removed: '
                    f'[{[id[:7] for id in missions_to_remove]}]')

        # check area change
        reduced_coverage = create_coverage_map(self._vertices, self._empty_map,
                                               missions_to_ignore=missions_to_remove)
        original_area = self._calculate_coverage_area(self._coverage)
        reduced_area = self._calculate_coverage_area(reduced_coverage)
        logger.info(f'area would be changed to {reduced_area:.2f} from {original_area:.2f} [m^2]')

        return missions_to_remove

    def filter_by_count(self, missions_to_remove: List[str] = [], max_merge_missions_count: Optional[int] = None) -> List[str]:
        """
        :missions_to_remove: list of mission id to remove selected by another filter
        :max_merge_missions_count: maximum number of missions to merge
        :return: list of mission id to remove
        """
        # calculate the number of missions to remove
        if max_merge_missions_count is None:
            max_merge_missions_count = MAX_MERGE_MISSIONS_COUNT
        missions_count = len(self._vertices.missions) - len(missions_to_remove)
        count_to_remove = missions_count - max_merge_missions_count
        if count_to_remove <= 0:
            return missions_to_remove
        logger.info(f'mission count limit({max_merge_missions_count}) exceeded: '
                    f'{missions_count} mission(s) need to be removed.')

        # pick candidate missions to remove
        # old missions and the latest mission except the missions in `missions_to_remove`
        # the number of candidate is `count_to_remove` + 1
        missions_to_remove_candidate = []
        for mission in self._vertices.missions:
            if mission in missions_to_remove:
                continue
            missions_to_remove_candidate.append(mission)
            if len(missions_to_remove_candidate) == count_to_remove:
                break
        missions_to_remove_candidate.append(self._vertices.missions[-1])

        # prepare missions to ignore sets
        missions_to_ignore_set = []
        for i in range(len(missions_to_remove_candidate)):
            missions_to_ignore_set.append(
                missions_to_remove +
                missions_to_remove_candidate[:i] +
                missions_to_remove_candidate[i+1:]
            )

        # calculate coverage of ignore sets.
        coverages = []
        for missions_to_ignore in missions_to_ignore_set:
            area = self._calculate_coverage_area(create_coverage_map(
                self._vertices,
                self._empty_map,
                missions_to_ignore=missions_to_ignore
            ))
            logger.info(f'remove {missions_to_ignore[:7]}: {area:.2f} [m^2]')
            coverages.append(area)
        max_coverages_index = coverages.index(max(coverages))
        for i, m in enumerate(missions_to_remove_candidate):
            result = "removed"
            if i == max_coverages_index:
                result = "saved"
            logger.info(f'mission {m[:7]}: {result}')

        # select a ignore set that has maximum coverage area
        return missions_to_ignore_set[max_coverages_index]
