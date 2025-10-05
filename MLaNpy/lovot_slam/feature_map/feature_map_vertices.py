import os
from logging import getLogger
from typing import List, Optional

import numpy as np
import quaternion
import yaml

from lovot_map.accuracy_map import CostMap

from lovot_slam.env import MAP_FEATUREMAP, MAP_FEATUREMAP_MISSIONS_YAML, MAP_VERTICES, MAP_VERTICES_CSV

logger = getLogger(__name__)

# indices consist of (vertex id, timestamp and mission index)
EMPTY_INDICES = np.empty((0, 3), dtype=np.int64)
# vertices consist of 3 dimensional position and orientatioin
EMPTY_VERTICES = np.empty((0, 7), dtype=np.float64)


class Vertex:
    def __init__(self, mission_id, timestamp, pose=None):
        self._mission_id = mission_id
        self._timestamp = timestamp
        self._pose = pose

    @property
    def mission_id(self):
        return self._mission_id

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def pose(self):
        return self._pose


class FeatureMapVertices:
    VERTICES_VERTEX_IDX_COL = 0
    VERTICES_TIMESTAMP_COL = 1
    VERTICES_POSE_COLS = np.index_exp[2:9]
    INDICES_RANGE = (0, 3)
    INDICES_VERTEX_IDX_COL = 0
    INDICES_TIMESTAMP_COL = 1
    INDICES_MISSION_IDX_COL = 2

    def __init__(self, indices: np.ndarray, vertices: np.ndarray, missions: List[str]):
        self.indices = indices
        self.vertices = vertices
        self.missions = missions

    @classmethod
    def from_map_path(cls, map_path) -> Optional['FeatureMapVertices']:
        try:
            # Read mission id list from missions.yaml
            missions_yaml = os.path.join(map_path, MAP_FEATUREMAP, MAP_FEATUREMAP_MISSIONS_YAML)
            with open(missions_yaml, 'r') as f:
                metadata = yaml.safe_load(f)
                missions = metadata.get('mission_ids', [])

            # Read vertices from each mission vertices.csv
            vertices_root = os.path.join(map_path, MAP_VERTICES)
            indices = EMPTY_INDICES
            vertices = EMPTY_VERTICES
            for mission_idx, mission_id in enumerate(missions):
                csv_file_name = os.path.join(vertices_root, mission_id, MAP_VERTICES_CSV)
                single_indices = np.genfromtxt(csv_file_name, skip_header=1,
                                               dtype=np.int64,
                                               delimiter=",",
                                               usecols=(0, 1))
                single_vertices = np.genfromtxt(csv_file_name, skip_header=1,
                                                dtype=np.float64,
                                                delimiter=",",
                                                usecols=(2, 3, 4, 5, 6, 7, 8))
                # Append mission index (not mission id) to each vertex
                single_indices = np.insert(single_indices, cls.INDICES_MISSION_IDX_COL, mission_idx, axis=1)
                indices = np.concatenate((indices, single_indices), axis=0)
                vertices = np.concatenate((vertices, single_vertices), axis=0)
            return cls(
                indices=indices,
                vertices=vertices,
                missions=missions
            )
        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"Failed to read vertices files: {e}")
            return None

    @classmethod
    def filter_missions(cls, original: 'FeatureMapVertices',
                        missions_to_remain: List[str]) -> 'FeatureMapVertices':
        """Return new instance with some missions filtered.
        :param missions_to_remain: list of mission_ids to remain in the new instance
        :return: new instance which is filtered the missions
        """
        new_missions = []
        new_indices = EMPTY_INDICES
        new_vertices = EMPTY_VERTICES
        for mission_idx, mission_id in enumerate(original.missions):
            if mission_id not in missions_to_remain:
                continue
            idx = np.where(original.indices[:, cls.INDICES_MISSION_IDX_COL] == mission_idx)[0]
            single_indices = original.indices[idx, :]
            single_vertices = original.vertices[idx, :]

            # replace the mission index with the new index (which is the position in the new_missions list)
            new_mission_idx = len(new_missions)
            single_indices[:, cls.INDICES_MISSION_IDX_COL] = new_mission_idx
            new_indices = np.concatenate((new_indices, single_indices), axis=0)
            new_vertices = np.concatenate((new_vertices, single_vertices), axis=0)
            new_missions.append(mission_id)
        return cls(new_indices, new_vertices, new_missions)

    def find_closest_vertex(self, pose):
        try:
            # Find the vertex closest to given posistion in 2 dimension (x, y)
            closest_idx = np.linalg.norm(self.vertices[:, :2] - pose[:2], axis=1).argmin()
            mission_idx = self.indices[closest_idx, self.INDICES_MISSION_IDX_COL]
            mission_id = self.missions[mission_idx]
            timestamp = self.indices[closest_idx, self.INDICES_TIMESTAMP_COL]
            vertex = Vertex(mission_id, timestamp, pose=self.vertices[closest_idx])
            return vertex
        except ValueError:
            logger.debug('No closest key frame was found.')
            return None

    def find_corresponding_vertex(self, vertex: Vertex) -> Optional[Vertex]:
        """
        Find the vertex which has the same mission id and timestamp as the given vertex.
        Args:
            vertex: Vertex to find in the merged map set.
        Returns:
            Vertex if found, otherwise None.
        """
        try:
            mission_idx = self.missions.index(vertex.mission_id)
            index_to_find = np.array([vertex.timestamp, mission_idx], dtype=np.int64)
            
            columns_to_compare = [self.INDICES_TIMESTAMP_COL, self.INDICES_MISSION_IDX_COL]
            idx = np.where(np.all(self.indices[:, columns_to_compare] == index_to_find, axis=1))[0]

            if len(idx) == 0:
                logger.debug(f'Vertex not found: mission_id={vertex.mission_id}, timestamp={vertex.timestamp}')
                return None
            elif len(idx) > 1:
                logger.warning(f'Multiple vertices found: mission_id={vertex.mission_id}, timestamp={vertex.timestamp}')
                return None
            
            return Vertex(vertex.mission_id, vertex.timestamp, pose=self.vertices[idx[0]])
        
        except ValueError as e:
            logger.debug(f'Cannot find the vertex: {e}')
            return None

    def pose_of_vertex(self, vertex):
        try:
            mission_idx = self.missions.index(vertex.mission_id)
            index_to_find = np.array([vertex.timestamp, mission_idx], dtype=np.int64)

            columns_to_compare = [self.INDICES_TIMESTAMP_COL, self.INDICES_MISSION_IDX_COL]
            idx = np.where(np.all(self.indices[:, columns_to_compare] == index_to_find, axis=1))[0]
            
            return self.vertices[idx][0]
        except ValueError:
            logger.debug('Given key frame was not found on the vertices.')
            return None

    def get_number_of_vertices(self):
        return self.vertices.shape[0]

    def get_height_statistics(self):
        mean = float(np.mean(self.vertices[:, 2]))
        std = float(np.std(self.vertices[:, 2]))
        return mean, std

    def vertices_of(self, mission_id):
        mission_idx = self.missions.index(mission_id)
        idx = np.where(self.indices[:, self.INDICES_MISSION_IDX_COL] == mission_idx)[0]
        return self.vertices[idx, :]

    def timpestamps_ns_of(self, mission_id):
        mission_idx = self.missions.index(mission_id)
        idx = np.where(self.indices[:, self.INDICES_MISSION_IDX_COL] == mission_idx)[0]
        return self.indices[idx, 1]

    def vertices_with_timestamps_of(self, mission_id):
        timestamps = self.timpestamps_ns_of(mission_id) / 1.e9
        vertices = self.vertices_of(mission_id)
        return np.hstack((timestamps[:, np.newaxis], vertices))

    def get_boundary(self) -> np.ndarray:
        left = np.min(self.vertices[:, 0])
        bottom = np.min(self.vertices[:, 1])
        right = np.max(self.vertices[:, 0])
        top = np.max(self.vertices[:, 1])
        return np.array([left, bottom, right, top])


class VerticesComparator:
    """Compare two vertices.
    Two vertices should have the same missions and each mission should be the same length.
    """
    def __init__(self, reference: FeatureMapVertices, target: FeatureMapVertices):
        self._reference = reference
        self._target = target

    def translational_diffs(self, mission_id):
        t_r = self._reference.vertices_of(mission_id)[:, :3]
        t_t = self._target.vertices_of(mission_id)[:, :3]
        diffs = t_r - t_t
        return diffs

    def distance_diffs(self, mission_id):
        return np.linalg.norm(self.translational_diffs(mission_id), axis=1)

    def rotational_diffs(self, mission_id):
        q_r = self._reference.vertices_of(mission_id)[:, 3:]
        q_r = quaternion.as_quat_array(np.roll(q_r, 1, axis=1))
        q_t = self._target.vertices_of(mission_id)[:, 3:]
        q_t = quaternion.as_quat_array(np.roll(q_t, 1, axis=1))

        q_r_t = np.conjugate(q_r) * q_t
        return q_r_t

    def angle_diffs(self, mission_id):
        # anglular difference
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Recovering_the_axis-angle_representation
        q_r_t = self.rotational_diffs(mission_id)
        q_r_t = quaternion.as_float_array(q_r_t)
        anglular_diffs = 2 * np.arctan2(np.linalg.norm(q_r_t[:, 1:4], axis=1), q_r_t[:, 0])
        # convert range from [0, +2pi] to [-pi, +pi]
        anglular_diffs = (anglular_diffs + np.pi) % (2 * np.pi) - np.pi
        return anglular_diffs


def rotation_matrix_from_pose(pose):
    orientation = quaternion.from_float_array(np.roll(pose[3:], 1))
    return quaternion.as_rotation_matrix(orientation)


def look_up_transform(pose, base):
    rot_base = rotation_matrix_from_pose(base)
    trans = np.dot(rot_base.T, pose[:3] - base[:3])
    rot_pose = rotation_matrix_from_pose(pose)
    rot = np.matmul(rot_base.T, rot_pose)
    return rot, trans


def transform(pose, rot, trans):
    new_rot = np.matmul(rotation_matrix_from_pose(pose), rot)
    new_trans = np.dot(rotation_matrix_from_pose(pose), trans) + pose[:3]
    orientation = quaternion.from_rotation_matrix(new_rot)
    return np.concatenate((new_trans, np.roll(quaternion.as_float_array(orientation), -1)), axis=0)


def find_common_mission_ids(vertices_a: FeatureMapVertices, vertices_b: FeatureMapVertices) -> List[str]:
    missions_a = set(vertices_a.missions)
    missions_b = set(vertices_b.missions)

    return list(missions_a.intersection(missions_b))


def transform_pose_between_maps(original_map_path, destination_map_path, pose):
    """
    Transform pose from original map to destination map, using closest key frame of the feature maps.
    Process is:
    1. find a key frame which is closest to the given pose, on the original map,
       and look up transform from the key frame to the given pose.
    2. find the same key frame on the destination map,
       and transform the given pose using stored transformation.
    If fails to transform, return the same pose.
    Args:
        original_map_path: path to the original map
        destination_map_path: path to the destination map
        pose: np.ndarray of shape (7,): (x, y, z, qx, qy, qz, qw)
    Returns:
        transformed pose: np.ndarray of shape (7,): (x, y, z, qx, qy, qz, qw)
    """
    logger.debug(f'Transform pose {pose} from {original_map_path} to {destination_map_path}.')

    # Load key frames (vertices)
    org_vertices = FeatureMapVertices.from_map_path(original_map_path)
    dest_vertices = FeatureMapVertices.from_map_path(destination_map_path)
    if not org_vertices or not dest_vertices:
        return pose

    # Filter the original vertices to remove missions which are not in the destination.
    common_missions = find_common_mission_ids(org_vertices, dest_vertices)
    org_vertices = FeatureMapVertices.filter_missions(org_vertices, common_missions)

    # Find closest key frame and transformation, on the original (older) map
    closest_vertex = org_vertices.find_closest_vertex(pose)
    if closest_vertex is None:
        return pose
    logger.debug(f'Closest vertex on the original map: {closest_vertex.mission_id}, {closest_vertex.timestamp}.')
    logger.debug(f'Closest vertex pose on the original map: {closest_vertex.pose}')

    # Look up transformation from the key frame to the given pose
    rot, trans = look_up_transform(pose, closest_vertex.pose)

    # Transform pose relative to the key frame pose on the destination (newer) map
    dest_kf_pose = dest_vertices.pose_of_vertex(closest_vertex)
    if dest_kf_pose is None:
        logger.warning(f'The vertex is not found on the destination map: {closest_vertex.mission_id}, {closest_vertex.timestamp}.')
        logger.warning('the pose has not been transformed and keep the original.')
        return pose
    logger.debug(f'The key frame pose on the destination map: {dest_kf_pose}')

    transformed_pose = transform(dest_kf_pose, rot, trans)
    logger.debug(f'Transformed pose is {transformed_pose}')

    return transformed_pose


def transform_points_between_maps(original_map_path, destination_map_path, points):
    """
    Transform points from original map to destination map, using closest key frame of the feature maps.
    sed also transform_pose_between_maps().

    points: np.ndarray of shape (n, 2), representing 2d-points.
    """
    logger.debug(f'Transform points {points} from {original_map_path} to {destination_map_path}.')

    # Load key frames (vertices)
    org_vertices = FeatureMapVertices.from_map_path(original_map_path)
    dest_vertices = FeatureMapVertices.from_map_path(destination_map_path)
    if not org_vertices or not dest_vertices:
        return points

    # Filter the original vertices to remove missions which are not in the destination.
    common_missions = find_common_mission_ids(org_vertices, dest_vertices)
    org_vertices = FeatureMapVertices.filter_missions(org_vertices, common_missions)

    transformed_points = points.copy()

    for idx, point in enumerate(points):
        # Find closest key frame and transformation, on the original (older) map
        # TODO: 各点の closest vertex を一括検索したい気持ち
        pose = np.array((*point, 0., 0., 0., 0., 1.))
        closest_vertex = org_vertices.find_closest_vertex(pose)
        if closest_vertex is None:
            continue

        # Transform pose relative to the key frame pose on the destination (newer) map
        dest_kf_pose = dest_vertices.pose_of_vertex(closest_vertex)
        if dest_kf_pose is None:
            continue

        # Look up transformation from the key frame to the given pose
        rot, trans = look_up_transform(pose, closest_vertex.pose)

        transformed_points[idx] = transform(dest_kf_pose, rot, trans)[:2]

    logger.debug(f'Transformed points: {transformed_points}')
    return transformed_points


def create_cost_map_from_vertices(vertices_list: List[FeatureMapVertices],
                                  margin: float = 0., resolution: float = 0.05) -> CostMap:
    """Create cost map whose size covers the all vertices' positions.
    :vertices_list: list of FeatureMapVertices
    :margin: map size margin to the vertices positions [m]
    :resolution: meter / pixel
    :return: CostMap filled with 0
    """
    boundaries = np.empty((0, 4))  # left, bottom, right, top
    for vertices in vertices_list:
        boundary = vertices.get_boundary()
        boundaries = np.append(boundaries, boundary[np.newaxis, :], axis=0)
    boundary = np.hstack((np.floor(np.min(boundaries[:, :2] - margin, axis=0) / resolution),
                          np.ceil(np.max(boundaries[:, 2:] + margin, axis=0) / resolution))).astype(np.int32)

    width = boundary[2] - boundary[0]
    height = boundary[3] - boundary[1]
    logger.debug(f'vertices 2d map size: {width} x {height}')
    assert width > 0 and height > 0

    data = np.full([height, width], 0, dtype=np.uint8)
    origin = np.array([boundary[0] * resolution, boundary[1] * resolution])
    logger.debug(f'vertices 2d map origin: {origin}')

    return CostMap(data, origin, resolution)


# develop
def analyze_vertex_movements(original_map_path, destination_map_path):
    """
    Analyze how all vertices moved from original map to destination map.
    
    Returns:
        pandas.DataFrame with columns:
        - mission_id: Mission identifier
        - timestamp: Vertex timestamp  
        - orig_pos: Original position [x, y, z]
        - dest_pos: Destination position [x, y, z]
        - translation: Translation vector [dx, dy, dz]
        - distance: Translation distance (meters)
        - rotation_change: Z-axis rotation change (degrees)
    """
    try:
        import pandas as pd
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        logger.error("pandas/scipy not available. Install with: pip install pandas scipy")
        raise ImportError("Required libraries not installed")

    # Load vertices
    org_vertices = FeatureMapVertices.from_map_path(original_map_path)
    dest_vertices = FeatureMapVertices.from_map_path(destination_map_path)

    if not org_vertices or not dest_vertices:
        return None

    # Find common missions
    common_missions = find_common_mission_ids(org_vertices, dest_vertices)
    org_vertices = FeatureMapVertices.filter_missions(org_vertices, common_missions)

    movements = []

    # Iterate through all vertices in original map
    for i in range(len(org_vertices.vertices)):
        # Get vertex info
        mission_idx = org_vertices.indices[i, org_vertices.INDICES_MISSION_IDX_COL]
        mission_id = org_vertices.missions[mission_idx]
        timestamp = org_vertices.indices[i, org_vertices.INDICES_TIMESTAMP_COL]

        # Create vertex object
        org_vertex = Vertex(mission_id, timestamp, pose=org_vertices.vertices[i])

        # Find corresponding vertex in destination map
        dest_pose = dest_vertices.pose_of_vertex(org_vertex)

        if dest_pose is not None:
            # Calculate movement
            orig_pos = org_vertices.vertices[i][:3]
            dest_pos = dest_pose[:3]
            translation = dest_pos - orig_pos
            distance = np.linalg.norm(translation)

            # Calculate rotation change (Z-axis only for ground robots)
            orig_quat = org_vertices.vertices[i][3:]
            dest_quat = dest_pose[3:]
            rotation_diff = R.from_quat(dest_quat) * R.from_quat(orig_quat).inv()
            # Extract only Z-axis rotation (yaw)
            euler_diff = rotation_diff.as_euler('xyz', degrees=True)
            rotation_angle = euler_diff[2]  # Z-axis rotation only

            movements.append({
                'mission_id': mission_id,
                'timestamp': timestamp,
                'orig_pos': orig_pos,
                'dest_pos': dest_pos,
                'translation': translation,
                'distance': distance,
                'rotation_change': rotation_angle,
                'orig_pose': org_vertices.vertices[i],
                'dest_pose': dest_pose
            })

    df = pd.DataFrame(movements)
    logger.info(f"Analyzed {len(df)} vertex movements")
    return df


def plot_vertex_movements(movements_df):
    """Plot vertex movement analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")
        return

    if movements_df is None or len(movements_df) == 0:
        logger.warning("No movement data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Translation distances histogram
    axes[0,0].hist(movements_df['distance'], bins=50, alpha=0.7)
    axes[0,0].set_xlabel('Translation Distance (m)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Distribution of Vertex Translation Distances')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Z-axis rotation changes histogram  
    axes[0,1].hist(movements_df['rotation_change'], bins=50, alpha=0.7)
    axes[0,1].set_xlabel('Z-axis Rotation Change (degrees)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Distribution of Yaw Rotation Changes')
    axes[0,1].grid(True, alpha=0.3)

    # 3. 2D translation vectors 
    orig_positions = np.array(list(movements_df['orig_pos']))
    translations = np.array(list(movements_df['translation']))

    scatter_trans = axes[1,0].scatter(orig_positions[:, 0], orig_positions[:, 1], 
                                     c=movements_df['distance'], cmap='viridis', alpha=0.6, s=20)
    axes[1,0].quiver(orig_positions[:, 0], orig_positions[:, 1],
                    translations[:, 0], translations[:, 1],
                    angles='xy', scale_units='xy', scale=1, alpha=0.4, width=0.002)
    axes[1,0].set_xlabel('X (m)')
    axes[1,0].set_ylabel('Y (m)')
    axes[1,0].set_title('Translation Movements')
    axes[1,0].set_aspect('equal', adjustable='box')
    axes[1,0].grid(True, alpha=0.3)

    # Add colorbar for translation
    plt.colorbar(scatter_trans, ax=axes[1,0], label='Translation Distance (m)')

    # 4. 2D rotation scatter
    scatter_rot = axes[1,1].scatter(orig_positions[:, 0], orig_positions[:, 1], 
                                   c=movements_df['rotation_change'], cmap='RdBu_r', alpha=0.6, s=20)
    axes[1,1].set_xlabel('X (m)')
    axes[1,1].set_ylabel('Y (m)')
    axes[1,1].set_title('Rotation Changes')
    axes[1,1].set_aspect('equal', adjustable='box')
    axes[1,1].grid(True, alpha=0.3)

    # Add colorbar for rotation
    plt.colorbar(scatter_rot, ax=axes[1,1], label='Z-axis Rotation (degrees)')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("=== VERTEX MOVEMENT SUMMARY ===")
    print(f"Total vertices analyzed: {len(movements_df)}")
    print(f"Average translation distance: {movements_df['distance'].mean():.4f} m")
    print(f"Max translation distance: {movements_df['distance'].max():.4f} m")
    print(f"Average rotation change: {movements_df['rotation_change'].mean():.2f}°")
    print(f"Max rotation change: {movements_df['rotation_change'].max():.2f}°")

    return movements_df

