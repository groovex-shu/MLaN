import argparse
import pathlib
from typing import Dict

import numpy as np
import quaternion
import yaml
from attr import attrs, attrib

from lovot_slam.feature_map.feature_map_vertices import FeatureMapVertices


def parse_args():
    parser = argparse.ArgumentParser(description="compare trajectories given by csv")
    parser.add_argument("--reference_map",
                        help="reference map name")
    parser.add_argument("--mission_id",
                        help="reference mission id")
    parser.add_argument("--depth_csv",
                        help="csv file containing poses estimated by depth")
    parser.add_argument("--visual_csv",
                        help="csv file containing poses estimated by visual")
    parser.add_argument("--gt_csv",
                        help="csv file containing poses estimated by groundtruth")
    parser.add_argument("--gt_tf_yaml",
                        help="yaml file which stores transform map coords to groundtruth")
    parser.add_argument("--output", help="directory path to output results")
    parser.add_argument("--long_format", "-l", action="store_true", default=False,
                        help="csv format contains 19 columns")

    args = parser.parse_args()
    return args


@attrs
class _Transform:
    translation: np.ndarray = attrib()
    rotation: quaternion.quaternion = attrib()

    @classmethod
    def load_from_yaml(cls, yaml_path: pathlib.Path) -> '_Transform':
        with open(yaml_path, 'r') as f:
            dict = yaml.safe_load(f)
            translation = np.array(dict['transform']['translation'])
            rotation = quaternion.from_float_array(
                np.roll(np.array(dict['transform']['rotation']), 1))
        return cls(translation, rotation)

    @classmethod
    def load_from_array(cls, array: np.ndarray) -> '_Transform':
        """tx, ty, tz, qx, qy, qz, qw
        """
        return cls(array[0:3],
                   quaternion.from_float_array(np.roll(array[3:7], 1)))


class Trajectory:
    def __init__(self, vertices):
        self._vertices = vertices

    @classmethod
    def load_from_feature_map_vertices(cls, map_path: pathlib.Path, mission_id: str) -> 'Trajectory':
        """Load from feature map vertices with scale compensation.
        """
        with open(map_path / 'lovot_slam.yaml', 'r') as f:
            map_conf = yaml.safe_load(f)
            scale = map_conf['lovot_slam']['scale_odom_loc']
        reference = FeatureMapVertices.from_map_path(map_path)
        assert reference
        vertices = reference.vertices_with_timestamps_of(mission_id)
        vertices[:, 1:4] *= scale
        return cls(vertices)

    @classmethod
    def load_from_csv(cls, csv_path: pathlib.Path, nanoseconds_used: bool = False,
                      long_format: bool = False) -> 'Trajectory':
        """Load from csv which is formatted as:
        timestamp[sec or nanosec], pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w
         or
        timestamp, tf_stamp, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w,
            cov_stamp,cov_00,cov_01,cov_02,cov_10,cov_11,cov_12,cov_20,cov_21,cov_22
        """
        target_cols = (0, 1, 2, 3, 4, 5, 6, 7)
        if long_format:
            target_cols = (0, 2, 3, 4, 5, 6, 7, 8)

        vertices = np.genfromtxt(csv_path, skip_header=1,
                                 dtype=np.float64,
                                 delimiter=",",
                                 usecols=target_cols)
        if nanoseconds_used:
            vertices[:, 0] = vertices[:, 0] / 1.0e9
        return cls(vertices)

    @property
    def vertices(self):
        return self._vertices

    def timestamp(self, index):
        return self._vertices[index, 0]

    def find_vertex_close_in_time(self, timestamp):
        idx = (np.abs(self._vertices[:, 0] - timestamp)).argmin()
        return self._vertices[idx, :]

    def transform(self, tf: _Transform):
        rotation = quaternion.as_rotation_matrix(tf.rotation)
        # transform translation (with broadcast)
        self._vertices[:, 1:4] = np.dot(rotation, self._vertices[:, 1:4].T).T + tf.translation
        # transform rotation (with broadcast)
        # format of numpy quaternion is (qw, qx, qy, qz), while ours is (qx, qy, qz, qw)
        self._vertices[:, 4:8] = quaternion.as_float_array(
            tf.rotation * quaternion.from_float_array(np.roll(self._vertices[:, 4:8], 1, axis=1)))
        self._vertices[:, 4:8] = np.roll(self._vertices[:, 4:8], -1, axis=1)


def _compare_trajectories(target: Trajectory, reference: Trajectory) -> np.ndarray:
    """Compare two trajectories and return error history with timestamp.
    Scan all pose (vertex) in the target trajectory,
    and compare each with pose in the reference which is the closest in time.
    """
    errors = np.empty((0, 2))  # timestamp, error
    for idx in range(0, target.vertices.shape[0]):
        vertex_tgt = target.vertices[idx]
        timestamp = target.timestamp(idx)
        vertex_ref = reference.find_vertex_close_in_time(timestamp)
        time_diff = timestamp - vertex_ref[0]
        if timestamp != 0.0 and abs(time_diff) < 0.5:
            error = np.array((timestamp, np.linalg.norm(vertex_tgt[1:4] - vertex_ref[1:4])))
            errors = np.append(errors, error[np.newaxis, :], axis=0)
    return errors


def _plot_trajectories(trajectories: Dict[str, Trajectory], output_dir: pathlib.Path = None):
    """Plot trajectories given as a dictionary in x-y plane.
    """
    import matplotlib.pyplot as plt
    # x-y
    fig, ax = plt.subplots()
    for label, trajectory in trajectories.items():
        plt.plot(trajectory.vertices[:, 1].flatten().tolist(),
                 trajectory.vertices[:, 2].flatten().tolist(),
                 "-", markersize=1,
                 label=label)

    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.title("xy_plot")
    plt.grid()
    ax.set_aspect('equal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0)

    if output_dir:
        plt.savefig(output_dir / 'trajectory.png', dpi=300)


def _plot_errros(errors_dict, reference: str = "", output_dir: pathlib.Path = None):
    """Plot errors time history.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for label, errors in errors_dict.items():
        # this might cause inconsistency in timestamps between different trajectories
        timestamps = errors[:, 0].flatten() - errors[0, 0]
        plt.plot(timestamps,
                 errors[:, 1].flatten().tolist(),
                 "-", markersize=1,
                 label=label)

    plt.xlabel("timestamp [sec]")
    plt.ylabel("error [m]")
    plt.ylim(0, 1.0)
    plt.grid()
    plt.title(f"distance errors to {reference}")
    plt.legend()

    if output_dir:
        plt.savefig(output_dir / 'errors.png', dpi=300)


def run():
    args = parse_args()

    def pathlib_or_None(path: str) -> pathlib.Path:
        return pathlib.Path(path) if path else None

    depth_csv = pathlib_or_None(args.depth_csv)
    visual_csv = pathlib_or_None(args.visual_csv)
    gt_csv = pathlib_or_None(args.gt_csv)
    gt_tf_yaml = pathlib_or_None(args.gt_tf_yaml)
    reference_map = pathlib_or_None(args.reference_map)
    output_dir = pathlib_or_None(args.output)

    long_csv_format = args.long_format
    # Load trajectories from csv files.
    trajectories = {}
    if depth_csv and depth_csv.exists():
        trajectories["depth"] = Trajectory.load_from_csv(depth_csv,
                                                         nanoseconds_used=False,
                                                         long_format=long_csv_format)
    if visual_csv and visual_csv.exists():
        trajectories["visual"] = Trajectory.load_from_csv(visual_csv,
                                                          nanoseconds_used=False,
                                                          long_format=long_csv_format)

    # Load reference trajectories from groundtruth or keyframes.
    reference = "not available"
    if gt_csv and gt_csv.exists() and gt_tf_yaml and gt_tf_yaml.exists():
        # Transform trajectories on map coords to groundtruth coords
        tf = _Transform.load_from_yaml(args.gt_tf_yaml)
        for trajectory in trajectories.values():
            trajectory.transform(tf)
        # Load groundtruth (from vicon, vive, etc.)
        trajectories["groundtruth"] = Trajectory.load_from_csv(gt_csv,
                                                               nanoseconds_used=False,
                                                               long_format=long_csv_format)
        reference = "groundtruth"
    elif reference_map and reference_map.exists() and args.mission_id:
        trajectories["keyframes"] = Trajectory.load_from_feature_map_vertices(
            reference_map, args.mission_id)
        reference = "keyframes"

    # Calculate time history of distance error reference to the keyframes.
    errors_dict = {}
    if "depth" in trajectories and reference in trajectories:
        errors = _compare_trajectories(trajectories["depth"], trajectories[reference])
        errors_dict["depth"] = errors
        print('depth vs groundtruth:')
        print(f'  mean: {np.nanmean(errors, axis=0)[1]:.3f} m')
        print(f'  max : {np.nanmax(errors, axis=0)[1]:.3f} m')

    if "visual" in trajectories and reference in trajectories:
        errors = _compare_trajectories(trajectories["visual"], trajectories[reference])
        errors_dict["visual"] = errors
        print('visual vs groundtruth')
        print(f'  mean: {np.nanmean(errors, axis=0)[1]:.3f} m')
        print(f'  max : {np.nanmax(errors, axis=0)[1]:.3f} m')

    # Plot
    _plot_trajectories(trajectories, output_dir)
    _plot_errros(errors_dict, reference, output_dir)


if __name__ == '__main__':
    run()
