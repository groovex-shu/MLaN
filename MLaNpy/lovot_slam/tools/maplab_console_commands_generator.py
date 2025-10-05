from typing import Dict, List, Optional

import yaml


class MaplabConsoleCommandsRepository:
    def __init__(self):
        self._commands = []
        self._vi_map_folder_paths = []

        self._loop_closure_options = {}

    def _options_to_string(self, options):
        """Convert options dict to string:
        ex.
        convert
          {"lc_num_ransac_iters": 400,
           "lc_ransac_pixel_sigma": 4}
        to
          '--lc_num_ransac_iters 400 --lc_ransac_pixel_sigma 4'
        """
        if options:
            return ' '.join([f'--{key} {options[key]}' for key in options])
        return ''

    @property
    def commands(self):
        return self._commands

    @property
    def vi_map_folder_paths(self) -> List[str]:
        return list(map(str, self._vi_map_folder_paths))

    @property
    def loop_closure_options(self):
        return self._loop_closure_options

    @loop_closure_options.setter
    def loop_closure_options(self, options: Dict):
        self._loop_closure_options = options

    def load(self, map_folder: str):
        """Loads a map from a given path. Usage: load [--map_folder=<map_path>].
        If --map_folder isn't specified, the current path (".") is taken.
        """
        self._commands.append('load ' +
                              f'--map_folder {map_folder}')
        self._vi_map_folder_paths.append(map_folder)

    def load_merge_map(self, map_folder: str):
        """Loads the map from the given path and merges the map with the currently
        selected map. If no map is selected, a new map is created with the key
        name as the name of the folder to load the maps from. Usage:
        load_merge_map --map_folder=<map_path>
        """
        self._commands.append('load_merge_map ' +
                              f'--map_folder={map_folder}')

    def set_mission_baseframe_to_known(self):
        """Mark mission baseframe of a mission as known.
        """
        self._commands.append('set_mission_baseframe_to_known')

    def anchor_all_missions(self, lc_min_num_neighbors=None):
        """Try to anchor all missions to another mission with known baseframe.
        """
        options = self._loop_closure_options.copy()  # copy current options to protect original options
        if lc_min_num_neighbors is not None:
            options['lc_min_num_neighbors'] = lc_min_num_neighbors

        self._commands.append('anchor_all_missions ' +
                              self._options_to_string(options))

    def remove_mission(self, map_mission: str):
        """Removes mission including all vertices, edges and landmarks.
        """
        self._commands.append('remove_mission '
                              f'--map_mission={map_mission}')

    def remove_unknown_missions(self):
        """Remove missions with unknown baseframes.
        """
        self._commands.append('remove_unknown_missions')

    def relax(self):
        """Relax posegraph.
        """
        self._commands.append('relax ' +
                              self._options_to_string(self._loop_closure_options))

    def retriangulate_landmarks(self):
        """Retriangulate all landmarks.
        """
        self._commands.append('retriangulate_landmarks')

    def keyframe_heuristic(self, kf_distance_threshold_m=0.2, kf_every_nth_vertex=5):
        """Keyframe map based on heuristics.
        Use the flag --map_mission or --map_mission_list to specify which missions to keyframe.
        If neither flag is given, all missions of the selected map will be keyframed.
        """
        self._commands.append('keyframe_heuristic ' +
                              f'--kf_distance_threshold_m={kf_distance_threshold_m} ' +
                              f'--kf_every_nth_vertex={kf_every_nth_vertex}')

    def loopclosure_all_missions(self):
        """Find loop closures between all missions.
        """
        self._commands.append('loopclosure_all_missions ' +
                              self._options_to_string(self._loop_closure_options))

    def loopclosure_missions_to_all(self, missions: List[str]):
        """Find loop closures between missions and all.
        """
        self._commands.append('loopclosure_missions_to_all ' +
                              '--map_mission_list ' + ','.join(missions) + ' ' +
                              self._options_to_string(self._loop_closure_options))

    def optimize_visual_inertial(self, ba_num_iterations, ba_fix_ncamera_intrinsics: bool = True,
                                 ba_distance_edge_filename: Optional[str] = None):
        """Visual-inertial optimization over the selected missions (per default all).
        """
        self._commands.append('optimize_visual_inertial ' +
                              f'--ba_num_iterations={ba_num_iterations} ' +
                              '--ba_visualize_every_n_iterations=10 ' +
                              '--ba_fix_ncamera_intrinsics=' +
                              ('true' if ba_fix_ncamera_intrinsics else 'false') + ' ' +
                              (f'--ba_distance_edge_filename={ba_distance_edge_filename}'
                               if ba_distance_edge_filename else ''))

    def save(self, map_folder: str, overwrite: bool = False):
        """Saves a map to the given path. Usage: save [--overwrite]
        [--map_folder=<path>]
        [--copy_resources_to_map_folder=<true/false> /
         --move_resources_to_map_folder=<true/false> /
         --copy_resources_to_external_folder=<path> /
         --move_resources_to_external_folder=<path>].
        If --map_folder isn't specified,
        the map will be saved in the map folder (defined by the map metadata).
        """
        self._commands.append('save ' +
                              f'--map_folder={map_folder} ' +
                              ('--overwrite' if overwrite else ''))

    def export_mission_ids(self, mission_ids_yaml: str):
        """Export all mission ids to yaml file.
        """
        self._commands.append('export_mission_ids ' +
                              f'--mission_ids_yaml={mission_ids_yaml}')

    def csv_export_vertices_only(self, csv_export_path: str):
        """Exports only vertices in a CSV file in a folder specified by
        --csv_export_path.
        """
        self._commands.append('csv_export_vertices_only ' +
                              f'--csv_export_path={csv_export_path}')

    def set_mission_vertices_fixed(self, map_mission: str):
        """Set this mission vertices to be fixed during optimization.
        """
        self._commands.append('set_mission_vertices_fixed ' +
                              f'--map_mission={map_mission}')

    def generate_summary_map_and_save_to_disk(self, summary_map_save_path: str, overwrite: bool = False):
        """Generate a summary map of the selected map and save it
        to the path given by --summary_map_save_path.
        """
        self._commands.append('generate_summary_map_and_save_to_disk ' +
                              f'--summary_map_save_path={summary_map_save_path} ' +
                              ('--overwrite' if overwrite else ''))

    def create_missions_adjacency_matrix(self, adjacency_matrix_csv_path: str):
        """Create missions adjacency matrix using loop closures.
        specify path by --adjacency_matrix_csv_path.
        """
        self._commands.append('create_missions_adjacency_matrix ' +
                              f'--adjacency_matrix_csv_path={adjacency_matrix_csv_path}')

    def export_map_stats_yaml(self, map_stats_yaml_path: str):
        """Export map statistics to YAML file.
        """
        self._commands.append('export_map_stats_yaml ' +
                              f'--map_stats_yaml={map_stats_yaml_path}')


class MaplabConsoleCommandsGenerator:
    """Generates commands yaml file for maplab_console.
    Usage:
        with MaplabConsoleCommandsGenerator('/path/to/commands.yaml') as gen:
            gen.commands.load('/path/to/map')
            gen.commands.optimize_visual_inertial()
            gen.commands.save('/path/to/map', overwrite=True)

    The generated yaml file can be executed by:
        rosrun maplab_console batch_runner --batch_control_file='/path/to/yaml'
    """

    def __init__(self, yaml_path: str):
        self._yaml_path = yaml_path

        self._commands = MaplabConsoleCommandsRepository()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    @property
    def commands(self):
        return self._commands

    def set_loop_closure_options(self,
                                 lc_num_ransac_iters: Optional[int] = None,
                                 lc_ransac_pixel_sigma: Optional[int] = None
                                 ):
        """ Set loop closure options for the commands.
        :param lc_num_ransac_iters: Number of RANSAC iterations for loop closure.
        :param lc_ransac_pixel_sigma: RANSAC pixel sigma for loop closure.
        """
        options = {}
        if lc_num_ransac_iters:
            options["lc_num_ransac_iters"] = lc_num_ransac_iters
        if lc_ransac_pixel_sigma:
            options["lc_ransac_pixel_sigma"] = lc_ransac_pixel_sigma

        self._commands.loop_closure_options = options

    def save(self):
        data = {}
        data['vi_map_folder_paths'] = self._commands.vi_map_folder_paths
        data['commands'] = [cmd.strip() for cmd in self._commands.commands]
        with open(self._yaml_path, 'w') as stream:
            yaml.dump(data, stream=stream, default_flow_style=False, width=200)
