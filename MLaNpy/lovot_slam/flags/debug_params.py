from lovot_slam.flags.common import get_param_from_ltm

PARAM_DISABLE_REMOVING_FILES = get_param_from_ltm("slam:debug:disable_removing_files", False)
PARAM_BUILD_MAP_RATE = get_param_from_ltm("slam:debug:build_map_rate", 0.5)
PARAM_BUILD_DENSE_MAP_RATE = get_param_from_ltm("slam:debug:build_dense_map_rate", 4.0)
PARAM_MAP_OPTIMIZATION_THREAD_NUM = get_param_from_ltm("slam:debug:map_optimization_thread_num", 1)
PARAM_MERGE_MAPS_MINIMUM_MISSIONS_TO_OPTIMIZE = get_param_from_ltm(
    "slam:debug:param_merge_maps_minimum_missions_to_optimize", 4)
PARAM_MERGE_MAPS_MINIMUM_MISSIONS_TO_MAINTAIN = get_param_from_ltm(
    "slam:debug:param_merge_maps_minimum_missions_to_maintain", 8)
PARAM_MISSIONS_COUNT_TO_GIVE_UP_EXPLORATION = get_param_from_ltm(
    "slam:debug:param_missions_count_to_give_up_exploration", 20)
PARAM_EXPLORATION_STATUS_RECHECK_PERIOD_AFTER_SUSPENSION = get_param_from_ltm(
    "slam:debug:param_exploration_status_recheck_period_after_suspension", 7 * 24 * 60 * 60)
PARAM_EXPLORATION_STATUS_INTERVAL_UNTIL_ACCURACY_CHECK = get_param_from_ltm(
    "slam:debug:param_exploration_status_interval_until_accuracy_check", 35 * 24 * 60 * 60)
PARAM_ALWAYS_BUILD_MAP_ON_BUILDER = get_param_from_ltm(
    "slam:debug:always_build_map_on_builder", False)
PARAM_FRONT_CAMERA_FRAMERATE = get_param_from_ltm(
    "slam:debug:front_camera_framerate", 0.25)
