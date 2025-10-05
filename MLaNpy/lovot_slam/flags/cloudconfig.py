from lovot_slam.flags.common import get_param_from_device

# lovot slam
CLOUDCONFIG_DISABLE_BUILD_MAP = get_param_from_device("cloud:config:disable_build_map", False)
CLOUDCONFIG_LC_MIN_NEIGHBORS = get_param_from_device("cloud:config:lc_min_neighbors", 2)

# ros
CLOUDCONFIG_ENABLE_MARKER_LOCALIZATION_RELOCALIZATION = get_param_from_device("cloud:config:localization_enable_marker_relocalization", False)
CLOUDCONFIG_ENABLE_MARKER_LOCALIZATION_REGISTER = get_param_from_device("cloud:config:localization_enable_marker_register", False)
CLOUDCONFIG_DISABLE_MAP_UPDATE = get_param_from_device("cloud:config:localization_disable_map_update", False)