"""
Prepare data directory for the localization mapset.

This script is intended to be called as an ExecStartPre script in the systemd service file,
to prepare the data directory for the localization mapset.
If a data directory with the old path "/data/localization-builder" exists,
it will be moved to a new directory with a unique ID.

Call this script with root privileges.

TODO: Create database entry for the mapset of moved data directory, with the current nest id.
"""
import os
import shutil
import sys
from logging import getLogger
from pathlib import Path

from lovot_slam.utils.map_utils import MapSetUtils

_logger = getLogger(__name__)

OLD_LOCALIZATION_DATA_DIR = Path('/data/localization-builder')

USER = 'system'
GROUP = 'system'


def main():
    # Define directories
    symlink_path = Path(os.getenv('LOCALIZATION_DATA_DIR', '/data/localization-mapset/current'))
    localization_mapset_dir = Path(os.getenv('LOCALIZATION_MAPSET_DIR', '/data/localization-mapset'))

    mapset_utils = MapSetUtils(localization_mapset_dir)

    if symlink_path.is_symlink():
        _logger.info(f"Symlink already exists: {symlink_path}")
        sys.exit(0)

    new_mapset_name = mapset_utils.generate_new_mapset_name()
    target_dir = localization_mapset_dir / new_mapset_name

    if OLD_LOCALIZATION_DATA_DIR.is_dir():
        _logger.info(f"Moving {OLD_LOCALIZATION_DATA_DIR} to {target_dir}")
        shutil.move(OLD_LOCALIZATION_DATA_DIR, target_dir)
        # TODO: Create database entry for the mapset of moved data directory, with the current nest id.
    else:
        _logger.info(f"Creating {target_dir}")
        mapset_utils.create_mapset(target_dir)
        # NOTE: Is this directory for non-map mode?

    shutil.chown(target_dir, user=USER, group=GROUP)
    symlink_path.symlink_to(target_dir)


if __name__ == "__main__":
    main()
