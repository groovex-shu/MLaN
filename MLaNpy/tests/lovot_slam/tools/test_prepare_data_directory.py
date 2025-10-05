import os
from pathlib import Path
from tempfile import TemporaryDirectory

from lovot_slam.tools import prepare_data_directory


def test_prepare_without_prior_data_directory(monkeypatch):
    prepare_data_directory.USER = os.getuid()
    prepare_data_directory.GROUP = os.getgid()

    with TemporaryDirectory() as mapset_dir:
        mapset_dir = Path(mapset_dir)
        localization_data_dir = mapset_dir / 'current'
        # set the environment variables
        os.environ['LOCALIZATION_MAPSET_DIR'] = str(mapset_dir)
        os.environ['LOCALIZATION_DATA_DIR'] = str(localization_data_dir)

        # call main function
        prepare_data_directory.main()

        # there are 1 directory and 1 symlink in the mapset directory
        assert len(list(mapset_dir.iterdir())) == 2
        assert localization_data_dir.is_symlink()
        # the symlink points to the directory
        assert localization_data_dir.resolve() in mapset_dir.iterdir()


def test_prepare_with_prior_data_directory(monkeypatch):
    prepare_data_directory.USER = os.getuid()
    prepare_data_directory.GROUP = os.getgid()

    with TemporaryDirectory() as old_data_dir, TemporaryDirectory() as mapset_dir:
        old_data_dir = Path(old_data_dir)
        # create a dummy file in the old directory, for testing purposes
        (old_data_dir / 'dummy_file').touch()

        mapset_dir = Path(mapset_dir)
        localization_data_dir = mapset_dir / 'current'
        # set the environment variables
        os.environ['LOCALIZATION_MAPSET_DIR'] = str(mapset_dir)
        os.environ['LOCALIZATION_DATA_DIR'] = str(localization_data_dir)
        prepare_data_directory.OLD_LOCALIZATION_DATA_DIR = old_data_dir

        # call main function
        prepare_data_directory.main()

        # there are 1 directory and 1 symlink in the mapset directory
        assert len(list(mapset_dir.iterdir())) == 2
        assert localization_data_dir.is_symlink()
        # the symlink points to the directory
        assert localization_data_dir.resolve() in mapset_dir.iterdir()
        # the directory contains the file from the old directory
        assert (localization_data_dir / 'dummy_file').exists()

        # the old directory is gone
        assert not old_data_dir.exists()
