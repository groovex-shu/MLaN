import hashlib
import os
import pathlib
import shutil
import zipfile
from logging import getLogger
from typing import List

from lovot_slam.flags.debug_params import PARAM_DISABLE_REMOVING_FILES

logger = getLogger(__name__)


def sync_to_disk():
    os.sync()
    os.sync()
    os.sync()


def remove_file_if_exists(filepath):
    if PARAM_DISABLE_REMOVING_FILES:
        return True
    if not os.path.isfile(filepath):
        return False
    try:
        logger.info(f'removing file {filepath}')
        os.remove(filepath)
        return True
    except OSError:
        logger.error(f'failed to remove {filepath}')
        return False


def remove_directory_if_exists(dirpath):
    if PARAM_DISABLE_REMOVING_FILES:
        return True
    if not os.path.isdir(dirpath):
        return False
    try:
        logger.info(f'removing directory {dirpath}')
        shutil.rmtree(dirpath)
        return True
    except OSError:
        logger.error(f'failed to remove {dirpath}')
        return False


def _scan_dir(base_dir):
    for entry in os.scandir(base_dir):
        yield entry.path
        if entry.is_dir():
            yield from _scan_dir(entry.path)


def zip_archive(base_name, root_dir, base_dir=None):
    """Create a zip archive file. (similar to shutil.make_archive)

    'base_name' is the name of the file to create, with extension ".zip"

    'root_dir' is a directory that will be the root directory of the archive.
    'base_dir' is the directory where we start archiving from;
    ie. 'base_dir' will be the common prefix of all files and
    directories in the archive.
    Returns the name of the archive file.
    """
    base_name = os.path.abspath(base_name)

    os.makedirs(os.path.dirname(base_name), exist_ok=True)

    if base_dir is not None:
        base_dir = os.path.join(root_dir, base_dir)
    else:
        base_dir = root_dir

    try:
        with zipfile.ZipFile(base_name, 'w',
                             compression=zipfile.ZIP_DEFLATED, strict_timestamps=False) as new_zip:
            for path in _scan_dir(base_dir):
                new_zip.write(path, arcname=os.path.relpath(path, root_dir))
    except FileNotFoundError:
        raise RuntimeError("file not found during archiving.")

    sync_to_disk()
    return base_name


def unzip_archive(archive_path, target_dir):
    try:
        with zipfile.ZipFile(archive_path, strict_timestamps=False) as existing_zip:
            existing_zip.extractall(target_dir)
    except zipfile.BadZipfile:
        raise RuntimeError("bad zip file.")
    sync_to_disk()


def get_file_md5sum(file_path: str):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(2048 * md5.block_size), b''):
            md5.update(chunk)
        return md5.hexdigest()


def verify_md5sum(file_path: str, expected_md5: str):
    md5 = get_file_md5sum(file_path)
    logger.debug(f"md5: {md5} / {expected_md5}")
    return md5 == expected_md5


def get_file_size(file_path: pathlib.Path) -> int:
    assert file_path.is_file()
    stat = file_path.stat()
    return stat.st_size


def get_last_modified_time(directory: pathlib.Path):
    timestamps = [path.lstat().st_mtime for path in directory.glob('**/*')]
    return max(timestamps) if timestamps else 0.


def tail(file: str, num_lines: int) -> List[str]:
    lines = []
    with open(file, 'r') as f:
        for line in f:
            lines.append(line.rstrip())
            if num_lines < len(lines):
                lines.pop()
    return lines


def get_directory_size(target_directory) -> int:
    target_directory = pathlib.Path(target_directory)
    total_size = 0

    # Traverse the directory and get the size of each file
    for file_path in target_directory.rglob('*'):
        if file_path.is_file():
            stat = file_path.stat()
            total_size += stat.st_size

    return total_size
