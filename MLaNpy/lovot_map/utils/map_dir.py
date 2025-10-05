import pathlib



class DataDirectories:
    def __init__(self, data_dir) -> None:
        self._data_root = pathlib.Path(data_dir)

        self._bags_directory = 'rosbag'
        self._maps_directory = 'maps'
        self._monitor_directory = 'monitor'
        self._segmentation_directory = 'segmentation'
        self._camera_config_directory = 'calibration'
        self._tmp_directory = 'tmp'

    @property
    def data_root(self) -> pathlib.Path:
        return self._data_root

    @property
    def bags(self) -> pathlib.Path:
        return self.data_root / self._bags_directory

    @property
    def maps(self) -> pathlib.Path:
        return self.data_root / self._maps_directory

    @property
    def monitor(self) -> pathlib.Path:
        return self.data_root / self._monitor_directory

    @property
    def segmentation(self) -> pathlib.Path:
        return self.data_root / self._segmentation_directory

    @property
    def camera_config(self) -> pathlib.Path:
        return self.data_root / self._camera_config_directory

    @property
    def tmp(self) -> pathlib.Path:
        return self.data_root / self._tmp_directory