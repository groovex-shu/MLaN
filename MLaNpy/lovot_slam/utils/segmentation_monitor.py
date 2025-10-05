import json
from logging import getLogger

import trio

from lovot_slam import ContextMixin
from lovot_slam.env import MAP_2DMAP, data_directories
from lovot_slam.utils.exceptions import SlamProcedureCallError, SlamTransferError
from lovot_slam.utils.file_util import remove_directory_if_exists
from MLaNpy.lovot_map.rosmap import RosMap
from lovot_slam.utils.segmentation import SEGMENTATION_VERSION, rebuild_segmentation
from lovot_slam.utils.unwelcomed_area import calc_unwelcomed_area_hash

logger = getLogger(__name__)


class SegmentationMonitor:
    """セグメンテーションの更新必要性の監視

    - latest_map が変更されている
    - 来ないでエリアが変更されている
    - すでにあるsegmentationのバージョンが古い
    """

    def __init__(self, map_utils, spot_utils):
        self.map_utils = map_utils
        self.spot_utils = spot_utils
        self.segmentation_dir = data_directories.segmentation
        self._existing_segmentation_info = None, None

    def _remove(self):
        remove_directory_if_exists(self.segmentation_dir)
        self._existing_segmentation_info = None, None

    def _check_existing_segmentation(self):
        try:
            with open(self.segmentation_dir / 'segmentation.json') as fin:
                j = json.load(fin)
                if j['version'] == SEGMENTATION_VERSION:
                    self._existing_segmentation_info = j['map_id'], j['unwelcomed_area_hash']
        except (OSError, json.decoder.JSONDecodeError, KeyError):
            pass

    def _should_update(self):
        map_id = self.map_utils.get_latest_merged_map()
        if not map_id:
            return False
        unwelcomed_area_hash = calc_unwelcomed_area_hash(self.spot_utils.get_unwelcomed_area_from_redis())
        return (map_id, unwelcomed_area_hash) != self._existing_segmentation_info


class SegmentationUpdater(SegmentationMonitor):
    """セグメンテーションが古くなったらrebuild(spike)"""

    def __init__(self, map_utils, spot_utils):
        super().__init__(map_utils, spot_utils)
        self.cancel_scope = trio.CancelScope()
        self._active = True
        self._forcing = False
        self._activation_event = trio.Event()

    def update(self):
        """
        Recalc segmentation, and save it to 2d_map
        """
        map_name = self.map_utils.get_latest_merged_map()
        if not map_name:
            logger.info("segmentation rebuild cancelled: no map")
            return
        if not self.map_utils.check_map(map_name):
            logger.info("segmentation rebuild cancelled: map broken")
            return
        map2d_path = self.map_utils.get_full_path(map_name) / MAP_2DMAP / 'map.yaml'
        rosmap = RosMap.from_map_yaml(map_name, map2d_path)

        unwelcomed_area = self.spot_utils.get_unwelcomed_area_from_redis()
        segmentation = rebuild_segmentation(rosmap, unwelcomed_area)

        self.segmentation_dir.mkdir(parents=True, exist_ok=True)

        # serialized segmentation
        with open(self.segmentation_dir / 'segmentation.json', 'w') as fout:
            print(segmentation.encode(), file=fout)

        # image preview
        sgmt_path = self.segmentation_dir / 'segmentation.png'
        def from_base_path_getter(index):
            return self.segmentation_dir / f'cost_from_base_{index:03}.png'

        segmentation.image_preview(rosmap, unwelcomed_area, sgmt_path, from_base_path_getter)

        self._existing_segmentation_info = \
            segmentation.costmap.map_id, segmentation.costmap.unwelcomed_area_hash

    def rebuild(self):
        """コマンドから強制的にupdateする場合呼ばれる"""
        self.inactivate()
        self._forcing = True
        self.activate()

    def remove(self):
        if self._active:
            logger.warning("SegmentationUpdater.remove() implicitly deactivates the updater")
            self.inactivate()
        self._remove()

    def inactivate(self):
        """現在実行中のupdateを中止してアップデート要件の監視を停止する

        マップの作成・マージ、来ないでエリアの更新などの前に呼ばれる
        """
        self.cancel_scope.cancel()
        self._active = False

    def activate(self):
        """アップデート要件の監視を再開する

        nest-slamがひまになったら呼ばれる
        """
        self._active = True
        self._activation_event.set()

    async def run(self):
        """メインループ"""
        self._check_existing_segmentation()

        while True:
            if not self._active:
                self._activation_event = trio.Event()
                await self._activation_event.wait()

            try:
                with self.cancel_scope:
                    if self._should_update():
                        self.update()
                    self._forcing = False
                    await trio.sleep(180)
            finally:
                self.cancel_scope = trio.CancelScope()


class SegmentationDownloader(SegmentationMonitor, ContextMixin):
    """セグメンテーションが古くなっていたらdownload(tom)"""

    def __init__(self, map_utils, spot_utils):
        super().__init__(map_utils, spot_utils)
        self._forced_event = trio.Event()
        self._remove_flag = False

    def force(self):
        self._forced_event.set()

    def remove(self):
        self._remove_flag = True
        self.force()

    async def _download(self):
        map_id_tom = self.map_utils.get_latest_merged_map()
        if not map_id_tom:
            return
        try:
            map_id_spike = await self.context.slam_servicer_client.get_latest_map()
            if map_id_tom != map_id_spike:
                # logger.debug(
                #     "segmentation download postponed: map not updated "
                #     f"(tom={map_id_tom}, spike={map_id_spike}")
                return
        except SlamProcedureCallError:
            logger.error("SegmentationDownloader - latest map check failed (SlamProcedureCallError)")
            return

        logger.debug(
            "downloading segmentation... "
            f"current sgmt.info.:{self._existing_segmentation_info}")

        try:
            await self.context.slam_servicer_client.download_segmentation()
            self._check_existing_segmentation()
            logger.debug(
                "downloaded segmentation... "
                f"current sgmt.info.:{self._existing_segmentation_info}")
        except SlamProcedureCallError:
            logger.warning("download_segmentation: slam procedure call error")
        except SlamTransferError:
            logger.warning("download_segmentation: slam transfer error")

    async def run(self):
        self._check_existing_segmentation()
        forced = False
        while True:
            if forced or self._should_update():
                if self._remove_flag:
                    self._remove()
                    self._remove_flag = False
                else:
                    await self._download()
                forced = False
            with trio.move_on_after(60):
                self._forced_event = trio.Event()
                await self._forced_event.wait()
                forced = True
