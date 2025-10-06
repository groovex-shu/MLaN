import hashlib
import json
import pathlib
import shutil
from enum import Enum
from logging import getLogger
from typing import IO, Iterable, Optional, Tuple

import grpc
import numpy as np
import prometheus_client
from google.protobuf import empty_pb2

from lovot_slam.protobufs.slam_pb2 import (Area, CheckExploreRightsResponse, Coordinate,
                                                       DownloadFileRequest, DownloadFileResponse, FileInformation,
                                                       GetFrontierResponse, GetLatestMapResponse, GetSpotsResponse,
                                                       GetUnwelcomedAreaResponse, ListMapResponse, PullMapResponse,
                                                       PushBagResponse, Spot, UploadFileRequest, UploadFileResponse)
from lovot_slam.protobufs.slam_pb2_grpc import SlamServicer as ISlamServicer

from lovot_slam.env import GRPC_STREAM_CHUNK_SIZE, MAP_ID, data_directories, redis_keys
from lovot_slam.exploration.exploration_status import ExplorationStatusMonitor
from lovot_slam.exploration.exploration_token import MAX_DAILY_TOKEN, ExplorationTokenManager
from lovot_slam.map_build.request_queue import BuildSingleMapOption, RequestQueue, RequestTypes
from lovot_slam.redis.clients import create_ltm_client
from lovot_slam.utils.file_util import get_file_md5sum, remove_file_if_exists, sync_to_disk, unzip_archive, zip_archive
from lovot_map.utils.map_utils import MapUtils
from lovot_slam.utils.segmentation import validate_segmentation_metadata

logger = getLogger(__name__)
_push_bag_attempt_count = prometheus_client.Counter(
    'localization_push_bag_attempt', 'attempt count of push bag')
_push_bag_fail_count = prometheus_client.Counter(
    'localization_push_bag_fail', 'fail count of push bag')
_pull_map_attempt_count = prometheus_client.Counter(
    'localization_pull_map_attempt', 'attempt count of pull map')
_pull_map_fail_count = prometheus_client.Counter(
    'localization_pull_map_fail', 'fail count of pull map')


class UploadFileType(Enum):
    """Type of a file which is uploaded by UploadFile api.
    value (string) of each type is set to UploadFileRequest.information.type.

    If you'd like to add a new type,
    1. add the type (NEW_FILE_TYPE = 'new_file_type') here
    2. create a private method to SlamServicer that handles a received file
       and call it from SlamServicer.UploadFile()
       (such like: _store_accuracy_map)
    3. create a public method to SlamSpikeClient that prepares a file to upload
       and call SlamSpikeClient._upload_file()
       by passing the UploadFileType.NEW_FILE_TYPE as file_type arg.
       (such like: upload_accuracy_map)
    """
    ACCURACY_MAP = 'accuracy_map'
    ROSBAG = 'rosbag'
    # add types here, when adding new file types

    @classmethod
    def value_of(cls, target_value):
        for e in UploadFileType:
            if e.value == target_value:
                return e
        return None


class DownloadFileType(Enum):
    """Type of a file which is downloaded by DownloadFile api.
    value (string) of each type is set to DownloadFileRequest.information.type.

    If you'd like to add a new type,
    1. add the type (NEW_FILE_TYPE = 'new_file_type') here
    2. create a private method to SlamServicer that prepares an archive file to transfer
       and call it from SlamServicer.DownloadFile()
       (e.g. _prepare_segmentation_for_download)
    3. create a public method to SlamSpikeClient that handles the downloaded archive file
       and call SlamSpikeClient._download_file() from the method
       with the DownloadFileType.NEW_FILE_TYPE as request_type arg.
       (e.g. download_segmentation)
    """
    SEGMENTATION = 'segmentation'
    MAP = 'map'
    # add types here, when adding new file types

    @classmethod
    def value_of(cls, target_value):
        for e in DownloadFileType:
            if e.value == target_value:
                return e
        return None


class SlamServicer(ISlamServicer):
    def __init__(self, requests: RequestQueue,
                 exploration_status_monitor: ExplorationStatusMonitor,
                 exploration_token_manager: ExplorationTokenManager) -> None:
        self.requests = requests
        self.redis_ltm = create_ltm_client()

        self.map_utils = MapUtils(data_directories.maps, data_directories.bags)

        self._exploration_status_monitor = exploration_status_monitor
        self._exploration_token_manager = exploration_token_manager

    def BuildSingleMissionMap(self, request, context):
        # If the queue size exceeds the limit,
        # reject the new request and remove the corresponding bag file.
        # NOTE: The reason why we don't remove the oldest reqeust is
        #       that the request queue contains also the request for merging maps
        #       and it cannot be removed.
        map_names_in_requests = self.requests.get_map_names_in_requests()
        if len(map_names_in_requests) >= MAX_DAILY_TOKEN:
            logger.warning(f'queue size is {len(map_names_in_requests)}. '
                           'Reject the new request and remove the corresponding bag file.')
            remove_file_if_exists(data_directories.bags / f'{request.map_name}.bag')
            return empty_pb2.Empty()

        self.requests.push(RequestTypes.BuildMap, BuildSingleMapOption(request.map_name))
        logger.info('push build single mission map command to queue')
        return empty_pb2.Empty()

    def gen_pullmap_response(self, f, chunksize, md5):
        while True:
            d = f.read(chunksize)
            if len(d) == 0:
                break
            yield PullMapResponse(data=d, md5=md5)

    def calc_md5(self, filename):
        logger.debug(f'calc md5 of {filename}')
        with open(filename, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
            return checksum

    def obatain_coordinate(self, position):
        coordinate = Coordinate()
        coordinate.px = position[0]
        coordinate.py = position[1]
        coordinate.pz = 0.0
        coordinate.ox = 0.0
        coordinate.oy = 0.0
        coordinate.oz = 0.0
        coordinate.ow = 1.0
        return coordinate

    def obatain_area(self, position):
        coordinate = Coordinate()
        coordinate.px = position[0]
        coordinate.py = position[1]
        coordinate.pz = 0.0
        coordinate.ox = 0.0
        coordinate.oy = 0.0
        coordinate.oz = 0.0
        coordinate.ow = 1.0
        return Area(coordinate=coordinate, area=1.0)

    @staticmethod
    def coordinate_from_string(pose_str: str) -> Optional[Coordinate]:
        pose = list(map(float, pose_str.split(",")))
        if len(pose) != 7:
            logger.warning(f"invalid pose length {len(pose)} != 7")
            return None
        coordinate = Coordinate()
        coordinate.px = pose[0]
        coordinate.py = pose[1]
        coordinate.pz = pose[2]
        coordinate.ox = pose[3]
        coordinate.oy = pose[4]
        coordinate.oz = pose[5]
        coordinate.ow = pose[6]
        return coordinate

    @staticmethod
    def coordinate_to_string(coordinate: Coordinate) -> Optional[str]:
        return f"{coordinate.px:.4f},{coordinate.py:.4f},{coordinate.pz:.4f}," \
               f"{coordinate.ox:.4f},{coordinate.oy:.4f},{coordinate.oz:.4f},{coordinate.ow:.4f}"

    def PushBag(self, request_iterator, context):
        logger.info("PushBag called.")

        tmp_file_name = data_directories.bags / 'tmp.bag'
        if tmp_file_name.exists():
            tmp_file_name.unlink()
        file_name = ''
        empty_end = False

        try:
            _push_bag_attempt_count.inc()
            with open(tmp_file_name, 'ab') as f:
                for ite in request_iterator:
                    if len(ite.data) == 0:
                        empty_end = True
                    else:
                        empty_end = False
                    file_name = ite.name
                    f.write(ite.data)
        except Exception as e:
            _push_bag_fail_count.inc()
            logger.error('push bag error.')
            logger.error(e)
            return PushBagResponse(message='ng', md5='')

        if empty_end:
            full_path = data_directories.bags / f'{file_name}.bag'
            shutil.move(tmp_file_name, full_path)
            # sync to be secure writing to the disk, before checking md5
            sync_to_disk()

            md5 = self.calc_md5(full_path)

            return PushBagResponse(message='ok', md5=md5)

        else:
            return PushBagResponse(message='ng', md5='')

    def PullMap(self, request, context):
        logger.info('pull map requested')
        map_name = request.map_name
        archive_path = data_directories.maps / f'{map_name}.zip'
        if not self.map_utils.check_map(map_name):
            logger.info(f'map {map_name} does not exist')
            yield PullMapResponse(data='', md5='')
            return

        try:
            _pull_map_attempt_count.inc()
            logger.info('start archiving map')

            zip_archive(archive_path, data_directories.maps / map_name)

            md5 = self.calc_md5(archive_path)
            logger.debug('md5 : ' + str(md5))

            chunksize = GRPC_STREAM_CHUNK_SIZE
            with open(archive_path, 'rb') as f:
                for chunk in self.gen_pullmap_response(f, chunksize, md5):
                    yield chunk
        except RuntimeError:
            _pull_map_fail_count.inc()
            logger.error('failed to archive map.')
        except Exception as e:
            _pull_map_fail_count.inc()
            logger.error('pull map error')
            logger.error(e)
        finally:
            remove_file_if_exists(archive_path)

    def ListMap(self, request, context):
        logger.info('list map requested')
        map_list = self.map_utils.get_map_list()
        res = ListMapResponse()
        res.map_name[:] = map_list
        return res

    def GetFrontier(self, request, context):
        """Get frontier or low accuracy area.
        find frontiers first and return frontiers if found, else find low accuracy area and return it.
        """
        # TODO consider renaming: to show it supports both frontier and low accuracy area
        logger.info('get frontier requested')
        map_name = request.map_name
        res = GetFrontierResponse()
        res.map_name = map_name
        if not self.map_utils.check_map(map_name):
            logger.warning(f'map {map_name} does not exist or is invalid')
            return res

        start = np.array([request.pose.px, request.pose.py])
        frontier = self._exploration_status_monitor.find_new_frontier(map_name, True, start=start)
        low_accuracy_area = self._exploration_status_monitor.find_low_accuracy_area(map_name, update_history=True)
        logger.debug(f'frontier: {frontier}, low accuracy: {low_accuracy_area}')

        if frontier is not None:
            logger.info(f'selected frontier: {frontier}')
            res.pose.extend([self.obatain_coordinate(frontier)])
        elif low_accuracy_area is not None:
            logger.info(f'selected low accuracy area: {low_accuracy_area}')
            res.pose.extend([self.obatain_coordinate(low_accuracy_area)])
        return res

    def GetLatestMap(self, request, context):
        logger.debug('get latest map requested')
        latest_map = self.map_utils.get_latest_merged_map()
        res = GetLatestMapResponse()
        res.map_name = latest_map
        return res

    def CheckExploreRights(self, request, context):
        logger.debug(f'check explore rights requested from Ghost_ID {request.ghost_id}')
        success, token = self._exploration_token_manager.inquire_token()
        return CheckExploreRightsResponse(success=success, token=token)

    def GetSpots(self, request, context):
        """
        Get "slam:spot:{name}" key from redis LTM and response with their coordinates.
        Multiple spots can be requested.
        """
        # logger.info(f'get spots: {[spot_name for spot_name in request.spot_names]}')

        def get_spot_coordinate_from_redis(spot_name):
            pose_str = self.redis_ltm.hget(redis_keys.spot(spot_name), 'coordinate')
            if pose_str:
                return self.coordinate_from_string(pose_str)
            return None

        spots = []
        for spot_name in request.spot_names:
            coordinate = get_spot_coordinate_from_redis(spot_name)
            if coordinate:
                spots.append(Spot(name=spot_name, coordinate=coordinate))
        return GetSpotsResponse(spots=spots)

    def SetSpots(self, request, context):
        """
        Set "slam:spot:{name}" to redis LTM.
        Multiple spots can be passed.
        """
        logger.info(f'set spots: {[spot.name for spot in request.spots]}')

        def set_spot_coordinate_to_redis(spot):
            spot_dict = {
                "map_id": MAP_ID,
                "name": spot.name,
                "coordinate": self.coordinate_to_string(spot.coordinate)
            }
            self.redis_ltm.hset(redis_keys.spot(spot.name), mapping=spot_dict)

        for spot in request.spots:
            set_spot_coordinate_to_redis(spot)

        return empty_pb2.Empty()

    def GetUnwelcomedArea(self, request, context):
        """
        Get "slam:unwelcomed_area" key from redis LTM and response with the value string.
        """
        # logger.info('get unwelcomed_area')

        unwelcomed_area = self.redis_ltm.get(redis_keys.unwelcomed_area)
        return GetUnwelcomedAreaResponse(unwelcomed_area=unwelcomed_area)

    def PushLocalizationAccuracy(self, request, context):
        logger.warning('PushLocalizationAccuracy is deprecated')
        return empty_pb2.Empty()

    def _store_accuracy_map(self, archive_file: pathlib.Path, information: FileInformation) -> UploadFileResponse:
        """Extract accucary map archive and store them to the directory named after ghost_id.
        """
        metadata = json.loads(information.metadata)
        assert 'ghost_id' in metadata and metadata['ghost_id']
        ghost_id = metadata['ghost_id']

        # create accuracy_map directory under <ghost_id>, and extract the files into it
        directory = data_directories.monitor / ghost_id / 'accuracy_map'
        directory.mkdir(parents=True, exist_ok=True)
        unzip_archive(archive_file, directory)
        sync_to_disk()
        logger.info(f'accuracy map is stored in {directory}.')
        return UploadFileResponse(status='ok', message='accuracy map is stored')

    def _store_rosbag(self, rosbag_file: pathlib.Path, information: FileInformation) -> UploadFileResponse:
        """Copy the temporary rosbag file to localization/rosbag
        """
        metadata = json.loads(information.metadata)
        assert 'map_name' in metadata and metadata['map_name']
        map_name = metadata['map_name']

        # copy the temporary file to rosbag
        directory = data_directories.bags
        shutil.copy(rosbag_file, directory / f'{map_name}.bag')
        sync_to_disk()
        logger.info(f'rosbag was saved as {map_name}.bag.')
        return UploadFileResponse(status='ok', message='rosbag is stored')

    def UploadFile(self, request_iterator: Iterable[UploadFileRequest], context):
        """Upload a file to the nest by streaming.
        """
        logger.info('UploadFile called.')

        try:
            tmp_file = data_directories.tmp / 'upload_tmp_file'
            information: FileInformation = None
            empty_end = False

            # store streaming data as a file
            with open(tmp_file, 'wb') as f:
                for ite in request_iterator:
                    if ite.WhichOneof('information_optional') == 'information':
                        information = ite.information
                    f.write(ite.data)
                    empty_end = len(ite.data) == 0
            if not empty_end:
                return UploadFileResponse(status='err: invalid end of requests')
            if not information:
                return UploadFileResponse(status='err: file information not found')

            # verify md5
            md5 = get_file_md5sum(tmp_file)
            if information.md5 != md5:
                return UploadFileResponse(status=f'err: md5 unmatch: {md5}')

            # handle the temporary file
            file_type = UploadFileType.value_of(information.type)
            if file_type == UploadFileType.ACCURACY_MAP:
                logger.info('UploadFile file_type: ACCURACY_MAP')
                return self._store_accuracy_map(tmp_file, information)
            elif file_type == UploadFileType.ROSBAG:
                logger.info('UploadFile file_type: ROSBAG')
                return self._store_rosbag(tmp_file, information)
            # add handlers here, when adding new file types
        finally:
            tmp_file.unlink(missing_ok=True)

        return UploadFileResponse(status='ok', message='no operation has been done')

    def _prepare_segmentation_for_download(self, archive_path: pathlib.Path, req_metadata_str: str) -> Tuple[str, str]:
        """archive files for transfer to tom.
        :archive_path: file path to archive
        :req_metadata: jsonized metadata
        :return: Tuple of status and jsonized metadata
        jsonized metadata would be transferred to tom with the archive file
        """
        if not data_directories.segmentation.exists():
            return 'segmentation directory not found', ''
        req_metadata = json.loads(req_metadata_str)
        metadata_or_errmsg = validate_segmentation_metadata(data_directories.segmentation, **req_metadata)
        if isinstance(metadata_or_errmsg, str):
            return metadata_or_errmsg, ''
        zip_archive(archive_path, data_directories.segmentation)
        return '', json.dumps(metadata_or_errmsg)

    def _prepare_map_for_download(self, archive_path: pathlib.Path, req_metadata_str: str) -> Tuple[str, str]:
        logger.info('download map requested')
        req_metadata = json.loads(req_metadata_str)
        assert 'map_name' in req_metadata, 'map_name is not in metadata'

        map_name = req_metadata['map_name']
        if not self.map_utils.check_map(map_name):
            logger.warning(f'map {map_name} does not exist')
            return f'map {map_name} does not exist', ''

        zip_archive(archive_path, data_directories.maps / map_name)
        res_metadata = {
            'map_name': map_name
        }
        return '', json.dumps(res_metadata)

    def DownloadFile(self, request: DownloadFileRequest, context: grpc.ServicerContext):
        """Download a file (usually an archive file) (from spike to tom).
        Preparation of the file is depending on the requested file type.
        """
        logger.info(f'DownloadFile: requested type is {request.type}, metadata is {request.metadata}')

        def gen(f: IO, information: FileInformation) -> DownloadFileResponse:
            while True:
                d = f.read(GRPC_STREAM_CHUNK_SIZE)
                yield DownloadFileResponse(data=d, information=information)
                information = None  # information is sent only with the first chunk
                if len(d) == 0:
                    # send 0-byte datum at last, for confirmation of end
                    break

        # transfer the file with streaming
        try:
            # prepare an archive file and information to transfer
            archive_path = data_directories.tmp / 'temporary_download_file'
            information = FileInformation(type=request.type)
            file_type = DownloadFileType.value_of(request.type)
            if file_type == DownloadFileType.SEGMENTATION:
                status, information.metadata = self._prepare_segmentation_for_download(archive_path, request.metadata)
            elif file_type == DownloadFileType.MAP:
                status, information.metadata = self._prepare_map_for_download(archive_path, request.metadata)
            else:
                logger.error(f'DownloadFile: invalid request type {request.type}')
                context.abort(code=grpc.StatusCode.INVALID_ARGUMENT,
                              details=f'invalid request type {request.type}')
            if not archive_path.exists():
                context.abort(code=grpc.StatusCode.NOT_FOUND, details=status)

            information.md5 = get_file_md5sum(archive_path)
            logger.debug(f'DownloadFile: file type {information.type}, md5sum {information.md5}')
            logger.debug(f'DownloadFile: file metadata {information.metadata}')
            with open(archive_path, 'rb') as f:
                for chunk in gen(f, information):
                    yield chunk
        except FileNotFoundError as e:
            logger.error(f'DownloadFile: failed to read file {e}')
        finally:
            archive_path.unlink(missing_ok=True)
