from abc import abstractmethod
from enum import Enum
from logging import getLogger
from typing import Optional

from attr import define

from lovot_slam.redis.clients import create_device_client
from lovot_slam.redis.keys import DYNAMIC_VARIANT_TRACKING_MODULE_MODEL_KEY, ROBOT_MODEL_KEY
from lovot_slam.utils import OrderedEnum

_logger = getLogger(__name__)


class Model(OrderedEnum):
    @staticmethod
    @abstractmethod
    def get():
        raise NotImplementedError


class LovotModel(Model):
    LV100 = 1   # coor1 1.0
    LV101 = 2   # coro1 2.0
    LV110 = 3   # coro2

    @staticmethod
    def get():
        redis_device = create_device_client()
        model = redis_device.get(ROBOT_MODEL_KEY)
        if model.startswith('lv110'):
            return LovotModel.LV110
        if model.startswith('lv101'):
            return LovotModel.LV101
        if model.startswith('lv100'):
            return LovotModel.LV100
        if model == 'dvt2':
            return LovotModel.LV100
        if model == 'dvt1':
            return LovotModel.LV100
        raise RuntimeError(f'Unknown LOVOT model: {model}')


class NestModel(Model):
    LN100 = 1
    LN101 = 2

    @staticmethod
    def get():
        redis_device = create_device_client()
        model = redis_device.get(ROBOT_MODEL_KEY)
        if model is None:
            # Default to LN100 if model key is not set
            return NestModel.LN100
        if model.startswith('ln101'):
            return NestModel.LN101
        if model.startswith('ln100'):
            return NestModel.LN100
        if model == 'dvt2':
            return NestModel.LN100
        if model == 'dvt1':
            return NestModel.LN100
        raise RuntimeError(f'Unknown NEST model: {model}')


class TrackingModuleModel(Enum):
    G45 = 'G45'
    X45 = 'X45'

    @staticmethod
    def get():
        """returns TrackingModuleModel which is read from DVM dynamic_variant.
        """
        redis_device = create_device_client()
        model = redis_device.get(DYNAMIC_VARIANT_TRACKING_MODULE_MODEL_KEY)
        for e in TrackingModuleModel:
            if e.value == model:
                return e
        return None


class DepthCameraModel(Enum):
    MTT010 = 'mtt010'
    MTP017 = 'mtp017'
    OZT0358 = 'ozt-0358-10_binning'

    @staticmethod
    def of(tm_model: TrackingModuleModel):
        """returns DepthCameraModel of the given TrackingModuleModel.
        """
        if tm_model == TrackingModuleModel.G45:
            return DepthCameraModel.MTT010
        elif tm_model == TrackingModuleModel.X45:
            return DepthCameraModel.MTP017
        return None


@define
class HardwareVariants:
    model: Model
    depth_camera: Optional[DepthCameraModel]

    @staticmethod
    def get() -> 'HardwareVariants':
        model = LovotModel.get()
        if model < LovotModel.LV110:
            # coro1
            tracking_module_model = TrackingModuleModel.get()
            _logger.info(f'tracking module model: {tracking_module_model}')
            depth_camera = DepthCameraModel.of(tracking_module_model)
        else:
            # coro2
            depth_camera = DepthCameraModel.OZT0358
        _logger.info(f'depth camera model: {depth_camera}')
        return HardwareVariants(model, depth_camera)
