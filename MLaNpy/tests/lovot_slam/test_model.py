import pytest

from lovot_slam.model import LovotModel, NestModel
from lovot_slam.redis.clients import create_device_client
from lovot_slam.redis.keys import ROBOT_MODEL_KEY


@pytest.fixture(name='model_redis_key')
def fixture_model_redis_key(request):
    device_client = create_device_client()
    device_client.set(ROBOT_MODEL_KEY, request.param)
    yield
    device_client.delete(ROBOT_MODEL_KEY)


@pytest.mark.parametrize(
    'model_redis_key,model',
    [
        ('lv110', LovotModel.LV110),
        ('lv110-prepvt', LovotModel.LV110),
        ('lv110-dvt1', LovotModel.LV110),
        ('lv110-dvt2', LovotModel.LV110),
        ('lv101', LovotModel.LV101),
        ('lv100', LovotModel.LV100),
        ('dvt2', LovotModel.LV100),
        ('dvt1', LovotModel.LV100),
    ],
    indirect=['model_redis_key'],
)
def test_lovot_model(model_redis_key, model):
    assert LovotModel.get() == model


@pytest.mark.parametrize(
    'model_redis_key',
    [
        ('fs2.7'),
        ('fs1'),
    ],
    indirect=['model_redis_key'],
)
def test_lovot_model_expect_raise(model_redis_key):
    with pytest.raises(RuntimeError):
        LovotModel.get()


@pytest.mark.parametrize(
    'model_redis_key,model',
    [
        ('ln101', NestModel.LN101),
        ('ln100', NestModel.LN100),
        ('dvt2', NestModel.LN100),
        ('dvt1', NestModel.LN100),
    ],
    indirect=['model_redis_key'],
)
def test_nest_model(model_redis_key, model):
    assert NestModel.get() == model
