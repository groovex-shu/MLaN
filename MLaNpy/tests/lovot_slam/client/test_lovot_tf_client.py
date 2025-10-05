import pytest
from google.protobuf import timestamp_pb2

from lovot_apis.lovot_tf.tf.tf_grpc import TfServiceServicer
from lovot_apis.lovot_tf.tf.tf_pb2 import (GetTransformRequest, GetTransformResponse, Header, Quaternion,
                                           SetTransformRequest, SetTransformResponse, Transform, TransformStamped,
                                           Vector3)

from lovot_slam.client import lovot_tf_client, open_lovot_tf_client

from .util import open_servicer_and_client


class MockTfServicer(TfServiceServicer):
    def __init__(self) -> None:
        super().__init__()

        self.response = None
        self.request = None

    async def GetTransform(self, request: GetTransformRequest):
        assert self.response
        print(request)
        self.request = request
        return self.response

    async def SetTransform(self, request: GetTransformRequest):
        assert self.response
        self.request = request
        return self.response


@pytest.fixture
async def open_mock_servicer_and_client(monkeypatch, nursery):
    # NOTE: サーバの起動に時間がかかることがあるので、タイムアウトを長めに設定する
    monkeypatch.setattr(lovot_tf_client, 'GRPC_TIMEOUT', 10.0)

    async with open_servicer_and_client(MockTfServicer, open_lovot_tf_client) as (mock_service, client):
        yield mock_service, client


@pytest.mark.parametrize(
    "parent_frame_id,child_frame_id,stamp,expected_response",
    [
        ('map', 'base_link', None,
         GetTransformResponse(
                transform_stamped=TransformStamped(
                    header=Header(
                        frame_id="map",
                        stamp=timestamp_pb2.Timestamp(seconds=1, nanos=0),
                    ),
                    child_frame_id="base_link",
                    transform=Transform(
                        translation=Vector3(x=0.0, y=0.0, z=0.0),
                        rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    ),
                ),
         )),
    ]
)
async def test_get_transform(open_mock_servicer_and_client,
                             parent_frame_id, child_frame_id, stamp,
                             expected_response: GetTransformResponse):
    mock_servicer, client = open_mock_servicer_and_client
    mock_servicer.response = expected_response

    res = await client.get_transform(parent_frame_id, child_frame_id, stamp=stamp)

    assert mock_servicer.request == GetTransformRequest(
        parent_frame_id=parent_frame_id,
        child_frame_id=child_frame_id,
        stamp=stamp,
    )
    assert res == expected_response.transform_stamped


@pytest.mark.parametrize(
    "reuqest",
    [
        (SetTransformRequest(
                transform_stamped=TransformStamped(
                    header=Header(
                        frame_id="map",
                        stamp=timestamp_pb2.Timestamp(seconds=1, nanos=0),
                    ),
                    child_frame_id="base_link",
                    transform=Transform(
                        translation=Vector3(x=0.0, y=0.0, z=0.0),
                        rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    ),
                ),
                authority='wifi_fingerprints_positioning',
                is_static=False,
         )),
    ]
)
async def test_set_transform(open_mock_servicer_and_client,
                             reuqest):
    mock_servicer, client = open_mock_servicer_and_client
    mock_servicer.response = SetTransformResponse()

    _ = await client.set_transform(reuqest.transform_stamped)

    assert mock_servicer.request == reuqest
