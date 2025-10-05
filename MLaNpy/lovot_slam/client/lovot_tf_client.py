"""
lovot-tf gRPC client

e.g.
async def _async_main():
    import trio_util

    async with open_lovot_tf_client() as tf_client:
        async for _ in trio_util.periodic(1):
            ret = await tf_client.get_transform('omni_map', 'base')
            print(ret)

if __name__ == '__main__':
    trio.run(_async_main)
"""
import os
from contextlib import asynccontextmanager
from logging import getLogger
from typing import Optional

import purerpc
import trio
from h2.connection import ConnectionState

from lovot_apis.lovot_tf.tf.tf_grpc import TfServiceStub
from lovot_apis.lovot_tf.tf.tf_pb2 import GetTransformRequest, SetTransformRequest, TransformStamped

from lovot_slam.client.purerpc_context import GRPC_TIMEOUT, GrpcContextManager
from lovot_slam.utils.protobuf_util import unix_time_to_pb_timestamp

_logger = getLogger(__name__)

LOVOT_TF_ENDPOINT = os.getenv('LOVOT_TF_ENDPOINT', '127.0.0.1:39080').split(':')


class LovotTfClient:
    """ gRPC client for LovotTfClient

    purerpc doesn't support timeout or disconnection error, so we use
    `trio.fail_after` alternatively.
    """

    def __init__(self, grpc_context_manager: GrpcContextManager):
        self._grpc_context_manager = grpc_context_manager

    @property
    def state(self) -> ConnectionState:
        return self._grpc_context_manager.state

    async def get_transform(self, parent_frame_id: str, child_frame_id: str,
                            *, stamp: Optional[float]) -> TransformStamped:
        async with self._grpc_context_manager.open() as stub:
            try:
                with trio.fail_after(GRPC_TIMEOUT):
                    request = GetTransformRequest(
                        stamp=unix_time_to_pb_timestamp(stamp) if stamp is not None else None,
                        parent_frame_id=parent_frame_id,
                        child_frame_id=child_frame_id
                    )
                    res = await stub.GetTransform(request)
                    return res.transform_stamped
            except trio.TooSlowError:
                raise purerpc.DeadlineExceededError from None

    async def set_transform(self, transform_stamped: TransformStamped, is_static: bool = False) -> bool:
        async with self._grpc_context_manager.open() as stub:
            try:
                with trio.fail_after(GRPC_TIMEOUT):
                    request = SetTransformRequest(
                        transform_stamped=transform_stamped,
                        authority='wifi_fingerprints_positioning',
                        is_static=is_static,
                    )
                    ret = await stub.SetTransform(request)
                    return ret.result
            except trio.TooSlowError:
                raise purerpc.DeadlineExceededError from None


@asynccontextmanager
async def open_lovot_tf_client(host: str = LOVOT_TF_ENDPOINT[0], port: int = LOVOT_TF_ENDPOINT[1]):
    """
    e.g.
    async with open_lovot_tf_client() as tf_client:
        async for _ in periodic(1):
            ret = await tf_client.get_transform('omni_map', 'base')
            print(ret)
    """
    connection_manager = GrpcContextManager(host, port, TfServiceStub)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(connection_manager.run)
        yield LovotTfClient(connection_manager)
        nursery.cancel_scope.cancel()
