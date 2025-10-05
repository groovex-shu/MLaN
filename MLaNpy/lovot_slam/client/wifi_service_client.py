"""
lovot-wifiの提供するWifiServiceのgRPCクライアント

e.g.
async def _async_main():
    import time, trio_util

    async with open_wifi_service_client() as wifi_client:
        async for _ in trio_util.periodic(5):
            ret = await wifi_client.get_available_ap()
            print(time.monotonic())
            for ap in ret.ap:
                print(f'hw_address: {ap.hw_address}, '
                      f'freq: {ap.frequency}, '
                      f'strength: {ap.strength:3d}, '
                      f'last_seen: {ap.last_seen}, '
                      f'({ap.ssid})')
            print('')

if __name__ == '__main__':
    trio.run(_async_main)
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import purerpc
import trio
from google.protobuf import empty_pb2
from h2.connection import ConnectionState

from lovot_apis.lovot_minid.wifi.wifi_grpc import WifiServiceStub
from lovot_apis.lovot_minid.wifi.wifi_pb2 import GetAvailableAPResponse

from lovot_slam.client.purerpc_context import GRPC_TIMEOUT, GrpcContextManager

_logger = logging.getLogger(__name__)

LOVOT_WIFI_ENDPOINT = os.getenv('LOVOT_WIFI_ENDPOINT', '127.0.0.1:39010').split(':')


class WifiServiceClient:
    """ gRPC client for WifiService

    purerpc doesn't support timeout or disconnection error, so we use
    `trio.fail_after` alternatively.
    """

    def __init__(self, grpc_context_manager: GrpcContextManager):
        self._grpc_context_manager = grpc_context_manager

    @property
    def state(self) -> ConnectionState:
        return self._grpc_context_manager.state

    async def get_available_ap(self) -> Optional[GetAvailableAPResponse]:
        async with self._grpc_context_manager.open() as stub:
            try:
                with trio.fail_after(GRPC_TIMEOUT):
                    res = await stub.GetAvailableAP(empty_pb2.Empty())
                    return res
            except trio.TooSlowError:
                raise purerpc.DeadlineExceededError from None


@asynccontextmanager
async def open_wifi_service_client(host: str = LOVOT_WIFI_ENDPOINT[0], port: int = LOVOT_WIFI_ENDPOINT[1]):
    """
    e.g.
    async with open_wifi_service_client() as wifi_client:
        async for _ in periodic(5):
            ret = await wifi_client.get_available_ap()
            print(time.monotonic())
            for ap in ret.ap:
                print(f'hw_address: {ap.hw_address}, '
                      f'freq: {ap.frequency}, '
                      f'strength: {ap.strength:3d}, '
                      f'last_seen: {ap.last_seen}, '
                      f'({ap.ssid})')
            print('')
    """
    connection_manager = GrpcContextManager(host, port, WifiServiceStub)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(connection_manager.run)
        yield WifiServiceClient(connection_manager)
        nursery.cancel_scope.cancel()
