"""
This module provides a context manager for purerpc.
Copied from lovot-neodm/object_detection/client.py
and modified some parts.
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

import anyio
import h2
import purerpc
import trio
from h2.connection import ConnectionState
from trio_util import AsyncValue

_logger = logging.getLogger(__name__)

GRPC_TIMEOUT = 2.0
RETRY_DURATION = 5


class GrpcContextManager:
    def __init__(self, host, port, stub_class,
                 concurrent_connections: Optional[int] = None):
        self.host = host
        self.port = port
        self.stub_class = stub_class
        self._stub_event = AsyncValue(None)
        self._stub_cancel_scope = trio.CancelScope()
        self._connected = False

        self._reconnect_trigger = trio.Event()
        self._concurrent_connections = concurrent_connections
        self._connection_count_event = AsyncValue(0)

    @property
    def _stub(self):
        return self._stub_event.value

    @property
    def state(self) -> ConnectionState:
        if not self._stub:
            return ConnectionState.CLOSED
        return self._stub._client.channel._grpc_socket._grpc_connection.h2_connection.state_machine.state

    def change_host(self, host):
        """Change the host of the gRPC server,
            and close the connection only when the host is actually changed.
        The connection will be automatically re-established in the `run` coroutine.
        This method does not wait for the connection to be re-established or even to be closed.
        """
        if self.host == host:
            return  # do nothing, if the host is not changed
        self.host = host
        self._stub_cancel_scope.cancel()

    async def _get_stub(self, timeout):
        with trio.move_on_after(timeout):
            if self._concurrent_connections:
                # +1 for the current connection
                await self._connection_count_event.wait_value(lambda v: v < self._concurrent_connections + 1)
            return await self._stub_event.wait_value(lambda v: v is not None)
        raise purerpc.UnavailableError

    def get_channel_cm(self):
        return purerpc.insecure_channel(self.host, self.port)

    def _close_connection(self):
        if not self._connected:
            return  # do nothing, if the connection is not established
        _logger.debug("close gRPC server connection")
        self._stub_cancel_scope.cancel()

    def trigger_reconnect(self):
        """Trigger reconnection to the gRPC server.
        Not like _close_connection, this method does not close the connection immediately.
        """
        self._reconnect_trigger.set()

    async def run(self, *, task_status=trio.TASK_STATUS_IGNORED):
        """
        Repeatedly make a singleton connection and sleep forever.
        The connection can be cancelled manually using self._stub_cancel_scope.

        With the task_status protocol, returns when the initial connection
        is established.  (Maybe useful for testing.)
        """
        first = True
        lost_connection = False
        while True:
            self._stub_cancel_scope = trio.CancelScope()
            with self._stub_cancel_scope:
                try:
                    async with self.get_channel_cm() as channel:
                        _logger.info(f"connected to gRPC server {self.host}:{self.port}")
                        self._stub_event.value = self.stub_class(channel)
                        if first:
                            task_status.started()
                            first = False
                        if lost_connection:
                            _logger.debug("reconnected to gRPC server")
                            lost_connection = False

                        # delay to wait for the server to be ready
                        # NOTE: this is a workaround for the issue
                        # the server seems not being ready soon after insecure_channel open
                        # and it causes infinite loop of open/close.
                        await trio.sleep(2)
                        self._connected = True

                        # close connection if trigger is set and no connection is active
                        self._reconnect_trigger = trio.Event()
                        await self._reconnect_trigger.wait()
                        await self._connection_count_event.wait_value(0)
                except (OSError, anyio.BrokenResourceError, h2.exceptions.ProtocolError):
                    if not lost_connection:
                        _logger.debug("lost connection to gRPC server")
                        lost_connection = True
                finally:
                    self._stub_event.value = None
                    self._connected = False
            await trio.sleep(RETRY_DURATION)
            _logger.debug("reconnecting to gRPC server...")

    @asynccontextmanager
    async def open(self, timeout=GRPC_TIMEOUT):
        self._connection_count_event.value += 1
        try:
            yield await self._get_stub(timeout)
        except purerpc.RpcFailedError as exc:
            if isinstance(exc, purerpc.DeadlineExceededError):
                # close channel if server doesn't respond. otherwise, keep connection
                self._close_connection()
            _logger.debug(f"RpcFailedError: {exc}")
            raise
        finally:
            self._connection_count_event.value -= 1
