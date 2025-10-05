from contextlib import asynccontextmanager
from typing import Callable

import purerpc
import trio
from h2.connection import ConnectionState


async def _get_free_port():
    sock = trio.socket.socket()
    await sock.bind(('', 0))
    return sock.getsockname()[1]


@asynccontextmanager
async def open_servicer_and_client(servicer_cls: purerpc.Servicer, open_client_func: Callable):
    """Open a server and client for a given servicer and client function.
    - Server opens on a free port and is closed when the context exits.
    - Wait until the client is in IDLE state before yielding.
    """
    async with trio.open_nursery() as nursery:
        port = await _get_free_port()
        server = purerpc.Server(port)
        mock_servicer = servicer_cls()
        server.add_service(mock_servicer.service)
        await nursery.start(server.serve_async)

        async with open_client_func(host='localhost', port=port) as client:
            while client.state != ConnectionState.IDLE:
                await trio.sleep(0.1)
            yield mock_servicer, client

        nursery.cancel_scope.cancel()
