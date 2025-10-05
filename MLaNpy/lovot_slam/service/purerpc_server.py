"""
NOTE: Consider creating a pull request to purerpc to include this change in the library.
"""
from contextlib import AsyncExitStack
from logging import getLogger

import anyio
import purerpc
from anyio import TASK_STATUS_IGNORED
from anyio.streams.tls import TLSListener
from purerpc.server import ConnectionHandler

_logger = getLogger(__name__)


class Server(purerpc.Server):
    """Wrapper around purerpc.Server to specify the host to listen on.

    When the host is not specified, anyio.create_tcp_listener resolves a host address using getaddrinfo.
    This can sometimes fail to resolve the address for IPv4, and the issue may go unnoticed.
    To address this, we explicitly specify the host to listen on, ensuring the process fails intentionally
    if the host cannot be resolved.
    """

    def __init__(self, host=None, port=50055, ssl_context=None):
        super().__init__(port, ssl_context)
        self.host = host

    async def serve_async(self, *, task_status=TASK_STATUS_IGNORED):
        """Run the grpc server

        The task_status protocol lets the caller know when the server is
        listening, and yields the port number (same given to Server constructor).

        :raises socket.gaierror: if the host cannot be resolved
        """

        # TODO: resource usage warning
        async with AsyncExitStack() as stack:
            tcp_server = await anyio.create_tcp_listener(local_host=self.host, local_port=self.port, reuse_port=True)
            # read the resulting port, in case it was 0
            self.port = tcp_server.extra(anyio.abc.SocketAttribute.local_port)
            if self._ssl_context:
                tcp_server = TLSListener(tcp_server, self._ssl_context,
                                         standard_compatible=False)
            task_status.started(self.port)

            services_dict = {}
            for key, value in self.services.items():
                services_dict[key] = await stack.enter_async_context(value)

            _logger.info(f"Starting server on {self.host}:{self.port}...")
            await tcp_server.serve(ConnectionHandler(services_dict, self))
