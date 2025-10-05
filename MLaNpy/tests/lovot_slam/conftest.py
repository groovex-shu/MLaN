from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from .mock.mock_map_builder import mock_maps_from_maps_vertices as mock_maps_from_maps_vertices


@pytest.fixture
def mock_httpx(monkeypatch):
    mock_post = AsyncMock(
        return_value=httpx.Response(
            default_encoding='utf-8',
            status_code=201,
            headers=[('test', 'header')],
        )
    )
    client = Mock()
    client.post = mock_post

    @asynccontextmanager
    async def async_client():
        yield client

    monkeypatch.setattr('httpx.AsyncClient', async_client)
    yield mock_post
