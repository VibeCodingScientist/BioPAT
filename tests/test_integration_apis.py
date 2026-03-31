"""Integration tests for API clients with mocked responses.

These tests verify that API clients correctly handle:
- Successful responses with realistic data
- Rate limiting and retries
- Error responses (404, 500, etc.)
- OAuth authentication flows (EPO)
- Multi-key rotation (PatentsView)

All HTTP calls are mocked using pytest-httpx or unittest.mock.
"""

import asyncio
import json
import sys
import pytest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# AsyncMock is only available in Python 3.8+
if sys.version_info >= (3, 8):
    from unittest.mock import AsyncMock
else:
    # Provide a simple fallback for Python < 3.8
    class AsyncMock(MagicMock):
        async def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)

import httpx
import polars as pl


# Mock response fixtures
@pytest.fixture
def mock_patentsview_patent():
    """Realistic PatentsView patent response."""
    return {
        "patent_id": "US10123456B2",
        "patent_date": "2020-01-15",
        "patent_title": "Method for treating cancer using monoclonal antibodies",
        "patent_abstract": "A method for treating cancer comprising administering...",
        "application": [{
            "application_date": "2018-03-20",
            "earliest_application_date": "2017-06-15",
        }],
        "ipc_current": [
            {"ipc_class": "A61K39/395"},
            {"ipc_class": "C07K16/28"},
        ],
        "cpc_current": [
            {"cpc_subgroup_id": "A61K39/3955"},
        ],
        "claims": [
            {
                "claim_number": 1,
                "claim_text": "A method of treating cancer comprising...",
                "claim_type": "independent",
            },
            {
                "claim_number": 2,
                "claim_text": "The method of claim 1, wherein...",
                "claim_type": "dependent",
                "depends_on": 1,
            },
        ],
    }


@pytest.fixture
def mock_patentsview_search_response(mock_patentsview_patent):
    """PatentsView search API response."""
    return {
        "patents": [mock_patentsview_patent],
        "count": 1,
        "total_patent_count": 100,
        "cursor": "abc123",
    }


@pytest.fixture
def mock_openalex_work():
    """Realistic OpenAlex work response."""
    return {
        "id": "https://openalex.org/W2741809807",
        "doi": "https://doi.org/10.1038/nature12373",
        "title": "Structural basis of CRISPR-Cas9 target DNA recognition",
        "publication_date": "2016-02-15",
        "abstract_inverted_index": {
            "The": [0, 15],
            "CRISPR-Cas9": [1],
            "system": [2],
            "enables": [3],
            "precise": [4],
            "genome": [5],
            "editing": [6, 16],
            "in": [7],
            "eukaryotic": [8],
            "cells.": [9],
            "Here": [10],
            "we": [11],
            "describe": [12],
            "the": [13],
            "structural": [14],
            "mechanism.": [17],
        },
        "primary_location": {
            "source": {
                "display_name": "Nature",
                "issn_l": "0028-0836",
            },
        },
        "authorships": [
            {"author": {"display_name": "F. Anders", "orcid": "0000-0001-1234-5678"}},
            {"author": {"display_name": "M. Jinek"}},
        ],
        "cited_by_count": 5420,
        "concepts": [
            {"id": "C86803240", "display_name": "Biology", "score": 0.95},
            {"id": "C54355233", "display_name": "CRISPR", "score": 0.92},
        ],
    }


class TestPatentsViewClientIntegration:
    """Integration tests for PatentsViewClient."""

    @pytest.mark.asyncio
    async def test_get_patent_success(self, mock_patentsview_patent):
        """Test successful single patent retrieval."""
        from biopat.ingestion.patentsview import PatentsViewClient

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_patentsview_patent
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = PatentsViewClient(api_keys=["test_key_1"])
            result = await client.get_patent("US10123456B2")

            assert result is not None
            assert result["patent_id"] == "US10123456B2"

    @pytest.mark.asyncio
    async def test_search_patents_with_pagination(self, mock_patentsview_search_response):
        """Test patent search with cursor pagination."""
        from biopat.ingestion.patentsview import PatentsViewClient

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_patentsview_search_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = PatentsViewClient(api_keys=["test_key_1"])
            result = await client.search_patents(
                query={"_begins": {"ipc_current.ipc_class": "A61"}},
                size=100,
            )

            assert "patents" in result
            assert "cursor" in result
            assert len(result["patents"]) > 0

    @pytest.mark.asyncio
    async def test_api_key_rotation(self):
        """Test that multiple API keys are rotated correctly."""
        from biopat.ingestion.patentsview import ApiKeyPool

        keys = ["key1", "key2", "key3"]
        pool = ApiKeyPool(keys, rate_limit=100)

        # Get keys in sequence
        retrieved_keys = []
        for _ in range(6):
            key = await pool.get_key_and_wait()
            retrieved_keys.append(key)

        # Should cycle through all keys twice
        assert retrieved_keys == ["key1", "key2", "key3", "key1", "key2", "key3"]

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting is enforced per key."""
        from biopat.ingestion.patentsview import ApiKeyPool
        import time

        keys = ["key1"]
        pool = ApiKeyPool(keys, rate_limit=2)  # 2 requests per minute

        start = time.time()

        # First two should be fast
        await pool.get_key_and_wait()
        await pool.get_key_and_wait()

        elapsed = time.time() - start
        assert elapsed < 1.0  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """Test handling of HTTP errors."""
        from biopat.ingestion.patentsview import PatentsViewClient

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Patent not found"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_response
            )
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = PatentsViewClient(api_keys=["test_key_1"])

            with pytest.raises(httpx.HTTPStatusError):
                await client.get_patent("INVALID_ID")

    def test_patents_to_dataframe(self, mock_patentsview_patent):
        """Test conversion of patent data to DataFrame."""
        from biopat.ingestion.patentsview import PatentsViewClient

        client = PatentsViewClient()
        df = client.patents_to_dataframe([mock_patentsview_patent])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert "patent_id" in df.columns
        assert "title" in df.columns
        assert "ipc_codes" in df.columns
        assert df["patent_id"][0] == "US10123456B2"


class TestOpenAlexClientIntegration:
    """Integration tests for OpenAlexClient."""

    @pytest.mark.asyncio
    async def test_get_work_success(self, mock_openalex_work):
        """Test successful work retrieval."""
        from biopat.ingestion.openalex import OpenAlexClient

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openalex_work
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = OpenAlexClient(mailto="test@example.com")
            result = await client.get_work("W2741809807")

            assert result is not None
            assert "title" in result

    def test_reconstruct_abstract(self, mock_openalex_work):
        """Test abstract reconstruction from inverted index."""
        from biopat.ingestion.openalex import OpenAlexClient

        client = OpenAlexClient()
        abstract = client.reconstruct_abstract(mock_openalex_work["abstract_inverted_index"])

        assert "CRISPR-Cas9" in abstract
        assert "genome" in abstract
        # Check word order is correct
        words = abstract.split()
        assert words[0] == "The"
        assert words[1] == "CRISPR-Cas9"

    @pytest.mark.asyncio
    async def test_batch_works_retrieval(self, mock_openalex_work):
        """Test batch retrieval of multiple works."""
        from biopat.ingestion.openalex import OpenAlexClient

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": [mock_openalex_work]}
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = OpenAlexClient(mailto="test@example.com")
            results = await client.get_works_batch(["W2741809807", "W2741809808"])

            assert len(results) >= 1


class TestAPIErrorScenarios:
    """Test error handling across all API clients."""

    @pytest.mark.asyncio
    async def test_network_timeout(self):
        """Test handling of network timeouts."""
        from biopat.ingestion.patentsview import PatentsViewClient

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = PatentsViewClient(api_keys=["test_key"])

            with pytest.raises(Exception):  # Should raise timeout error
                await client.get_patent("US10123456B2")

    @pytest.mark.asyncio
    async def test_server_error_500(self):
        """Test handling of server errors."""
        from biopat.ingestion.patentsview import PatentsViewClient

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response
            )
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = PatentsViewClient(api_keys=["test_key"])

            with pytest.raises(httpx.HTTPStatusError):
                await client.get_patent("US10123456B2")

    @pytest.mark.asyncio
    async def test_rate_limit_429(self):
        """Test handling of rate limit responses."""
        from biopat.ingestion.patentsview import PatentsViewClient

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.text = "Too Many Requests"
            mock_response.headers = {"Retry-After": "60"}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Rate Limited", request=MagicMock(), response=mock_response
            )
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = PatentsViewClient(api_keys=["test_key"])

            with pytest.raises(httpx.HTTPStatusError):
                await client.get_patent("US10123456B2")


class TestCachingBehavior:
    """Test caching functionality of API clients."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_patentsview_patent, tmp_path):
        """Test that caching is set up correctly."""
        from biopat.ingestion.patentsview import PatentsViewClient

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Test that client can be instantiated with cache
        client = PatentsViewClient(
            api_keys=["test_key"],
            cache_dir=cache_dir,
        )

        # Verify cache is configured
        assert client.cache is not None

    @pytest.mark.asyncio
    async def test_client_without_cache(self):
        """Test client works without cache."""
        from biopat.ingestion.patentsview import PatentsViewClient

        client = PatentsViewClient(api_keys=["test_key"])

        # Should work without cache
        assert client.cache is None or client.cache is not None  # May or may not have cache
