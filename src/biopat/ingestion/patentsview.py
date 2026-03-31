"""PatentsView API client for USPTO patent data.

Provides async access to the PatentsView API for retrieving patent metadata,
claims, and IPC classifications.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import httpx
import polars as pl
from diskcache import Cache
from tqdm.asyncio import tqdm_asyncio

from biopat.reproducibility import AuditLogger
from biopat.ingestion.retry import retry_with_backoff

logger = logging.getLogger(__name__)

PATENTSVIEW_BASE_URL = "https://search.patentsview.org/api/v1"
DEFAULT_RATE_LIMIT = 45  # requests per minute with API key
DEFAULT_BATCH_SIZE = 100


class ApiKeyPool:
    """Manages a pool of API keys with individual rate limits."""

    def __init__(self, api_keys: List[str], rate_limit: int = DEFAULT_RATE_LIMIT):
        self.keys = api_keys
        self.rate_limit = rate_limit
        self._semaphores = {key: asyncio.Semaphore(1) for key in api_keys}
        self._request_times: Dict[str, List[float]] = {key: [] for key in api_keys}
        self._index = 0
        self._lock = asyncio.Lock()

    async def get_key_and_wait(self) -> str:
        """Get next available key and respect its rate limit."""
        async with self._lock:
            key = self.keys[self._index]
            self._index = (self._index + 1) % len(self.keys)

        async with self._semaphores[key]:
            await self._rate_limit_wait(key)
            return key

    async def _rate_limit_wait(self, key: str):
        """Enforce rate limiting for a specific key."""
        import time
        now = time.time()
        self._request_times[key] = [t for t in self._request_times[key] if now - t < 60]

        if len(self._request_times[key]) >= self.rate_limit:
            sleep_time = 60 - (now - self._request_times[key][0])
            if sleep_time > 0:
                logger.debug(f"Rate limiting key {key[:8]}...: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        self._request_times[key].append(time.time())


class PatentsViewClient:
    """Async client for PatentsView API with multi-key rotation."""

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        audit_logger: Optional[AuditLogger] = None,
    ):
        # Allow single string or list
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        
        self.api_keys = api_keys or []
        self.rate_limit = rate_limit
        self.cache = Cache(str(cache_dir / "patentsview")) if cache_dir else None
        self.audit_logger = audit_logger
        
        if self.api_keys:
            self._pool = ApiKeyPool(self.api_keys, rate_limit)
        else:
            self._pool = None
            self._semaphore = asyncio.Semaphore(rate_limit)
            self._request_times: List[float] = []

    def _get_headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-Api-Key"] = api_key
        return headers

    async def _rate_limit_wait(self):
        """Enforce rate limiting for requests without a key pool (legacy/single)."""
        import time
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self._request_times.append(time.time())

    async def _make_request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """Make a rate-limited API request with key rotation."""
        if self._pool:
            api_key = await self._pool.get_key_and_wait()
            # Key rotation handled by the pool
        else:
            async with self._semaphore:
                await self._rate_limit_wait()
                api_key = None

        url = f"{PATENTSVIEW_BASE_URL}/{endpoint}"
        
        try:
            headers = self._get_headers(api_key)
            if method == "POST":
                response = await retry_with_backoff(
                    lambda: client.post(url, headers=headers, json=body, timeout=60.0)
                )
            else:
                response = await retry_with_backoff(
                    lambda: client.get(url, headers=headers, params=params, timeout=60.0)
                )
            response.raise_for_status()
            result = response.json()

            # Log API call
            if self.audit_logger:
                response_count = None
                if isinstance(result, dict):
                    if "patents" in result:
                        response_count = len(result["patents"])
                    elif "patent_id" in result:
                        response_count = 1

                self.audit_logger.log_api_call(
                    service="patentsview",
                    endpoint=endpoint,
                    method=method,
                    params=params or body,
                    response_status=response.status_code,
                    response_count=response_count,
                )

            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    async def get_patent(
        self,
        patent_id: str,
        fields: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a single patent by ID.

        Args:
            patent_id: USPTO patent ID (e.g., "US10123456B2").
            fields: Fields to return. If None, returns default fields.

        Returns:
            Patent data dict or None if not found.
        """
        # Check cache
        cache_key = f"patent:{patent_id}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        if fields is None:
            fields = [
                "patent_id",
                "patent_date",
                "patent_title",
                "patent_abstract",
                "claims",
                "application",
                "ipc_current",
                "cpc_current",
                "assignees_at_grant",
                "inventors",
            ]

        async with httpx.AsyncClient(follow_redirects=True) as client:
            result = await self._make_request(
                client,
                f"patent/{patent_id}",
                params={"f": json.dumps(fields)},
            )

        # Cache result
        if self.cache and result:
            self.cache[cache_key] = result

        return result

    async def search_patents(
        self,
        query: Dict[str, Any],
        fields: Optional[List[str]] = None,
        size: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search patents with query.

        Args:
            query: Query object following PatentsView syntax.
            fields: Fields to return.
            size: Number of results per page (max 1000).
            cursor: Pagination cursor from previous request.

        Returns:
            Search results with patents and pagination info.
        """
        if fields is None:
            fields = [
                "patent_id",
                "patent_date",
                "patent_title",
                "application",
                "ipc_current",
            ]

        body = {
            "q": query,
            "f": fields,
            "o": {"size": size},
        }

        if cursor:
            body["o"]["after"] = cursor

        async with httpx.AsyncClient() as client:
            return await self._make_request(client, "patent/", body=body, method="POST")

    async def get_biomedical_patents(
        self,
        ipc_prefixes: List[str] = ["A61", "C07", "C12"],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get patents in biomedical IPC classes.

        Args:
            ipc_prefixes: IPC code prefixes to filter (e.g., ["A61", "C07"]).
            limit: Maximum number of patents to return.

        Returns:
            List of patent data dicts.
        """
        # Build OR query for IPC prefixes
        ipc_conditions = [
            {"_begins": {"ipc_current.ipc_class": prefix}}
            for prefix in ipc_prefixes
        ]
        query = {"_or": ipc_conditions}

        patents = []
        cursor = None

        async with httpx.AsyncClient(follow_redirects=True) as client:
            with tqdm_asyncio(desc="Fetching biomedical patents", unit="patents") as pbar:
                while True:
                    body = {
                        "q": query,
                        "f": [
                            "patent_id",
                            "patent_date",
                            "patent_title",
                            "patent_abstract",
                            "application",
                            "ipc_current",
                        ],
                        "o": {"size": DEFAULT_BATCH_SIZE},
                    }

                    if cursor:
                        body["o"]["after"] = cursor

                    result = await self._make_request(
                        client, "patent/", body=body, method="POST"
                    )

                    batch = result.get("patents", [])
                    if not batch:
                        break

                    patents.extend(batch)
                    pbar.update(len(batch))

                    if limit and len(patents) >= limit:
                        patents = patents[:limit]
                        break

                    # Get cursor for next page
                    cursor = result.get("cursor")
                    if not cursor:
                        break

        return patents

    async def get_patents_batch(
        self,
        patent_ids: List[str],
        fields: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get multiple patents by ID.

        Args:
            patent_ids: List of patent IDs.
            fields: Fields to return.
            show_progress: Show progress bar.

        Returns:
            List of patent data dicts.
        """
        if fields is None:
            fields = [
                "patent_id",
                "patent_date",
                "patent_title",
                "patent_abstract",
                "claims",
                "application",
                "ipc_current",
                "cpc_current",
            ]

        results = []

        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Process in batches
            tasks = []
            for pid in patent_ids:
                # Check cache first
                cache_key = f"patent:{pid}"
                if self.cache and cache_key in self.cache:
                    results.append(self.cache[cache_key])
                else:
                    tasks.append(self._get_single_patent(client, pid, fields))

            if tasks:
                if show_progress:
                    fetched = await tqdm_asyncio.gather(
                        *tasks, desc="Fetching patents"
                    )
                else:
                    fetched = await asyncio.gather(*tasks, return_exceptions=True)

                for item in fetched:
                    if isinstance(item, Exception):
                        logger.warning(f"Failed to fetch patent: {item}")
                    elif item:
                        results.append(item)
                        # Cache result
                        if self.cache and "patent_id" in item:
                            self.cache[f"patent:{item['patent_id']}"] = item

        return results

    async def _get_single_patent(
        self,
        client: httpx.AsyncClient,
        patent_id: str,
        fields: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single patent with rate limiting."""
        try:
            result = await self._make_request(
                client,
                f"patent/{patent_id}",
                params={"f": json.dumps(fields)},
            )
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch patent {patent_id}: {e}")
            return None

    def patents_to_dataframe(self, patents: List[Dict[str, Any]]) -> pl.DataFrame:
        """Convert patent data to Polars DataFrame.

        Args:
            patents: List of patent data dicts.

        Returns:
            DataFrame with patent data.
        """
        records = []
        for p in patents:
            # Extract IPC codes
            ipc_codes = []
            if "ipc_current" in p and p["ipc_current"]:
                for ipc in p["ipc_current"]:
                    if "ipc_class" in ipc:
                        ipc_codes.append(ipc["ipc_class"])

            # Extract priority date from application
            priority_date = None
            filing_date = None
            if "application" in p and p["application"]:
                app = p["application"][0] if isinstance(p["application"], list) else p["application"]
                priority_date = app.get("earliest_application_date")
                filing_date = app.get("application_date")

            # Extract claims
            claims = []
            if "claims" in p and p["claims"]:
                for c in p["claims"]:
                    claims.append({
                        "claim_number": c.get("claim_number"),
                        "claim_text": c.get("claim_text", ""),
                        "claim_type": c.get("claim_type", ""),
                        "depends_on": c.get("depends_on"),
                    })

            records.append({
                "patent_id": p.get("patent_id"),
                "patent_date": p.get("patent_date"),
                "priority_date": priority_date,
                "filing_date": filing_date,
                "title": p.get("patent_title"),
                "abstract": p.get("patent_abstract"),
                "ipc_codes": ipc_codes,
                "claims": claims,
            })

        return pl.DataFrame(records)

    async def get_patents_for_corpus(
        self,
        patent_ids: List[str],
        show_progress: bool = True,
    ) -> pl.DataFrame:
        """Fetch patents for corpus assembly (v2.0 dual-corpus).

        Optimized batch retrieval for building the patent corpus with
        titles, abstracts, and claims for BEIR formatting.

        Args:
            patent_ids: List of patent IDs to fetch.
            show_progress: Show progress bar.

        Returns:
            DataFrame with patent data ready for corpus formatting.
        """
        # Fields needed for corpus: title, abstract, claims, dates
        fields = [
            "patent_id",
            "patent_date",
            "patent_title",
            "patent_abstract",
            "claims",
            "application",
            "ipc_current",
        ]

        logger.info(f"Fetching {len(patent_ids)} patents for corpus assembly")

        # Deduplicate IDs
        unique_ids = list(set(patent_ids))

        # Use batch fetching
        patents = await self.get_patents_batch(
            unique_ids, fields=fields, show_progress=show_progress
        )

        logger.info(f"Fetched {len(patents)} patents successfully")

        return self.patents_to_dataframe(patents)

    async def search_patents_by_ids(
        self,
        patent_ids: List[str],
        fields: Optional[List[str]] = None,
        batch_size: int = 50,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search patents by multiple IDs using OR filter.

        More efficient than individual fetches for large ID lists.

        Args:
            patent_ids: List of patent IDs.
            fields: Fields to return.
            batch_size: Number of IDs per query batch.
            show_progress: Show progress bar.

        Returns:
            List of patent data dicts.
        """
        if fields is None:
            fields = [
                "patent_id",
                "patent_date",
                "patent_title",
                "patent_abstract",
                "claims",
                "application",
                "ipc_current",
            ]

        results = []
        batches = [
            patent_ids[i:i + batch_size]
            for i in range(0, len(patent_ids), batch_size)
        ]

        async with httpx.AsyncClient(follow_redirects=True) as client:
            iterator = batches
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(batches, desc="Fetching patent batches")

            for batch in iterator:
                # Build OR query for patent IDs
                query = {
                    "_or": [{"patent_id": pid} for pid in batch]
                }

                body = {
                    "q": query,
                    "f": fields,
                    "o": {"size": len(batch)},
                }

                try:
                    result = await self._make_request(
                        client, "patent/", body=body, method="POST"
                    )
                    batch_patents = result.get("patents", [])
                    results.extend(batch_patents)

                    # Cache individual patents
                    if self.cache:
                        for p in batch_patents:
                            if "patent_id" in p:
                                self.cache[f"patent:{p['patent_id']}"] = p

                except Exception as e:
                    logger.warning(f"Batch query failed: {e}, falling back to individual")
                    # Fallback to individual fetches for this batch
                    for pid in batch:
                        try:
                            p = await self._get_single_patent(client, pid, fields)
                            if p:
                                results.append(p)
                        except Exception:
                            pass

        logger.info(f"Retrieved {len(results)} patents from {len(patent_ids)} IDs")
        return results
