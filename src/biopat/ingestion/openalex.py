"""OpenAlex API client for scientific paper metadata.

Provides async access to OpenAlex for retrieving paper titles,
abstracts, publication dates, and concept tags.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx
import polars as pl
from diskcache import Cache
from tqdm.asyncio import tqdm_asyncio

from biopat.reproducibility import AuditLogger
from biopat.ingestion.retry import retry_with_backoff

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org"
DEFAULT_RATE_LIMIT = 10  # polite pool requests per second
DEFAULT_BATCH_SIZE = 50  # max IDs per filter request


def normalize_openalex_id(raw_id: str) -> str:
    """Normalize an OpenAlex ID to the W-prefixed short form.

    RoS dataset stores bare numeric IDs like '2100342970'.
    OpenAlex API requires 'W2100342970' or the full URL.

    Args:
        raw_id: Raw OpenAlex ID in any format.

    Returns:
        Normalized ID like 'W2100342970'.
    """
    raw_id = str(raw_id).strip()
    # Full URL: https://openalex.org/W2100342970
    if raw_id.startswith("https://"):
        return raw_id.split("/")[-1]
    # Already prefixed: W2100342970
    if raw_id.startswith("W") and raw_id[1:].isdigit():
        return raw_id
    # Bare numeric: 2100342970
    if raw_id.isdigit():
        return f"W{raw_id}"
    return raw_id


class OpenAlexClient:
    """Async client for OpenAlex API."""

    def __init__(
        self,
        mailto: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.mailto = mailto
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.cache = Cache(str(cache_dir / "openalex")) if cache_dir else None
        self._semaphore = asyncio.Semaphore(rate_limit)
        self.audit_logger = audit_logger

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.mailto:
            headers["User-Agent"] = f"mailto:{self.mailto}"
        return headers

    @staticmethod
    def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
        """Reconstruct abstract text from OpenAlex inverted index format.

        Args:
            inverted_index: Dict mapping words to their positions.

        Returns:
            Reconstructed abstract string.
        """
        if not inverted_index:
            return ""

        # Create position -> word mapping
        position_word = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word[pos] = word

        # Sort by position and join
        if not position_word:
            return ""

        max_pos = max(position_word.keys())
        words = [position_word.get(i, "") for i in range(max_pos + 1)]
        return " ".join(w for w in words if w)

    async def _make_request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Make a rate-limited API request."""
        async with self._semaphore:
            url = f"{OPENALEX_BASE_URL}/{endpoint}"
            if params is None:
                params = {}
            if self.api_key:
                params["api_key"] = self.api_key

            try:
                response = await retry_with_backoff(
                    lambda: client.get(
                        url,
                        headers=self._get_headers(),
                        params=params,
                        timeout=60.0,
                    )
                )
                response.raise_for_status()
                result = response.json()

                # Log API call
                if self.audit_logger:
                    response_count = None
                    if isinstance(result, dict):
                        if "results" in result:
                            response_count = len(result["results"])
                        elif "id" in result:
                            response_count = 1

                    self.audit_logger.log_api_call(
                        service="openalex",
                        endpoint=endpoint,
                        method="GET",
                        params=params,
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

    async def get_work(self, openalex_id: str) -> Optional[Dict[str, Any]]:
        """Get a single work by OpenAlex ID.

        Args:
            openalex_id: OpenAlex work ID (e.g., "W2123456789").

        Returns:
            Work data dict or None if not found.
        """
        work_id = normalize_openalex_id(openalex_id)

        # Check cache
        cache_key = f"work:{work_id}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        async with httpx.AsyncClient() as client:
            try:
                result = await self._make_request(client, f"works/{work_id}")

                # Cache result
                if self.cache and result:
                    self.cache[cache_key] = result

                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                raise

    async def get_works_batch(
        self,
        openalex_ids: List[str],
        show_progress: bool = True,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get multiple works by OpenAlex ID.

        Args:
            openalex_ids: List of OpenAlex work IDs.
            show_progress: Show progress bar.
            select_fields: If set, only return these fields (reduces memory).

        Returns:
            List of work data dicts.
        """
        results = []
        ids_to_fetch = [normalize_openalex_id(oid) for oid in openalex_ids]

        if not ids_to_fetch:
            return results

        # Fetch in batches using filter endpoint
        async with httpx.AsyncClient() as client:
            batches = [
                ids_to_fetch[i:i + DEFAULT_BATCH_SIZE]
                for i in range(0, len(ids_to_fetch), DEFAULT_BATCH_SIZE)
            ]

            if show_progress:
                iterator = tqdm_asyncio(
                    batches, desc="Fetching papers", unit="batch"
                )
            else:
                iterator = batches

            for batch in iterator:
                # Use filter endpoint for batch retrieval
                filter_str = "|".join(batch)
                params = {
                    "filter": f"openalex_id:{filter_str}",
                    "per-page": len(batch),
                }
                if select_fields:
                    params["select"] = ",".join(select_fields)

                try:
                    response = await self._make_request(client, "works", params=params)
                    works = response.get("results", [])
                    results.extend(works)

                except Exception as e:
                    logger.warning(f"Failed to fetch batch: {e}")

                # Small delay between batches
                await asyncio.sleep(0.1)

        return results

    async def search_works(
        self,
        query: Optional[str] = None,
        filter_str: Optional[str] = None,
        per_page: int = 50,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search works with query or filter.

        Args:
            query: Search query string.
            filter_str: Filter string (e.g., "concepts.id:C71924100").
            per_page: Results per page.
            cursor: Pagination cursor.

        Returns:
            Search results with works and pagination info.
        """
        params = {"per-page": per_page}

        if query:
            params["search"] = query
        if filter_str:
            params["filter"] = filter_str
        if cursor:
            params["cursor"] = cursor

        async with httpx.AsyncClient() as client:
            return await self._make_request(client, "works", params=params)

    async def get_biomedical_works(
        self,
        concept_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get works in biomedical concepts.

        Args:
            concept_ids: OpenAlex concept IDs. Defaults to Medicine, Biology, Chemistry.
            limit: Maximum number of works to return.

        Returns:
            List of work data dicts.
        """
        if concept_ids is None:
            # Default biomedical concepts
            concept_ids = [
                "C71924100",   # Medicine
                "C86803240",   # Biology
                "C185592680",  # Chemistry
            ]

        filter_str = "|".join(f"concepts.id:{cid}" for cid in concept_ids)

        works = []
        cursor = "*"

        async with httpx.AsyncClient() as client:
            with tqdm_asyncio(desc="Fetching biomedical works", unit="works") as pbar:
                while True:
                    params = {
                        "filter": filter_str,
                        "per-page": 200,
                        "cursor": cursor,
                    }

                    result = await self._make_request(client, "works", params=params)
                    batch = result.get("results", [])

                    if not batch:
                        break

                    works.extend(batch)
                    pbar.update(len(batch))

                    if limit and len(works) >= limit:
                        works = works[:limit]
                        break

                    # Get next cursor
                    meta = result.get("meta", {})
                    cursor = meta.get("next_cursor")
                    if not cursor:
                        break

                    await asyncio.sleep(0.1)

        return works

    def works_to_dataframe(self, works: List[Dict[str, Any]]) -> pl.DataFrame:
        """Convert work data to Polars DataFrame.

        Args:
            works: List of work data dicts.

        Returns:
            DataFrame with paper data.
        """
        records = []
        for w in works:
            # Get OpenAlex ID
            openalex_id = w.get("id", "")
            if openalex_id.startswith("https://"):
                openalex_id = openalex_id.split("/")[-1]

            # Reconstruct abstract
            abstract = ""
            if "abstract_inverted_index" in w and w["abstract_inverted_index"]:
                abstract = self.reconstruct_abstract(w["abstract_inverted_index"])

            # Extract concepts
            concepts = []
            if "concepts" in w and w["concepts"]:
                for c in w["concepts"]:
                    concepts.append({
                        "id": c.get("id", "").split("/")[-1],
                        "name": c.get("display_name", ""),
                        "score": c.get("score", 0),
                    })

            # Get external IDs
            ids = w.get("ids", {})
            pmid = ids.get("pmid", "")
            if pmid and pmid.startswith("https://"):
                pmid = pmid.split("/")[-1]
            doi = ids.get("doi", "")
            if doi and doi.startswith("https://"):
                doi = doi.replace("https://doi.org/", "")

            # Get authors
            authors = []
            if "authorships" in w and w["authorships"]:
                for auth in w["authorships"]:
                    if "author" in auth and auth["author"]:
                        authors.append(auth["author"].get("display_name", ""))

            # Get journal/source
            journal = ""
            if "primary_location" in w and w["primary_location"]:
                source = w["primary_location"].get("source")
                if source:
                    journal = source.get("display_name", "")

            records.append({
                "paper_id": openalex_id,
                "pmid": pmid or None,
                "doi": doi or None,
                "title": w.get("title", ""),
                "abstract": abstract,
                "publication_date": w.get("publication_date"),
                "journal": journal or None,
                "authors": authors,
                "concepts": concepts,
                "cited_by_count": w.get("cited_by_count", 0),
            })

        return pl.DataFrame(records)

    # Fields needed for the benchmark — requesting only these saves memory
    PAPER_SELECT_FIELDS = [
        "id", "title", "abstract_inverted_index", "publication_date",
        "ids", "authorships", "primary_location", "cited_by_count",
        "concepts",
    ]

    async def get_papers_for_citations(
        self,
        openalex_ids: List[str],
        show_progress: bool = True,
        chunk_size: int = 5000,
    ) -> pl.DataFrame:
        """Fetch papers and return as DataFrame.

        Processes in chunks to avoid OOM on large corpora.

        Args:
            openalex_ids: List of OpenAlex work IDs.
            show_progress: Show progress bar.
            chunk_size: Number of IDs per chunk (controls peak memory).

        Returns:
            DataFrame with paper data.
        """
        dfs = []
        for i in range(0, len(openalex_ids), chunk_size):
            chunk_ids = openalex_ids[i:i + chunk_size]
            logger.info(
                f"Fetching paper chunk {i // chunk_size + 1}/"
                f"{(len(openalex_ids) + chunk_size - 1) // chunk_size} "
                f"({len(chunk_ids)} papers)"
            )
            works = await self.get_works_batch(
                chunk_ids,
                show_progress=show_progress,
                select_fields=self.PAPER_SELECT_FIELDS,
            )
            if works:
                dfs.append(self.works_to_dataframe(works))
            # Free raw dicts immediately
            del works

        if not dfs:
            return pl.DataFrame()
        return pl.concat(dfs)
