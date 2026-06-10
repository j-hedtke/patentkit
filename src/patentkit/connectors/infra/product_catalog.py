"""Popular-product catalogs for infringement targeting.

Infringement analysis starts from a universe of accused products. This
module offers three ways to assemble one:

- :class:`RainforestAmazonCatalog` — Amazon search results via the
  Rainforest API (https://www.rainforestapi.com/, key in
  ``RAINFOREST_API_KEY``).
- :class:`WebPageProductExtractor` — fetch any product/marketing page and
  have an LLM extract structured products with verbatim evidence quotes
  (needs BeautifulSoup: ``pip install patentkit[scrape]``).
- :class:`CatalogRegistry` — register your own catalog implementations
  (anything with a ``search(term) -> list[Product]``-style interface).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from patentkit.config import resolve_key
from patentkit.connectors.http import RateLimiter, request_json, request_text
from patentkit.llm.base import LLM, get_llm

logger = logging.getLogger(__name__)

RAINFOREST_URL = "https://api.rainforestapi.com/request"


class Product(BaseModel):
    """A product that may be an infringement target."""

    name: str
    brand: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    price: Optional[str] = None
    images: list[str] = Field(default_factory=list)
    source: Optional[str] = None  # e.g. "rainforest_amazon", "webpage_llm"
    raw: dict[str, Any] = Field(default_factory=dict)


class RainforestAmazonCatalog:
    """Amazon product search via the Rainforest API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        amazon_domain: str = "amazon.com",
        min_interval_s: float = 0.0,
    ):
        self.api_key = resolve_key("RAINFOREST_API_KEY", api_key)
        self.amazon_domain = amazon_domain
        self._rate_limiter = RateLimiter(min_interval_s)

    def search(self, search_term: str, *, max_results: int = 20) -> list[Product]:
        """Search Amazon and return up to ``max_results`` products."""
        data = request_json(
            "GET",
            RAINFOREST_URL,
            params={
                "api_key": self.api_key,
                "type": "search",
                "amazon_domain": self.amazon_domain,
                "search_term": search_term,
            },
            rate_limiter=self._rate_limiter,
        )
        products: list[Product] = []
        for result in (data.get("search_results") or [])[:max_results]:
            title = (result.get("title") or "").strip()
            if not title:
                continue
            price = result.get("price") or {}
            products.append(
                Product(
                    name=title,
                    brand=result.get("brand"),
                    url=result.get("link"),
                    price=str(price.get("raw") or price.get("value") or "") or None,
                    images=[img for img in [result.get("image")] if img],
                    source="rainforest_amazon",
                    raw=result,
                )
            )
        return products


_EXTRACTION_PROMPT = """\
You are extracting product listings from a web page for patent infringement \
analysis. Identify every distinct product offered or described on the page.

Return ONLY a JSON array. Each item:
{{
  "name": "product name",
  "brand": "brand or null",
  "description": "one-sentence functional description or null",
  "price": "price string or null",
  "evidence": ["short verbatim quote from the page supporting this product", ...]
}}

Page URL: {url}

Page text:
{text}
"""


class WebPageProductExtractor:
    """LLM-backed product extraction from an arbitrary web page."""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        *,
        max_page_chars: int = 30_000,
        min_interval_s: float = 0.0,
    ):
        self._llm = llm
        self.max_page_chars = max_page_chars
        self._rate_limiter = RateLimiter(min_interval_s)

    def _page_text(self, url: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "Web page extraction requires BeautifulSoup. Install it with "
                "`pip install patentkit[scrape]` (or `pip install beautifulsoup4`)."
            ) from exc
        html = request_text(url, rate_limiter=self._rate_limiter)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " \n".join(
            line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
        )
        return text[: self.max_page_chars]

    def extract(self, url: str) -> list[Product]:
        """Fetch ``url`` and extract Products (with evidence quotes in raw)."""
        text = self._page_text(url)
        llm = self._llm or get_llm("medium")
        items = llm.complete_json(_EXTRACTION_PROMPT.format(url=url, text=text))
        if not isinstance(items, list):
            logger.warning("LLM returned non-list product extraction for %s", url)
            return []
        products: list[Product] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            products.append(
                Product(
                    name=str(item["name"]),
                    brand=item.get("brand") or None,
                    description=item.get("description") or None,
                    price=str(item["price"]) if item.get("price") else None,
                    url=url,
                    source="webpage_llm",
                    raw={
                        **item,
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )
        return products


class CatalogRegistry:
    """Registry so users can plug in custom product catalogs by name."""

    def __init__(self) -> None:
        self._catalogs: dict[str, Any] = {}

    def register(self, name: str, catalog: Any) -> None:
        self._catalogs[name] = catalog

    def get(self, name: str) -> Any:
        try:
            return self._catalogs[name]
        except KeyError:
            raise KeyError(
                f"No catalog registered as {name!r}. Known: {sorted(self._catalogs)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._catalogs)
