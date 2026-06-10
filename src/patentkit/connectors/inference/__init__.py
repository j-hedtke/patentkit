"""Inference-time connectors: fetch one patent's data on demand."""

from patentkit.connectors.inference.file_wrapper import (
    FileWrapperClient,
    FileWrapperDocument,
)
from patentkit.connectors.inference.google_patents import (
    GooglePatentsScraper,
    SerpApiGooglePatentsSearch,
    fetch_patent,
)

__all__ = [
    "FileWrapperClient",
    "FileWrapperDocument",
    "GooglePatentsScraper",
    "SerpApiGooglePatentsSearch",
    "fetch_patent",
]
