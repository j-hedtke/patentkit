"""Infra connectors: bulk/offline ingestion of patent and product data."""

from patentkit.connectors.infra.epo import EpoOpsClient
from patentkit.connectors.infra.google_patents_bulk import (
    BulkScrapeJob,
    BulkScrapeStats,
)
from patentkit.connectors.infra.product_catalog import (
    CatalogRegistry,
    Product,
    RainforestAmazonCatalog,
    WebPageProductExtractor,
)
from patentkit.connectors.infra.progress import FileProgressTracker
from patentkit.connectors.infra.ptab import IprDocument, IprProceeding, PtabClient
from patentkit.connectors.infra.sep import (
    EtsiSepConnector,
    SepDeclaration,
    load_etsi_declarations_csv,
    sep_patents_for_standard,
)
from patentkit.connectors.infra.uspto_bulk import (
    download_archive,
    iter_patents_from_archive,
    parse_redbook_xml,
    weekly_archive_urls,
)

__all__ = [
    "BulkScrapeJob",
    "BulkScrapeStats",
    "CatalogRegistry",
    "EpoOpsClient",
    "EtsiSepConnector",
    "FileProgressTracker",
    "IprDocument",
    "IprProceeding",
    "Product",
    "PtabClient",
    "RainforestAmazonCatalog",
    "SepDeclaration",
    "WebPageProductExtractor",
    "download_archive",
    "iter_patents_from_archive",
    "load_etsi_declarations_csv",
    "parse_redbook_xml",
    "sep_patents_for_standard",
    "weekly_archive_urls",
]
