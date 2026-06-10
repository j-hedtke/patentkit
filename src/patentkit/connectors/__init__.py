"""Data connectors: every external patent data source, one import away.

Inference connectors fetch single records at analysis time; infra
connectors do bulk/offline ingestion; training connectors build datasets.
All connectors return the canonical types from
:mod:`patentkit.models.patent` with provenance ``SourceRecord``s
(fidelity: google_patents=3, uspto_odp=2, uspto_bulk=2, epo_ops=1).
"""

from patentkit.connectors.inference import (
    FileWrapperClient,
    FileWrapperDocument,
    GooglePatentsScraper,
    SerpApiGooglePatentsSearch,
    fetch_patent,
)
from patentkit.connectors.infra import (
    BulkScrapeJob,
    BulkScrapeStats,
    CatalogRegistry,
    EpoOpsClient,
    EtsiSepConnector,
    FileProgressTracker,
    IprDocument,
    IprProceeding,
    Product,
    PtabClient,
    RainforestAmazonCatalog,
    SepDeclaration,
    WebPageProductExtractor,
    load_etsi_declarations_csv,
    sep_patents_for_standard,
)
from patentkit.connectors.training import (
    ExaminerLogBuilder,
    ExaminerQueryRecord,
    IprEvalRecord,
    build_ipr_eval_dataset,
    load_ipr_eval_dataset,
    parse_search_queries,
)

__all__ = [
    "BulkScrapeJob",
    "BulkScrapeStats",
    "CatalogRegistry",
    "EpoOpsClient",
    "EtsiSepConnector",
    "ExaminerLogBuilder",
    "ExaminerQueryRecord",
    "FileProgressTracker",
    "FileWrapperClient",
    "FileWrapperDocument",
    "GooglePatentsScraper",
    "IprDocument",
    "IprEvalRecord",
    "IprProceeding",
    "Product",
    "PtabClient",
    "RainforestAmazonCatalog",
    "SepDeclaration",
    "SerpApiGooglePatentsSearch",
    "WebPageProductExtractor",
    "build_ipr_eval_dataset",
    "fetch_patent",
    "load_etsi_declarations_csv",
    "load_ipr_eval_dataset",
    "parse_search_queries",
    "sep_patents_for_standard",
]
