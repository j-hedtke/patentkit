"""patentkit — modular open-source patent search & analysis toolkit.

Layers (each importable on its own):

- ``patentkit.models``       canonical patent data model
- ``patentkit.config``       bring-your-own-key configuration
- ``patentkit.llm``          provider-agnostic LLM with effort-based routing
- ``patentkit.connectors``   data sources (inference / infra / training)
- ``patentkit.search``       plug-and-play keyword + vector stores
- ``patentkit.parsing``      claim/PDF parsers incl. line-number regression
- ``patentkit.analysis``     invalidity / FTO / infringement / drafting skills
- ``patentkit.formatting``   claim charts and search reports (docx)
- ``patentkit.agents``       higher-level agentic search workflows
- ``patentkit.notify``       Slack / email completion notifications
- ``patentkit.viz``          patent set topic clustering
- ``patentkit.evals``        search-performance eval harness + datasets
- ``patentkit.integrations`` MCP server, OpenAI tools, Claude plugin
"""

__version__ = "0.1.0"

from patentkit.config import Keyring, MissingKeyError, resolve_key
from patentkit.models import Patent, PatentNumber

__all__ = ["Keyring", "MissingKeyError", "resolve_key", "Patent", "PatentNumber", "__version__"]
