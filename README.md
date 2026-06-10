# patentkit

Modular, open-source toolkit for patent search and analysis. patentkit packages
the building blocks of a production patent-intelligence stack — data
connectors, a canonical patent data model, plug-and-play search infrastructure,
LLM analysis skills, agentic search workflows, document formatters, and evals —
and exposes them as **tools, skills, and plugins for both Anthropic (Claude /
MCP) and OpenAI (function tools / Agents SDK)**.

```
pip install patentkit            # core (pure python + pydantic/httpx)
pip install 'patentkit[all]'     # everything
```

Extras: `anthropic`, `openai`, `elasticsearch`, `docx`, `pdf`, `viz`, `mcp`, `scrape`.

## Design principles

- **Bring your own keys.** Every connector, store, and model provider accepts
  an explicit `api_key=...` and falls back to a documented env var
  (`patentkit.config.KEY_REGISTRY` lists them all). Nothing phones home.
- **Sensible model defaults, routed by reasoning effort.** Tasks declare
  `low` / `medium` / `high` effort, not model ids. Defaults: Claude Haiku 4.5 /
  Sonnet 4.6 / Fable 5 (Anthropic) and gpt-5-mini / gpt-5.1 (OpenAI), all
  overridable in one place (`patentkit.llm.DEFAULT_MODELS`).
- **Everything degrades gracefully.** The whole pipeline runs offline with the
  in-memory BM25 store and no LLM key (LLM stages are skipped); add keys and
  backends to turn quality up.
- **One canonical data model.** Every source parses into
  `patentkit.models.Patent`; `Patent.merge()` reconciles records from multiple
  sources with per-source fidelity and provenance (`sources`).

## Layout

| Package | What's in it |
|---|---|
| `patentkit.models` | Canonical patent data model + multi-source reconciliation |
| `patentkit.connectors.inference` | USPTO file wrapper (ODP), Google Patents page + search API |
| `patentkit.connectors.infra` | Google Patents large-scale scrape jobs, USPTO bulk XML, EPO OPS, PTAB/IPR, ETSI SEP declarations, product catalogs (Amazon via Rainforest, LLM web extraction) |
| `patentkit.connectors.training` | Examiner search-query log builder (from SRNT search reports), IPR eval-dataset builder |
| `patentkit.search` | `SearchQuery` full param set; BM25 (in-memory), Elasticsearch, vector/RAG stores (in-memory, ES dense-vector; OpenAI/Voyage embeddings), hybrid RRF fusion |
| `patentkit.parsing` | Claim parser, document text extraction, **PDF line-number regression** (cite passages as "col. 3, ll. 45–52") |
| `patentkit.analysis` | Invalidity (atomic limitations → disclosure assessment → claim charts), FTO, infringement, drafting skills + prompt library |
| `patentkit.formatting` | Claim charts (docx/markdown/html, color-coded, line-number citations), invalidity / FTO / infringement reports |
| `patentkit.agents` | Agentic invalidity / FTO / infringement search, invalidity charting, **guided search sessions** (plan → feedback → execute → iterate), time estimation |
| `patentkit.notify` | Slack webhook, SendGrid / SMTP email completion notifications |
| `patentkit.viz` | Patent-set topic clustering (KMeans/DBSCAN + LLM topic naming) |
| `patentkit.evals` | Search-performance harness, recall@k / MRR / MAP metrics, toy IPR dataset, user-built eval sets |
| `patentkit.integrations` | MCP server (`patentkit-mcp`), OpenAI tool definitions + agent loop, Word drafting add-in |
| `plugins/claude` | Claude Code plugin: skills (invalidity/FTO/infringement search, claim chart, drafting) + MCP wiring |

## Quickstart (offline, no keys)

```python
from patentkit.models import Patent, PatentNumber
from patentkit.search import BM25Store, SearchQuery
from patentkit.agents import InvaliditySearchAgent

store = BM25Store()
store.index(my_patents)                      # any Iterable[Patent]

agent = InvaliditySearchAgent(keyword_store=store)   # no LLM: degraded mode
target = store.get(PatentNumber.parse("US10123456B2"))
result = agent.search(target)                # examiner art excluded by default
for r in result.results[:10]:
    print(r["patent_number"], r["score"], r["passages"][0]["text"][:80])
```

See `examples/quickstart.py` for a runnable end-to-end demo.

## With LLMs and real data

```python
from patentkit.llm import get_llm
from patentkit.connectors.inference.google_patents import fetch_patent
from patentkit.connectors.inference.file_wrapper import FileWrapperClient

llm = get_llm("high")                        # -> claude-fable-5 by default
# or: get_llm("high", provider="openai")     # -> gpt-5.1 (reasoning effort high)

patent = fetch_patent("US10123456B2")        # Google Patents scrape -> canonical Patent
patent = FileWrapperClient().enrich_patent(patent)   # + prosecution history, examiner art

agent = InvaliditySearchAgent(keyword_store=store, vector_store=vstore, llm=llm)
result = agent.search(patent, claims=[1])
```

## Guided search (the user-facing flow)

The guided flow is what the Claude skills / OpenAI tools drive:

1. `guided_search_start` — an LLM drafts a **search plan**: a series of
   queries with purposes, art classes, include/exclude tokens, date cutoffs,
   plus an **estimated completion time**.
2. The user reviews the plan; feedback on whole results, individual passages,
   or individual queries is structured (`patentkit.agents.feedback`).
3. `guided_search_execute` — runs the staged pipeline (keyword → hybrid
   rerank → LLM relevance), returning ranked results with **highlighted
   relevant passages**. Examiner-cited art from the file wrapper, family
   members, and the target itself are excluded by default (overridable).
4. `guided_search_feedback` — revises the plan and iterates.

Sessions are JSON-serializable, so the loop works across chat turns in any
agent harness.

## Exposing patentkit to Claude and OpenAI

**MCP (Claude Desktop, Claude Code, any MCP client):**

```jsonc
// .mcp.json
{ "mcpServers": { "patentkit": { "command": "patentkit-mcp" } } }
```

**Claude Code plugin** — `plugins/claude/` ships skills
(`/invalidity-search`, `/fto-search`, `/infringement-search`, `/claim-chart`,
`/patent-drafting`) plus the MCP server wiring.

**OpenAI:**

```python
from patentkit.integrations.toolset import PatentToolset
from patentkit.integrations.openai_tools import openai_tool_definitions, handle_tool_call

tools = openai_tool_definitions()   # pass to responses.create(tools=...)
# dispatch tool calls back through handle_tool_call(toolset, name, args_json)
```

**Word add-in** — `integrations/word-plugin/` is an Office.js taskpane for
patent drafting (draft claims, check antecedent basis, draft spec sections)
backed by a local patentkit HTTP server.

## Evals

```python
from patentkit.evals import EvalRunner, default_ipr_toy_dataset

report = EvalRunner(my_search_fn, default_ipr_toy_dataset()).run()
print(report.to_markdown())   # recall@k curves, MRR, MAP
```

Ships with a clearly-labeled toy IPR dataset; build real ones with
`patentkit.connectors.training.ipr_datasets` (PTAB final decisions → ground
truth) or from user feedback via `UserEvalSetBuilder`.

## Legal note

patentkit produces research aids, not legal advice. Invalidity, FTO, and
infringement outputs require review by qualified patent counsel.

## License

Apache-2.0
