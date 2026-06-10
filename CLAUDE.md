# patentkit — working notes for Claude

Open-source modular patent search & analysis toolkit. Orientation:
`README.md` (what/why), `docs/ARCHITECTURE.md` (layering rules, agentic
search design, model routing). Tests: `.venv/bin/python -m pytest -q` —
the suite must pass offline with core deps only (FakeLLM in `tests/fakes.py`
scripts tool-use rounds; no network, no SDKs).

## Hard-won constraints (do not regress)

- Searches are **pure agentic tool-use loops** (Anthropic Messages tool use /
  OpenAI Responses function tools): the model writes queries, executes them as
  tools, reads results, refines, decides when to stop — seconds to minutes,
  saved reasoning trace, resumable conversation for feedback. Never reintroduce
  staged brute-force pipelines or fixed-weight score blending.
- Exclusions (examiner art, family, self) and prior-art date cutoffs are
  enforced in the **tool layer**, never only in prompts.
- Eval corpora are **real data** (live Google Patents scrapes seeded from IPR
  examples), stored in managed Elasticsearch — not synthetic fixtures.
- Always label degraded keys-free results as such; never report them as
  agentic-mode performance.

## Lessons

Persistent lessons live in the project memory directory
(`~/.claude/projects/-Users-joshhedtke-patentkit/memory/`, indexed by
`MEMORY.md`). Policy: **one lesson per file with a one-line summary at the
top** (the `description:` frontmatter line). Record corrections and confirmed
approaches alike, including *why* they mattered. Don't save what the repo or
chat history already records; update an existing note rather than creating a
duplicate; delete notes that turn out to be wrong.
