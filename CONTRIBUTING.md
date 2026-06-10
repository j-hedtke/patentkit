# Contributing to patentkit

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all]' pytest ruff
pytest -q
```

## Ground rules

- **Layering**: see `docs/ARCHITECTURE.md`. Lower layers never import higher
  layers; optional dependencies are lazily imported with an error naming the
  pip extra.
- **Bring your own keys**: any new external service must accept an explicit
  credential argument, fall back to an env var, and register that env var in
  `patentkit.config.KEY_REGISTRY`.
- **Canonical model**: connectors return `patentkit.models.Patent` (or the
  documented record types) with a `SourceRecord` for provenance — never raw
  source payloads.
- **Effort, not models**: LLM-using code declares `ReasoningEffort`; never
  hardcode a model id outside `patentkit/llm/routing.py`.
- **Offline tests**: the test suite must pass with only core dependencies and
  no network. Use `tests/fakes.FakeLLM`, monkeypatch `httpx`, and guard
  optional-dep tests with `pytest.importorskip`.

## Adding a backend or connector

1. Implement the relevant protocol (`KeywordStore` / `VectorStore`) or return
   canonical models.
2. Unit-test the pure translation logic (e.g. query building) without the
   service installed.
3. Document the env vars and where to obtain keys in the module docstring.
