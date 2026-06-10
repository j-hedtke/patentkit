---
name: infringement-search
description: Run a guided infringement (evidence-of-use) search ranking candidate products against a patent's claims using the patentkit MCP tools. Use when the user owns/asserts a patent and wants to find products that may practice it.
---

# Guided infringement search

## Flow

1. **Identify the patent and claims.** Get the patent number; call `get_patent` and confirm with the user which claims to assert (default: independent claims). Quote the key limitations back so the user can confirm scope.
2. **Gather candidates.** Ask the user for candidate products (names + descriptions + URLs, and any evidence text such as datasheets or marketing copy). The ranking is only as good as the candidate descriptions.
3. **Start** with `guided_search_start` (`search_type="infringement"`, the patent number, and claims). **Present the plan and estimated time up front.**
4. **Collect plan feedback** via `guided_search_feedback`; revise until approved.
5. **Execute** with `guided_search_execute` and present the ranked candidates: name, score (0-10 scale when an LLM is configured), and the per-candidate rationale explaining which limitations appear practiced.
6. **Iterate**: ask which candidates merit deeper analysis, feed verdicts back, and re-execute with refined candidates/evidence.
7. **Next step**: for the top candidates, offer an element-by-element analysis — map each claim limitation to specific evidence, flagging limitations with no visible evidence (those drive discovery).

## Caveat

Present results as leads for investigation, never as infringement conclusions; element-by-element analysis under a proper claim construction is counsel's job.
