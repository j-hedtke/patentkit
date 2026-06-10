---
name: fto-search
description: Run a guided freedom-to-operate (FTO) screen for a product using the patentkit MCP tools. Use when the user asks whether a product/feature might infringe existing patents, wants a clearance screen, or asks "can we ship this without patent risk".
---

# Guided freedom-to-operate search

Execution is agentic: one LLM agent iteratively generates queries from the product description, runs them as tools, inspects hits, and finishes with ranked candidates — with a full saved reasoning trace.

## Flow

1. **Describe the product.** Collect a concrete technical description of the product or feature (what it does, how it works, key components). The better the description, the better the agent's queries.
2. **Start.** Call `guided_search_start` with `search_type="fto"` and the `product_description`. **Present the plan preview and the estimated time up front**: the starting angles (the agent generates its own queries at run time), the jurisdiction (default US), and the in-force filter (patents filed within the last ~21 years — an approximation of "possibly still in force").
3. **Collect plan feedback** with `guided_search_feedback` (angle verdicts + free text — seeded into the agent's initial guidance); iterate until approved.
4. **Execute** with `guided_search_execute`.
5. **Show the trace.** Call `get_search_trace` and walk the user through the queries the agent issued and its shortlist evolution, so they can critique query coverage as well as results.
6. **Present results** as a ranked list with patent number, title, confidence, the *why* sentence, and the key passages showing the claim/spec language closest to the product. Group obviously related families together when visible.
7. **Iterate** via result/query feedback (`guided_search_feedback`, then re-execute — feedback is injected into the SAME resumed agent conversation) until coverage feels adequate.

## Mandatory caveat

Always state clearly: this is a screening tool that surfaces patents **worth attorney review** — it is *not* a clearance or non-infringement opinion. Claim construction, actual term/maintenance-fee status, and prosecution history must be checked by qualified counsel (the result model carries `requires_attorney_review: true` for this reason).
