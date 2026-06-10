---
name: fto-search
description: Run a guided freedom-to-operate (FTO) screen for a product using the patentkit MCP tools. Use when the user asks whether a product/feature might infringe existing patents, wants a clearance screen, or asks "can we ship this without patent risk".
---

# Guided freedom-to-operate search

## Flow

1. **Describe the product.** Collect a concrete technical description of the product or feature (what it does, how it works, key components). The better the description, the better the keywords.
2. **Start.** Call `guided_search_start` with `search_type="fto"` and the `product_description`. **Present the plan and the estimated time up front**: planned queries, the jurisdiction (default US), and the in-force filter (patents filed within the last ~21 years — an approximation of "possibly still in force").
3. **Collect plan feedback** with `guided_search_feedback` (query verdicts + free text) and show the revision; iterate until approved.
4. **Execute** with `guided_search_execute`.
5. **Present results** as a ranked list with patent number, title, score, and the highlighted passages showing the claim/spec language closest to the product. Group obviously related families together when visible.
6. **Iterate** via result feedback (`guided_search_feedback`, then re-execute) until coverage feels adequate.

## Mandatory caveat

Always state clearly: this is a screening tool that surfaces patents **worth attorney review** — it is *not* a clearance or non-infringement opinion. Claim construction, actual term/maintenance-fee status, and prosecution history must be checked by qualified counsel (the result model carries `requires_attorney_review: true` for this reason).
