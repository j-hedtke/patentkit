"""Local helper server for the patentkit Drafting Word add-in.

Stdlib-only HTTP server (no Flask) exposing the drafting skills over JSON:

- ``POST /api/draft-claims``  {"disclosure", "n_independent", "n_dependent"}
- ``POST /api/check-basis``   {"claims_text"}
- ``POST /api/draft-section`` {"disclosure", "section", "claims_text"?}

All responses: ``{"text": "..."}``. CORS is open so the Office.js task pane
(served from https://localhost:3000) can call it.

LLM provider credentials come from the environment: set ``ANTHROPIC_API_KEY``
(default provider) or ``OPENAI_API_KEY`` before starting. Run:

    python serve.py [port]
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from patentkit.analysis.drafting import check_antecedent_basis, draft_claims, draft_spec_section
from patentkit.parsing.claims import parse_claims

logger = logging.getLogger("patentkit.word_plugin")

DEFAULT_PORT = 8756


def _format_claims(claims) -> str:
    return "\n\n".join(f"{c.number}. {c.text}" for c in claims)


def handle_draft_claims(payload: dict) -> str:
    claims = draft_claims(
        invention_disclosure=str(payload.get("disclosure", "")),
        n_independent=int(payload.get("n_independent", 1)),
        n_dependent=int(payload.get("n_dependent", 5)),
    )
    return _format_claims(claims)


def handle_check_basis(payload: dict) -> str:
    claims = parse_claims(str(payload.get("claims_text", "")))
    if not claims:
        return "No numbered claims found in the selection."
    issues = check_antecedent_basis(claims)
    if not issues:
        return "Antecedent basis check: no issues found."
    return "Antecedent basis issues:\n" + "\n".join(f"- {issue}" for issue in issues)


def handle_draft_section(payload: dict) -> str:
    claims_text = str(payload.get("claims_text", "") or "")
    claims = parse_claims(claims_text) if claims_text else None
    return draft_spec_section(
        disclosure=str(payload.get("disclosure", "")),
        section=str(payload.get("section", "Summary")),
        claims=claims,
    )


ROUTES = {
    "/api/draft-claims": handle_draft_claims,
    "/api/check-basis": handle_check_basis,
    "/api/draft-section": handle_draft_section,
}


class Handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802 (http.server naming)
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        handler = ROUTES.get(self.path)
        if handler is None:
            self._reply(404, {"error": f"Unknown endpoint: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = handler(payload)
            self._reply(200, {"text": text})
        except Exception as exc:
            logger.exception("Request to %s failed", self.path)
            self._reply(500, {"error": str(exc)})

    def _reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        logger.warning(
            "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set; "
            "LLM-backed endpoints will fail until one is exported."
        )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(("localhost", port), Handler)
    logger.info("patentkit drafting server listening on http://localhost:%d", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
