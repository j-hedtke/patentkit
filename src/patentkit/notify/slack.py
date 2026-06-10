"""Slack incoming-webhook notifier.

Posts Block Kit payloads (header + mrkdwn section) to a Slack incoming
webhook. The webhook URL follows the bring-your-own-key pattern:
explicit argument first, then the ``SLACK_WEBHOOK_URL`` env var.
"""

from __future__ import annotations

import logging

import httpx

from patentkit.config import resolve_key

logger = logging.getLogger(__name__)


class SlackNotifier:
    """Sends messages to a Slack incoming webhook.

    Args:
        webhook_url: the webhook URL; falls back to ``SLACK_WEBHOOK_URL``
            (create one at https://api.slack.com/messaging/webhooks).
        timeout: HTTP timeout in seconds.
    """

    def __init__(self, webhook_url: str | None = None, timeout: float = 30.0):
        self.webhook_url: str = resolve_key("SLACK_WEBHOOK_URL", webhook_url)  # type: ignore[assignment]
        self.timeout = timeout

    def send(self, subject: str, body: str, **kwargs) -> None:
        """POST a header + mrkdwn-section blocks payload; raises on HTTP error.

        Extra ``kwargs`` are merged into the payload (e.g. ``channel=...``
        for legacy webhooks).
        """
        payload = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": subject[:150], "emoji": True}},
                {"type": "section", "text": {"type": "mrkdwn", "text": body[:2900]}},
            ],
            **kwargs,
        }
        response = httpx.post(
            self.webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        logger.debug("Slack notification sent: %s", subject)


__all__ = ["SlackNotifier"]
