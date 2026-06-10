"""Tests for the notify layer (no network: httpx.post is monkeypatched)."""

from __future__ import annotations

import httpx
import pytest

from patentkit.notify import (
    SendGridNotifier,
    SlackNotifier,
    format_completion_message,
    notify_search_complete,
)


class _FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        pass


@pytest.fixture
def posted(monkeypatch) -> list[dict]:
    """Capture every httpx.post call as {url, json, headers}."""
    calls: list[dict] = []

    def fake_post(url, *, json=None, headers=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


# -------------------------------------------------------------------- slack

def test_slack_notifier_payload_shape(posted: list[dict]) -> None:
    notifier = SlackNotifier(webhook_url="https://hooks.slack.example/T000/B000/xyz")
    notifier.send("Invalidity search complete", "3 references found\n*top*: US7000001B1")

    assert len(posted) == 1
    call = posted[0]
    assert call["url"] == "https://hooks.slack.example/T000/B000/xyz"
    assert call["headers"]["Content-Type"] == "application/json"
    blocks = call["json"]["blocks"]
    assert blocks[0]["type"] == "header"
    assert blocks[0]["text"]["type"] == "plain_text"
    assert blocks[0]["text"]["text"] == "Invalidity search complete"
    assert blocks[1]["type"] == "section"
    assert blocks[1]["text"]["type"] == "mrkdwn"
    assert "US7000001B1" in blocks[1]["text"]["text"]


def test_slack_notifier_env_key(monkeypatch, posted: list[dict]) -> None:
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/env")
    SlackNotifier().send("s", "b")
    assert posted[0]["url"] == "https://hooks.slack.example/env"


def test_slack_notifier_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    from patentkit.config import MissingKeyError
    with pytest.raises(MissingKeyError):
        SlackNotifier()


# ----------------------------------------------------------------- sendgrid

def test_sendgrid_notifier_payload_shape(posted: list[dict]) -> None:
    notifier = SendGridNotifier("SG.test-key", from_email="bot@example.com",
                                to_email="user@example.com")
    notifier.send("Search done", "Results attached")

    call = posted[0]
    assert call["url"] == "https://api.sendgrid.com/v3/mail/send"
    assert call["headers"]["Authorization"] == "Bearer SG.test-key"
    payload = call["json"]
    assert payload["from"] == {"email": "bot@example.com"}
    personalization = payload["personalizations"][0]
    assert personalization["to"] == [{"email": "user@example.com"}]
    assert personalization["subject"] == "Search done"
    assert payload["content"] == [{"type": "text/plain", "value": "Results attached"}]


# --------------------------------------------------- completion formatting

_RESULT = {
    "search_type": "invalidity",
    "target": "US8123456B2",
    "results": [
        {"patent_number": "US7000001B1", "title": "Wireless soil moisture sensor network",
         "score": 0.91},
        {"patent_number": "US7000002B1", "title": "Soil moisture probe", "score": 0.74},
        {"patent_number": "US7000004B1", "title": "Greenhouse telemetry", "score": 0.31},
        {"patent_number": "US7000005B1", "title": "Fourth result never shown", "score": 0.10},
    ],
    "timing": {"total": 12.34},
}


def test_format_completion_message() -> None:
    subject, body = format_completion_message(_RESULT, link="https://example.com/run/1")
    assert subject == "patentkit: invalidity search complete — US8123456B2"
    assert "Results: 4" in body
    assert "US7000001B1" in body and "Wireless soil moisture sensor network" in body
    assert "Fourth result never shown" not in body  # only top 3 titles
    assert "12.3" in body                            # elapsed
    assert "https://example.com/run/1" in body


def test_notify_search_complete_fans_out_and_survives_failures() -> None:
    class Recorder:
        def __init__(self):
            self.messages = []

        def send(self, subject, body, **kwargs):
            self.messages.append((subject, body))

    class Broken:
        def send(self, subject, body, **kwargs):
            raise RuntimeError("webhook down")

    good_a, good_b = Recorder(), Recorder()
    sent = notify_search_complete([good_a, Broken(), good_b], _RESULT)
    assert sent == 2
    assert good_a.messages == good_b.messages
    subject, body = good_a.messages[0]
    assert "invalidity" in subject and "US8123456B2" in subject
    assert "Results: 4" in body


def test_notify_search_complete_with_session_like_object() -> None:
    class SessionLike:
        search_type = "fto"
        last_results = [{"patent_number": "US7000001B1", "title": "T", "score": 1.0}]
        params = {"elapsed_seconds": 3.0}
        plan = type("Plan", (), {"target": "smart widget"})()

    messages = []

    class Recorder:
        def send(self, subject, body, **kwargs):
            messages.append((subject, body))

    assert notify_search_complete([Recorder()], SessionLike()) == 1
    subject, body = messages[0]
    assert "fto" in subject and "smart widget" in subject
    assert "Results: 1" in body
