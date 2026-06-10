"""Email notifiers: SendGrid (HTTPS API) and plain SMTP (stdlib smtplib).

Both follow the bring-your-own-key pattern (explicit argument, then env var:
``SENDGRID_API_KEY`` / ``SMTP_URL``).
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from urllib.parse import unquote, urlsplit

import httpx

from patentkit.config import resolve_key

logger = logging.getLogger(__name__)

SENDGRID_SEND_URL = "https://api.sendgrid.com/v3/mail/send"


class SendGridNotifier:
    """Sends mail via the SendGrid v3 ``mail/send`` API.

    Args:
        api_key: SendGrid API key; falls back to ``SENDGRID_API_KEY``.
        from_email: verified sender address.
        to_email: recipient address.
        timeout: HTTP timeout in seconds.
    """

    def __init__(self, api_key: str | None = None, *,
                 from_email: str, to_email: str, timeout: float = 30.0):
        self.api_key: str = resolve_key("SENDGRID_API_KEY", api_key)  # type: ignore[assignment]
        self.from_email = from_email
        self.to_email = to_email
        self.timeout = timeout

    def send(self, subject: str, body: str, **kwargs) -> None:
        """POST a plain-text mail; raises on HTTP error. ``kwargs`` are merged
        into the JSON payload (e.g. ``template_id=...``)."""
        payload = {
            "from": {"email": self.from_email},
            "personalizations": [{"to": [{"email": self.to_email}], "subject": subject}],
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
            **kwargs,
        }
        response = httpx.post(
            SENDGRID_SEND_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        logger.debug("SendGrid notification sent to %s: %s", self.to_email, subject)


class SmtpNotifier:
    """Sends mail through any SMTP server using the stdlib.

    The server is configured by a ``smtp://user:pass@host:port`` URL
    (``SMTP_URL`` env var by default). ``smtps://`` connects over implicit
    TLS; plain ``smtp://`` upgrades with STARTTLS when the server offers it.

    Args:
        smtp_url: connection URL; falls back to ``SMTP_URL``.
        from_email: sender address.
        to_email: recipient address.
    """

    def __init__(self, smtp_url: str | None = None, *, from_email: str, to_email: str):
        self.smtp_url: str = resolve_key("SMTP_URL", smtp_url)  # type: ignore[assignment]
        self.from_email = from_email
        self.to_email = to_email

    def send(self, subject: str, body: str, **kwargs) -> None:
        """Deliver a plain-text message; raises smtplib errors on failure."""
        parts = urlsplit(self.smtp_url)
        host = parts.hostname or "localhost"
        port = parts.port or (465 if parts.scheme == "smtps" else 587)
        user = unquote(parts.username) if parts.username else None
        password = unquote(parts.password) if parts.password else None

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.from_email
        message["To"] = self.to_email
        message.set_content(body)

        smtp_cls = smtplib.SMTP_SSL if parts.scheme == "smtps" else smtplib.SMTP
        with smtp_cls(host, port, timeout=30) as server:
            if parts.scheme != "smtps":
                try:
                    server.starttls()
                except smtplib.SMTPNotSupportedError:
                    logger.warning("SMTP server %s does not support STARTTLS; sending unencrypted", host)
            if user and password:
                server.login(user, password)
            server.send_message(message)
        logger.debug("SMTP notification sent to %s: %s", self.to_email, subject)


__all__ = ["SendGridNotifier", "SmtpNotifier", "SENDGRID_SEND_URL"]
