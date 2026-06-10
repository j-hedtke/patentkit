"""Completion notifications: Slack webhooks, SendGrid, and plain SMTP."""

from patentkit.notify.base import Notifier, format_completion_message, notify_search_complete
from patentkit.notify.email import SendGridNotifier, SmtpNotifier
from patentkit.notify.slack import SlackNotifier

__all__ = [
    "Notifier",
    "format_completion_message",
    "notify_search_complete",
    "SendGridNotifier",
    "SmtpNotifier",
    "SlackNotifier",
]
