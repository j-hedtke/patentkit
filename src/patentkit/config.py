"""Central configuration and API-key handling.

Every connector, store, and LLM provider in patentkit follows the same
bring-your-own-key pattern:

1. an explicit ``api_key=...`` argument always wins,
2. otherwise the key is read from the environment variable listed below,
3. otherwise a :class:`MissingKeyError` is raised with a message that names
   the env var and where to obtain the key.

Use :class:`Keyring` to pass a bundle of keys around programmatically
(e.g. when keys come from a secrets manager rather than the environment).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


class MissingKeyError(RuntimeError):
    """Raised when a required API key is neither passed nor in the environment."""


#: env var name -> (human name, where to get it)
KEY_REGISTRY: dict[str, tuple[str, str]] = {
    "ANTHROPIC_API_KEY": ("Anthropic", "https://console.anthropic.com/settings/keys"),
    "OPENAI_API_KEY": ("OpenAI", "https://platform.openai.com/api-keys"),
    "VOYAGE_API_KEY": ("Voyage AI embeddings", "https://dash.voyageai.com/"),
    "USPTO_ODP_API_KEY": ("USPTO Open Data Portal", "https://data.uspto.gov/apis/getting-started"),
    "EPO_OPS_KEY": ("EPO Open Patent Services", "https://developers.epo.org/"),
    "EPO_OPS_SECRET": ("EPO Open Patent Services", "https://developers.epo.org/"),
    "SERPAPI_API_KEY": ("SerpApi (Google Patents search API)", "https://serpapi.com/manage-api-key"),
    "PATENTSVIEW_API_KEY": ("PatentsView", "https://patentsview.org/apis/keyrequest"),
    "RAINFOREST_API_KEY": ("Rainforest (Amazon product data)", "https://www.rainforestapi.com/"),
    "ELASTICSEARCH_HOST": ("Elasticsearch", "your cluster URL, e.g. https://localhost:9200"),
    "ELASTICSEARCH_API_KEY": ("Elasticsearch", "an API key for your cluster"),
    "SLACK_WEBHOOK_URL": ("Slack incoming webhook", "https://api.slack.com/messaging/webhooks"),
    "SENDGRID_API_KEY": ("SendGrid", "https://app.sendgrid.com/settings/api_keys"),
    "SMTP_URL": ("SMTP", "smtp://user:pass@host:587 form URL"),
}


def resolve_key(env_var: str, explicit: str | None = None, *, required: bool = True) -> str | None:
    """Resolve an API key: explicit argument first, then environment."""
    if explicit:
        return explicit
    value = os.environ.get(env_var)
    if value:
        return value
    if not required:
        return None
    name, where = KEY_REGISTRY.get(env_var, (env_var, "your provider"))
    raise MissingKeyError(
        f"No API key for {name}. Pass api_key=... explicitly or set the "
        f"{env_var} environment variable (obtain one at {where})."
    )


@dataclass
class Keyring:
    """A bundle of API keys that can be passed to any patentkit component.

    Field names mirror :data:`KEY_REGISTRY` env vars, lower-cased. Unset
    fields fall back to the environment when resolved.
    """

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    voyage_api_key: str | None = None
    uspto_odp_api_key: str | None = None
    epo_ops_key: str | None = None
    epo_ops_secret: str | None = None
    serpapi_api_key: str | None = None
    patentsview_api_key: str | None = None
    rainforest_api_key: str | None = None
    elasticsearch_host: str | None = None
    elasticsearch_api_key: str | None = None
    slack_webhook_url: str | None = None
    sendgrid_api_key: str | None = None
    smtp_url: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def get(self, env_var: str, *, required: bool = True) -> str | None:
        attr = env_var.lower()
        explicit = getattr(self, attr, None) if attr in {f.name for f in fields(self)} else None
        explicit = explicit or self.extra.get(env_var)
        return resolve_key(env_var, explicit, required=required)
