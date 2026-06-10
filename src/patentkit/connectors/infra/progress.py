"""Resumable-job checkpointing.

:class:`FileProgressTracker` persists a tiny JSON checkpoint (processed
count, failed ids, last processed id) so long-running ingestion jobs —
bulk scraping, dataset builds — can crash/stop and resume where they left
off. Writes are atomic (temp file + ``os.replace``) so a checkpoint is never
half-written.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class FileProgressTracker:
    """JSON-file checkpoint for resumable batch jobs.

    Usage::

        tracker = FileProgressTracker("job.progress.json")
        for item_id in items:
            ...
            tracker.record_success(item_id)  # or record_failure(item_id)
        tracker.save()

    On restart, ``FileProgressTracker(same_path).resume_after()`` returns the
    id of the last successfully processed item (or None for a fresh start);
    callers skip input items up to and including that id. Failed ids are kept
    for later retry/inspection.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.processed: int = 0
        self.failed: list[str] = []
        self.last_processed_id: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            state = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read checkpoint %s: %s", self.path, exc)
            return
        self.processed = int(state.get("processed", 0))
        self.failed = list(state.get("failed", []))
        self.last_processed_id = state.get("last_processed_id")

    def record_success(self, item_id: str) -> None:
        self.processed += 1
        self.last_processed_id = item_id

    def record_failure(self, item_id: str) -> None:
        if item_id not in self.failed:
            self.failed.append(item_id)

    def resume_after(self) -> Optional[str]:
        """Id of the last successfully processed item, or None."""
        return self.last_processed_id

    def save(self) -> None:
        """Atomically write the checkpoint (temp file + os.replace)."""
        state = {
            "processed": self.processed,
            "failed": self.failed,
            "last_processed_id": self.last_processed_id,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=self.path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(state, handle, indent=2)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
