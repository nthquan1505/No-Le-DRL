"""Lightweight thread-safe inter-module signal bus (`src/signals.py`).

Decouples producers (e.g., `src/extraction`) from downstream consumers
(e.g., `src/classification`, `src/credit_calculation`, `app.py`) without passing
full domain payloads in memory: producers emit lightweight completion signals,
and downstream modules query `drl_database.db` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.request
import json

logger = logging.getLogger(__name__)

SIGNAL_EXTRACTION_COMPLETED = "extraction.completed"
SIGNAL_CLASSIFICATION_COMPLETED = "classification.completed"

SignalListener = Callable[[Any], Any]


@dataclass(frozen=True)
class ExtractionCompletedSignal:
    """Emitted by `ExtractionPipeline` after persisting extracted records to `drl_database.db`."""

    db: Any
    created_count: int
    updated_count: int
    cancelled_count: int
    created_event_ids: Tuple[int, ...] = ()
    updated_event_ids: Tuple[int, ...] = ()
    cancelled_event_ids: Tuple[int, ...] = ()

    @property
    def has_unclassified_changes(self) -> bool:
        return bool(self.created_count > 0 or self.updated_event_ids)

    def to_webhook_dict(self) -> Dict[str, Any]:
        return {
            "signal": SIGNAL_EXTRACTION_COMPLETED,
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "cancelled_count": self.cancelled_count,
            "created_event_ids": list(self.created_event_ids),
            "updated_event_ids": list(self.updated_event_ids),
            "cancelled_event_ids": list(self.cancelled_event_ids),
        }


@dataclass(frozen=True)
class ClassificationCompletedSignal:
    """Emitted by `ClassificationPipeline` after querying `drl_database.db` and updating `events.category`."""

    db: Any
    total_queried_events: int
    db_categories_updated: int
    status_counts: Dict[str, int] = field(default_factory=dict)


class SignalBus:
    """Thread-safe synchronous pub/sub bus for intra-process module signaling."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._listeners: Dict[str, List[SignalListener]] = {}

    def subscribe(self, signal_name: str, listener: SignalListener) -> Callable[[], None]:
        """Register `listener(signal_payload)` for `signal_name` (idempotent). Returns an unsubscribe callable."""
        with self._lock:
            bucket = self._listeners.setdefault(signal_name, [])
            if listener not in bucket:
                bucket.append(listener)

        def _unsubscribe() -> None:
            self.unsubscribe(signal_name, listener)

        return _unsubscribe

    def unsubscribe(self, signal_name: str, listener: SignalListener) -> bool:
        with self._lock:
            bucket = self._listeners.get(signal_name)
            if not bucket or listener not in bucket:
                return False
            bucket.remove(listener)
            return True

    def clear(self, signal_name: Optional[str] = None) -> None:
        with self._lock:
            if signal_name is None:
                self._listeners.clear()
            else:
                self._listeners.pop(signal_name, None)

    def listener_count(self, signal_name: str) -> int:
        with self._lock:
            return len(self._listeners.get(signal_name, ()))

    def emit(self, signal_name: str, payload: Any) -> List[Any]:
        """Dispatch `payload` to all registered listeners, isolating exceptions per listener."""
        with self._lock:
            listeners = list(self._listeners.get(signal_name, ()))

        outputs: List[Any] = []
        for listener in listeners:
            try:
                outputs.append(listener(payload))
            except Exception as exc:
                logger.warning("Signal listener %r failed on %s: %s", listener, signal_name, exc)
        return outputs


# Global process-wide signal bus
signal_bus = SignalBus()


def post_signal_webhook(url: str, payload: Dict[str, Any], timeout: float = 3.0) -> bool:
    """Optional cross-process HTTP signal notifier (e.g., scraper process -> running `app.py` server)."""
    clean_url = (url or "").strip()
    if not clean_url:
        return False
    req = urllib.request.Request(
        clean_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception as exc:
        logger.debug("Cross-process signal webhook %s unreachable: %s", clean_url, exc)
        return False


def get_configured_classifier_webhook_url() -> Optional[str]:
    """Read optional `CLASSIFIER_SIGNAL_URL` environment variable for cross-process signaling."""
    val = os.environ.get("CLASSIFIER_SIGNAL_URL", "").strip()
    return val or None
