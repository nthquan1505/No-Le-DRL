"""Option B Direct REST LLM Client (`src/utility/extraction/llm_client.py`).

Uses Python's standard `urllib.request` with Gemini's server-side `responseSchema`
(`responseMimeType: "application/json"`) so zero external SDKs are required.

Features:
- Multi-key round-robin load balancing & `HTTP 429`/`403` cooldowns (`GeminiKeyPool`)
- Automatic model failover (`GEMINI_MODEL` -> `gemini-2.0-flash`)
- Secondary Groq OpenAI-compatible REST failover (`GROQ_API_KEY`)
- Guaranteed per-item deterministic fallback (`heuristic_extract_item`)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import urllib.error
import urllib.request

from .contracts import ExtractedEventDraft, RawIngestionItem
from .heuristics import heuristic_extract_item, sanitize_and_enrich_draft
from .prompts import GEMINI_BATCH_RESPONSE_SCHEMA, SYSTEM_INSTRUCTION, build_batch_prompt

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
FALLBACK_GEMINI_MODELS: Tuple[str, ...] = ("gemini-2.5-flash", "gemini-2.0-flash")
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BATCH_SIZE = 5

HttpPostCallable = Callable[[str, Dict[str, str], Dict[str, Any], float], Tuple[int, Dict[str, str], str]]


def _default_http_post(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float = 35.0,
) -> Tuple[int, Dict[str, str], str]:
    """Perform a standard-library HTTPS JSON POST and return `(status_code, headers, body_text)`."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json; charset=utf-8", **headers}
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.getcode() or 200)
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            body = resp.read().decode("utf-8", errors="replace")
            return status, resp_headers, body
    except urllib.error.HTTPError as exc:
        resp_headers = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return int(exc.code), resp_headers, err_body


def _strip_json_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class GeminiKeyPool:
    """Round-robin pool of Gemini API keys with automatic rate-limit cooldowns."""

    def __init__(
        self,
        keys: Optional[Sequence[str]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.keys: List[str] = []
        self._index: int = 0
        self._cooldown_until: Dict[str, float] = {}
        self._clock = clock
        if keys is not None:
            self.add_keys(keys)
        else:
            self.load_from_env()

    def add_keys(self, raw: Any) -> None:
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                self.add_keys(item)
        elif isinstance(raw, str) and raw.strip():
            for part in re.split(r"[,\s;]+", raw.strip()):
                clean = part.strip().strip("'\"")
                if clean and clean not in self.keys:
                    self.keys.append(clean)

    def load_from_env(self) -> None:
        for idx in range(1, 11):
            self.add_keys(os.environ.get(f"GEMINI_API_KEY_{idx}"))
        self.add_keys(os.environ.get("GEMINI_API_KEYS"))
        self.add_keys(os.environ.get("GEMINI_API_KEY"))

    @property
    def has_keys(self) -> bool:
        return len(self.keys) > 0

    def next_available_key(self) -> Optional[str]:
        if not self.keys:
            return None
        now = self._clock()
        n = len(self.keys)
        for offset in range(n):
            idx = (self._index + offset) % n
            key = self.keys[idx]
            if self._cooldown_until.get(key, 0.0) <= now:
                self._index = (idx + 1) % n
                return key
        return None

    def mark_rate_limited(self, key: str, retry_after_seconds: float = 60.0) -> None:
        self._cooldown_until[key] = self._clock() + max(1.0, retry_after_seconds)

    def mark_invalid(self, key: str, cooldown_seconds: float = 86400.0) -> None:
        self._cooldown_until[key] = self._clock() + max(60.0, cooldown_seconds)


class DirectLLMExtractor:
    """Batch-capable Direct REST extractor (Gemini -> Groq -> Deterministic Heuristic)."""

    def __init__(
        self,
        key_pool: Optional[GeminiKeyPool] = None,
        gemini_model: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        batch_size: Optional[int] = None,
        timeout_seconds: float = 35.0,
        http_post: Optional[HttpPostCallable] = None,
    ) -> None:
        self.key_pool = key_pool if key_pool is not None else GeminiKeyPool()
        self.gemini_model = (
            gemini_model
            or os.environ.get("GEMINI_MODEL", "").strip()
            or DEFAULT_GEMINI_MODEL
        )
        self.groq_api_key = (
            groq_api_key
            if groq_api_key is not None
            else os.environ.get("GROQ_API_KEY", "").strip()
        )
        self.groq_model = (
            groq_model
            or os.environ.get("GROQ_MODEL", "").strip()
            or DEFAULT_GROQ_MODEL
        )
        env_batch = os.environ.get("GEMINI_BATCH_SIZE", "").strip()
        self.batch_size = max(
            1,
            batch_size if batch_size is not None else (int(env_batch) if env_batch.isdigit() else DEFAULT_BATCH_SIZE),
        )
        self.timeout_seconds = timeout_seconds
        self._http_post: HttpPostCallable = http_post or _default_http_post

    def extract_batch(self, items: Sequence[RawIngestionItem]) -> List[ExtractedEventDraft]:
        """Extract `1..N` candidate items in chunks of `self.batch_size`."""
        if not items:
            return []
        all_drafts: List[ExtractedEventDraft] = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            all_drafts.extend(self._extract_chunk(chunk))
        return all_drafts

    def _extract_chunk(self, chunk: Sequence[RawIngestionItem]) -> List[ExtractedEventDraft]:
        user_prompt = build_batch_prompt(chunk)

        # 1. Primary: Direct Gemini REST API with server-side `responseSchema`
        parsed_rows = self._try_gemini(user_prompt)
        if parsed_rows is not None:
            return self._align_and_sanitize(chunk, parsed_rows, method="gemini")

        # 2. Secondary: Direct Groq OpenAI-compatible REST API
        groq_rows = self._try_groq(user_prompt)
        if groq_rows is not None:
            return self._align_and_sanitize(chunk, groq_rows, method="groq")

        # 3. Offline / All-cloud-down deterministic fallback
        return [heuristic_extract_item(item) for item in chunk]

    def _candidate_gemini_models(self) -> List[str]:
        models = [self.gemini_model]
        for m in FALLBACK_GEMINI_MODELS:
            if m not in models:
                models.append(m)
        return models

    def _try_gemini(self, user_prompt: str) -> Optional[List[Dict[str, Any]]]:
        if not self.key_pool.has_keys:
            return None

        payload: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_BATCH_RESPONSE_SCHEMA,
            },
        }

        for model in self._candidate_gemini_models():
            # Try up to the number of configured keys per model
            max_attempts = max(1, len(self.key_pool.keys))
            for _ in range(max_attempts):
                key = self.key_pool.next_available_key()
                if not key:
                    break
                url = f"{GEMINI_API_BASE}/{model}:generateContent"
                headers = {"x-goog-api-key": key}
                try:
                    status, resp_headers, body = self._http_post(
                        url, headers, payload, self.timeout_seconds
                    )
                except Exception as exc:
                    logger.warning("Gemini network error on model=%s: %s", model, exc)
                    continue

                if status == 200:
                    rows = self._parse_gemini_response(body)
                    if rows is not None:
                        return rows
                    logger.warning("Gemini returned non-list JSON on model=%s", model)
                    break

                if status == 429:
                    retry_hdr = resp_headers.get("retry-after", "")
                    wait_s = float(retry_hdr) if retry_hdr.isdigit() else 60.0
                    self.key_pool.mark_rate_limited(key, wait_s)
                    continue

                if status in (401, 403):
                    self.key_pool.mark_invalid(key)
                    continue

                if status == 404:
                    # Model name unavailable; move to next candidate model
                    break

                logger.warning("Gemini HTTP %s on model=%s: %s", status, model, body[:200])
                break

        return None

    @staticmethod
    def _parse_gemini_response(body: str) -> Optional[List[Dict[str, Any]]]:
        try:
            envelope = json.loads(body)
            candidates = envelope.get("candidates") or []
            if not candidates:
                return None
            parts = ((candidates[0].get("content") or {}).get("parts")) or []
            if not parts:
                return None
            raw_json = _strip_json_fences(str(parts[0].get("text") or ""))
            parsed = json.loads(raw_json)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                return [x for x in parsed["items"] if isinstance(x, dict)]
        except Exception:
            return None
        return None

    def _try_groq(self, user_prompt: str) -> Optional[List[Dict[str, Any]]]:
        if not self.groq_api_key:
            return None

        groq_prompt = (
            user_prompt
            + '\n\nTrả về JSON object dạng {"items": [ ... ]} chứa mảng kết quả cho từng `source_id`.'
        )
        payload: Dict[str, Any] = {
            "model": self.groq_model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": groq_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "User-Agent": "No-Le-DRL-Extractor/2.0",
        }
        try:
            status, _, body = self._http_post(GROQ_CHAT_URL, headers, payload, self.timeout_seconds)
            if status != 200:
                logger.warning("Groq HTTP %s: %s", status, body[:200])
                return None
            envelope = json.loads(body)
            content = (
                ((envelope.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            )
            parsed = json.loads(_strip_json_fences(content))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            if isinstance(parsed, dict):
                for key in ("items", "events", "results", "data"):
                    if isinstance(parsed.get(key), list):
                        return [x for x in parsed[key] if isinstance(x, dict)]
        except Exception as exc:
            logger.warning("Groq extraction failed: %s", exc)
        return None

    @staticmethod
    def _align_and_sanitize(
        chunk: Sequence[RawIngestionItem],
        rows: Sequence[Mapping[str, Any]],
        method: str,
    ) -> List[ExtractedEventDraft]:
        by_id: Dict[str, Mapping[str, Any]] = {}
        for row in rows:
            sid = str(row.get("source_id") or "").strip()
            if sid and sid not in by_id:
                by_id[sid] = row

        drafts: List[ExtractedEventDraft] = []
        for idx, item in enumerate(chunk):
            matched = by_id.get(item.source_id)
            if matched is None and idx < len(rows) and len(rows) == len(chunk):
                matched = rows[idx]
            if matched is not None:
                drafts.append(
                    sanitize_and_enrich_draft(
                        matched,
                        item=item,
                        method="gemini" if method == "gemini" else "groq",
                    )
                )
            else:
                drafts.append(heuristic_extract_item(item))
        return drafts
