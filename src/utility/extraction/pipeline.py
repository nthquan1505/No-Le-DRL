"""End-to-end `[EXT]` Extraction Pipeline (`src/utility/extraction/pipeline.py`).

Orchestrates the 4-stage extraction workflow defined in `SYSTEM_ARCHITECTURE.md`:
  Stage 0: Content-hash idempotency guard (skips unchanged seen items)
  Stage 1: Deterministic Vietnamese/English pre-filter (`reason_about_post`)
  Stage 2: Direct REST LLM batch extraction + deterministic fallback (`DirectLLMExtractor`)
  Stage 3: Local poster QR decoding (`resolve_registration_url_from_posters`) & coordinate geocoding (`EventGeocoder`)
  Stage 4: Anchor deduplication, updates, soft-cancellation & subscriber notifications (`persist_extracted_draft`)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

from database import DRLDatabaseManager, get_drl_db

from .contracts import (
    BatchExtractionSummary,
    ExtractedEventDraft,
    ItemExtractionResult,
    RawIngestionItem,
)
from .deduplicator import format_provenance_source, persist_extracted_draft
from .geocoder import EventGeocoder
from .llm_client import DirectLLMExtractor
from .media_qr import resolve_registration_url_from_posters
from .prefilter import reason_about_post

logger = logging.getLogger(__name__)

QrResolverCallable = Callable[[Sequence[str]], Optional[str]]


class ExtractionPipeline:
    """Stateful/configurable `[EXT]` pipeline bound to a `DRLDatabaseManager`."""

    def __init__(
        self,
        db: Optional[DRLDatabaseManager] = None,
        llm_extractor: Optional[DirectLLMExtractor] = None,
        geocoder: Optional[EventGeocoder] = None,
        qr_resolver: Optional[QrResolverCallable] = None,
        enable_qr_scan: bool = True,
        seen_hashes_path: Optional[Path] = None,
    ) -> None:
        self.db = db if db is not None else get_drl_db()
        self.llm_extractor = llm_extractor if llm_extractor is not None else DirectLLMExtractor()
        self.geocoder = geocoder if geocoder is not None else EventGeocoder()
        self.qr_resolver: QrResolverCallable = qr_resolver or resolve_registration_url_from_posters
        self.enable_qr_scan = enable_qr_scan
        self.seen_hashes_path = seen_hashes_path
        self._seen_hashes: Set[str] = self._load_seen_hashes()

    def _load_seen_hashes(self) -> Set[str]:
        if self.seen_hashes_path and self.seen_hashes_path.is_file():
            try:
                data = json.loads(self.seen_hashes_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return {str(x) for x in data if x}
            except Exception:
                pass
        return set()

    def _save_seen_hashes(self) -> None:
        if not self.seen_hashes_path:
            return
        try:
            self.seen_hashes_path.parent.mkdir(parents=True, exist_ok=True)
            # Cap persisted history to most recent 5,000 hashes
            recent = sorted(self._seen_hashes)[-5000:]
            self.seen_hashes_path.write_text(
                json.dumps(recent, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def extract_items(
        self,
        items: Sequence[RawIngestionItem],
        skip_seen: bool = True,
    ) -> BatchExtractionSummary:
        """Process a batch of `RawIngestionItem` objects end-to-end into `drl_database.db`."""
        summary = BatchExtractionSummary(total_items=len(items))
        if not items:
            return summary

        results_by_index: Dict[int, ItemExtractionResult] = {}
        candidate_indices: List[int] = []
        candidate_items: List[RawIngestionItem] = []

        # Stage 0 + Stage 1: Idempotency check & deterministic pre-filter
        for idx, item in enumerate(items):
            chash = item.content_hash
            if skip_seen and chash in self._seen_hashes:
                summary.skipped_seen += 1
                results_by_index[idx] = ItemExtractionResult(
                    source_id=item.source_id,
                    action="skipped_seen",
                    classification="not_event",
                    extraction_method="prefilter",
                    reason="already_processed_content_hash",
                )
                continue

            gate = reason_about_post(item.content, subject=item.subject)
            if not gate.is_candidate:
                summary.prefilter_rejected += 1
                self._seen_hashes.add(chash)
                results_by_index[idx] = ItemExtractionResult(
                    source_id=item.source_id,
                    action="ignored",
                    classification="not_event",
                    extraction_method="prefilter",
                    reason=gate.reason,
                )
                continue

            candidate_indices.append(idx)
            candidate_items.append(item)

        summary.llm_candidates = len(candidate_items)

        # Stage 2: Batch LLM / Heuristic Extraction
        drafts: List[ExtractedEventDraft] = (
            self.llm_extractor.extract_batch(candidate_items) if candidate_items else []
        )

        # Stage 3 + Stage 4: Local QR Decode, Coordinate Geocoding & DB Persistence
        for pos, idx in enumerate(candidate_indices):
            item = candidate_items[pos]
            draft = drafts[pos]
            qr_url: Optional[str] = None
            coord_reason: Optional[str] = None

            try:
                if not draft.is_actionable_event:
                    summary.ignored_not_event += 1
                    self._seen_hashes.add(item.content_hash)
                    results_by_index[idx] = ItemExtractionResult(
                        source_id=item.source_id,
                        action="ignored",
                        classification=draft.classification,
                        extraction_method=draft.extraction_method,
                        reason=draft.reason,
                    )
                    continue

                # Stage 3A: Local QR decoding ONLY when registration_url is None and posters exist
                if self.enable_qr_scan and draft.registration_url is None and draft.images_url:
                    qr_url = self.qr_resolver(draft.images_url)
                    if qr_url:
                        draft.registration_url = qr_url

                # Stage 3B: Campus / Venue Coordinate Resolution
                prov_source = format_provenance_source(item, draft.organizer)
                coord, coord_reason = self.geocoder.resolve(
                    location=draft.location,
                    sources=prov_source,
                    is_online=draft.is_online,
                )
                draft.coordinate = coord

                # Stage 4: Deduplicate / Update / Soft-Cancel in `drl_database.db`
                action, event_id, changed_fields, notif_count = persist_extracted_draft(
                    db=self.db,
                    draft=draft,
                    item=item,
                )
                self._seen_hashes.add(item.content_hash)

                if action == "created":
                    summary.created += 1
                elif action == "updated":
                    summary.updated += 1
                elif action == "cancelled":
                    summary.cancelled += 1
                elif action == "duplicate":
                    summary.duplicates += 1
                else:
                    summary.ignored_not_event += 1

                results_by_index[idx] = ItemExtractionResult(
                    source_id=item.source_id,
                    action=action,
                    classification=draft.classification,
                    extraction_method=draft.extraction_method,
                    reason=draft.reason,
                    event_id=event_id,
                    changed_fields=changed_fields,
                    notifications_created=notif_count,
                    qr_decoded_url=qr_url,
                    coordinate_reason=coord_reason,
                )
            except Exception as exc:
                logger.exception("Failed processing candidate source_id=%s", item.source_id)
                summary.errors += 1
                results_by_index[idx] = ItemExtractionResult(
                    source_id=item.source_id,
                    action="error",
                    classification=draft.classification,
                    extraction_method=draft.extraction_method,
                    reason="pipeline_exception",
                    error=str(exc),
                )

        self._save_seen_hashes()
        summary.results = [results_by_index[i] for i in range(len(items))]
        return summary

    def extract_scraped_posts(
        self,
        posts: Sequence[Mapping[str, Any]],
        fanpage_name: str = "",
        fanpage_id: Optional[int] = None,
        skip_seen: bool = True,
    ) -> BatchExtractionSummary:
        """Convert scraped Facebook post dicts (`scraper/fb_scraper.py`) and run `extract_items`."""
        raw_items = [
            RawIngestionItem.from_scraped_post(p, fanpage_name=fanpage_name, fanpage_id=fanpage_id)
            for p in posts
        ]
        return self.extract_items(raw_items, skip_seen=skip_seen)

    def extract_email(
        self,
        content: str,
        subject: str = "",
        sender: str = "",
        message_id: Optional[str] = None,
        received_at: Optional[Any] = None,
        images_url: Optional[Sequence[str]] = None,
        skip_seen: bool = True,
    ) -> ItemExtractionResult:
        """Extract a single client-submitted email into `drl_database.db`."""
        item = RawIngestionItem.from_email(
            content=content,
            subject=subject,
            sender=sender,
            message_id=message_id,
            received_at=received_at,
            images_url=images_url,
        )
        batch = self.extract_items([item], skip_seen=skip_seen)
        return batch.results[0]

    async def extract_single_facebook_url(self, post_url: str) -> Dict[str, Any]:
        """Scrape a single Facebook post URL asynchronously and run `EXT` in a worker thread.

        If the post is an actionable event and its parent fanpage is not yet in `fanpages`,
        auto-registers the fanpage in `drl_database.db`.
        """
        from scraper.fb_scraper import scrape_single_post

        scraped = await scrape_single_post(post_url)
        if not scraped or not scraped.get("content"):
            return {
                "ok": False,
                "detail": "Could not extract post text from Facebook URL.",
                "post_url": post_url,
            }

        fp_url = str(scraped.get("fanpage_url") or "").strip()
        fp_name = str(scraped.get("fanpage_name") or scraped.get("author_name") or "").strip()

        def _run_sync() -> Dict[str, Any]:
            fanpage_id: Optional[int] = None
            if fp_url:
                existing_fp = self.db.fetch_one(
                    "SELECT fanpage_id FROM fanpages WHERE facebook_url = ?;",
                    (fp_url,),
                    table="fanpages",
                )
                if existing_fp:
                    fanpage_id = int(existing_fp["fanpage_id"])

            item = RawIngestionItem.from_scraped_post(
                scraped,
                fanpage_name=fp_name,
                fanpage_id=fanpage_id,
            )
            res = self.extract_items([item], skip_seen=False).results[0]

            # Auto-register fanpage if this single URL turned out to be a valid new event
            if res.action == "created" and fp_url and fanpage_id is None:
                new_fp_id = self.db.insert_record(
                    "fanpages",
                    {
                        "fanpage_name": fp_name or fp_url,
                        "facebook_url": fp_url,
                        "number_of_events": 1,
                    },
                )
                if new_fp_id is not None:
                    fanpage_id = int(new_fp_id)

            event_row = (
                self.db.fetch_one(
                    "SELECT * FROM events WHERE event_id = ?;",
                    (res.event_id,),
                    table="events",
                )
                if res.event_id is not None
                else None
            )
            return {
                "ok": res.action != "error",
                "action": res.action,
                "classification": res.classification,
                "extraction_method": res.extraction_method,
                "reason": res.reason,
                "event_id": res.event_id,
                "fanpage_id": fanpage_id,
                "event": event_row,
            }

        return await asyncio.to_thread(_run_sync)


# ---------------------------------------------------------------------------
# Module-level singleton convenience functions
# ---------------------------------------------------------------------------
_default_pipeline: Optional[ExtractionPipeline] = None


def get_extraction_pipeline() -> ExtractionPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ExtractionPipeline()
    return _default_pipeline


def extract_items(items: Sequence[RawIngestionItem], skip_seen: bool = True) -> BatchExtractionSummary:
    return get_extraction_pipeline().extract_items(items, skip_seen=skip_seen)


def extract_scraped_posts(
    posts: Sequence[Mapping[str, Any]],
    fanpage_name: str = "",
    fanpage_id: Optional[int] = None,
    skip_seen: bool = True,
) -> BatchExtractionSummary:
    return get_extraction_pipeline().extract_scraped_posts(
        posts=posts,
        fanpage_name=fanpage_name,
        fanpage_id=fanpage_id,
        skip_seen=skip_seen,
    )


def extract_email(
    content: str,
    subject: str = "",
    sender: str = "",
    message_id: Optional[str] = None,
    received_at: Optional[Any] = None,
    images_url: Optional[Sequence[str]] = None,
    skip_seen: bool = True,
) -> ItemExtractionResult:
    return get_extraction_pipeline().extract_email(
        content=content,
        subject=subject,
        sender=sender,
        message_id=message_id,
        received_at=received_at,
        images_url=images_url,
        skip_seen=skip_seen,
    )


async def extract_single_facebook_url(post_url: str) -> Dict[str, Any]:
    return await get_extraction_pipeline().extract_single_facebook_url(post_url)
