"""`[EXT]` Event Extraction Module (`src/extraction`).

Clean-slate modular implementation aligned with `SYSTEM_ARCHITECTURE.md` and `drl_database.db`:
- Stage 1: Deterministic noise/lifecycle pre-filter (`prefilter.py`)
- Stage 2: Direct REST Gemini `responseSchema` + Groq + offline heuristic fallback (`llm_client.py`, `heuristics.py`)
- Stage 3: Local OpenCV QR poster decoding (`media_qr.py`) & HCMC/VNU-HCM geocoding (`geocoder.py`)
- Stage 4: Anchor deduplication, field updates, soft-cancellation (`is_cancelled = 1`), and subscriber notifications (`deduplicator.py`)
"""

from .contracts import (
    ALLOWED_FORMATS,
    ALLOWED_TOPICS,
    BatchExtractionSummary,
    EventClassification,
    ExtractedEventDraft,
    ExtractionMethod,
    ItemExtractionResult,
    PersistenceAction,
    PreFilterDecision,
    RawIngestionItem,
    SourceChannel,
)
from .deduplicator import (
    compute_title_similarity,
    find_matching_event,
    format_provenance_source,
    merge_sources_strings,
    persist_extracted_draft,
)
from .geocoder import EventGeocoder, clean_address_candidates
from .heuristics import (
    build_fallback_description,
    clean_registration_url,
    clean_summary_description,
    find_registration_url_in_text,
    heuristic_extract_item,
    sanitize_and_enrich_draft,
)
from .llm_client import DirectLLMExtractor, GeminiKeyPool
from .media_qr import resolve_registration_url_from_posters
from .pipeline import (
    ExtractionPipeline,
    extract_email,
    extract_items,
    extract_scraped_posts,
    extract_single_facebook_url,
    get_extraction_pipeline,
)
from .prefilter import normalize_vi_ascii, reason_about_post
from .prompts import DESCRIPTION_SUMMARIZATION_GUIDELINES, SYSTEM_INSTRUCTION

__all__ = [
    "ALLOWED_FORMATS",
    "ALLOWED_TOPICS",
    "BatchExtractionSummary",
    "DESCRIPTION_SUMMARIZATION_GUIDELINES",
    "DirectLLMExtractor",
    "EventClassification",
    "EventGeocoder",
    "ExtractedEventDraft",
    "ExtractionMethod",
    "ExtractionPipeline",
    "GeminiKeyPool",
    "ItemExtractionResult",
    "PersistenceAction",
    "PreFilterDecision",
    "RawIngestionItem",
    "SYSTEM_INSTRUCTION",
    "SourceChannel",
    "build_fallback_description",
    "clean_address_candidates",
    "clean_registration_url",
    "clean_summary_description",
    "compute_title_similarity",
    "extract_email",
    "extract_items",
    "extract_scraped_posts",
    "extract_single_facebook_url",
    "find_matching_event",
    "find_registration_url_in_text",
    "format_provenance_source",
    "get_extraction_pipeline",
    "heuristic_extract_item",
    "merge_sources_strings",
    "normalize_vi_ascii",
    "persist_extracted_draft",
    "reason_about_post",
    "resolve_registration_url_from_posters",
    "sanitize_and_enrich_draft",
]
