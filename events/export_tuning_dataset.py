"""Export labeled `(post -> extracted JSON)` training pairs for Google Colab Gemini Fine-Tuning.

Usage:
    py -3.13 -m events.export_tuning_dataset
    py -3.13 -m events.export_tuning_dataset --db rss_bot.db.bak --out-dir events/tuning_data
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BASE_DIR, DB_PATH
from events.extractor import SYSTEM_REGULATIONS, USER_POST_TEMPLATE

SCHEMA_KEYS = [
    "is_event",
    "reason",
    "title",
    "description",
    "event_start_datetime",
    "event_end_datetime",
    "reg_start_datetime",
    "reg_end_datetime",
    "location",
    "organizer",
    "registration_url",
    "fee",
    "is_online",
    "target_audience",
    "category",
    "format",
]


def _normalize_target_json(is_event: bool, raw_json_str: str) -> Dict[str, Any]:
    """Normalize stored `PostAnalysis.event_json` to match `GEMINI_RESPONSE_SCHEMA` keys cleanly."""
    try:
        parsed = json.loads(raw_json_str or "{}")
        if not isinstance(parsed, dict):
            parsed = {}
    except Exception:
        parsed = {}

    def _str_or_empty(val: Any) -> str:
        return str(val).strip() if val is not None else ""

    def _list_of_str(val: Any) -> List[str]:
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        return []

    is_ev = bool(parsed.get("is_event", is_event))
    reason = _str_or_empty(parsed.get("reason")) or (
        "Bài đăng thông báo sự kiện hoặc hoạt động sinh viên sắp/đang diễn ra."
        if is_ev
        else "Bài đăng không thông báo sự kiện hay hoạt động mở đăng ký sắp tới."
    )

    reg_url = _str_or_empty(parsed.get("registration_url") or parsed.get("registration"))
    ev_start = _str_or_empty(parsed.get("event_start_datetime") or parsed.get("start_datetime"))
    ev_end = _str_or_empty(parsed.get("event_end_datetime") or parsed.get("end_datetime"))

    return {
        "is_event": is_ev,
        "reason": reason,
        "title": _str_or_empty(parsed.get("title")) if is_ev else "",
        "description": _str_or_empty(parsed.get("description")) if is_ev else "",
        "event_start_datetime": ev_start if is_ev else "",
        "event_end_datetime": ev_end if is_ev else "",
        "reg_start_datetime": _str_or_empty(parsed.get("reg_start_datetime")) if is_ev else "",
        "reg_end_datetime": _str_or_empty(parsed.get("reg_end_datetime")) if is_ev else "",
        "location": _str_or_empty(parsed.get("location")) if is_ev else "",
        "organizer": _str_or_empty(parsed.get("organizer")) if is_ev else "",
        "registration_url": reg_url if is_ev else "",
        "fee": _str_or_empty(parsed.get("fee")) if is_ev else "",
        "is_online": bool(parsed.get("is_online", False)),
        "target_audience": _list_of_str(parsed.get("target_audience")) if is_ev else [],
        "category": _list_of_str(parsed.get("category")) if is_ev else [],
        "format": _list_of_str(parsed.get("format")) if is_ev else [],
    }


def _resolve_sqlite_path(explicit_db: Optional[str] = None) -> Path:
    if explicit_db:
        p = Path(explicit_db)
        return p if p.is_absolute() else (BASE_DIR / p)

    # Use active rss_bot.db if it has analyzed posts; otherwise fall back to rss_bot.db.bak
    if DB_PATH.is_file():
        try:
            with sqlite3.connect(DB_PATH) as con:
                cnt = con.execute("SELECT COUNT(*) FROM post_analyses").fetchone()[0]
                if cnt > 0:
                    return DB_PATH
        except Exception:
            pass

    bak = BASE_DIR / "rss_bot.db.bak"
    if bak.is_file():
        return bak
    return DB_PATH


def export_gemini_tuning_files(
    *,
    db_path: Optional[str] = None,
    out_dir: str = "events/tuning_data",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Export Gemini SFT `.jsonl` (Vertex AI) and `.json` (`tunedModels`) datasets from SQLite."""
    src_db = _resolve_sqlite_path(db_path)
    if not src_db.is_file():
        raise FileNotFoundError(f"SQLite database not found: {src_db}")

    target_dir = Path(out_dir) if Path(out_dir).is_absolute() else (BASE_DIR / out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    query = """
        SELECT
            p.post_id,
            p.content,
            p.published_at,
            COALESCE(f.name, f.slug, 'Unknown Source') AS source_name,
            pa.is_event,
            pa.event_json
        FROM post_analyses pa
        JOIN posts p ON p.post_id = pa.post_id
        LEFT JOIN fanpages f ON f.id = p.fanpage_id
        WHERE LENGTH(TRIM(COALESCE(p.content, ''))) >= 20
        ORDER BY p.published_at DESC
    """

    sft_rows: List[Dict[str, Any]] = []
    studio_rows: List[Dict[str, str]] = []

    with sqlite3.connect(src_db) as con:
        for _pid, content, pub_at, source_name, is_ev, ev_json in con.execute(query):
            user_prompt = USER_POST_TEMPLATE.format(
                published_at=str(pub_at or "unknown"),
                fanpage=str(source_name or "Facebook"),
                content=str(content or "")[:6000],
            )
            label_dict = _normalize_target_json(bool(is_ev), ev_json or "{}")
            label_str = json.dumps(label_dict, ensure_ascii=False)

            sft_rows.append(
                {
                    "systemInstruction": {
                        "role": "system",
                        "parts": [{"text": SYSTEM_REGULATIONS}],
                    },
                    "contents": [
                        {"role": "user", "parts": [{"text": user_prompt}]},
                        {"role": "model", "parts": [{"text": label_str}]},
                    ],
                }
            )
            studio_rows.append(
                {
                    "text_input": f"{SYSTEM_REGULATIONS}\n\n{user_prompt}",
                    "output": label_str,
                }
            )

    rng = random.Random(seed)
    indices = list(range(len(sft_rows)))
    rng.shuffle(indices)

    val_count = int(round(len(sft_rows) * val_ratio)) if len(sft_rows) >= 10 else 0
    val_set = set(indices[:val_count])

    train_sft = [row for idx, row in enumerate(sft_rows) if idx not in val_set]
    val_sft = [row for idx, row in enumerate(sft_rows) if idx in val_set]

    train_jsonl_path = target_dir / "gemini_sft_train.jsonl"
    val_jsonl_path = target_dir / "gemini_sft_val.jsonl"
    studio_json_path = target_dir / "gemini_tuned_models.json"

    train_jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_sft) + ("\n" if train_sft else ""),
        encoding="utf-8",
    )
    val_jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_sft) + ("\n" if val_sft else ""),
        encoding="utf-8",
    )
    studio_json_path.write_text(
        json.dumps(studio_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "source_db": str(src_db),
        "total_examples": len(sft_rows),
        "train_examples": len(train_sft),
        "val_examples": len(val_sft),
        "train_jsonl": str(train_jsonl_path),
        "val_jsonl": str(val_jsonl_path),
        "studio_json": str(studio_json_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Gemini fine-tuning datasets for Google Colab.")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (defaults to rss_bot.db or rss_bot.db.bak).")
    parser.add_argument("--out-dir", default="events/tuning_data", help="Output folder for exported JSONL/JSON.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split fraction (default: 0.15).")
    args = parser.parse_args()

    summary = export_gemini_tuning_files(db_path=args.db, out_dir=args.out_dir, val_ratio=args.val_ratio)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
