"""Fallback attribute extraction helpers for Module 6 (refactored from res/keyword_rules.py).

Fixes known bugs in the legacy draft script:
- Maps multi-day durations (2..6 days) cleanly to rubric.yaml's canonical '24h_to_lt_1w' bucket
  instead of emitting incompatible 'ambiguous_lt_24h_or_mid' / 'ge_1w?' strings.
- Resolves res/keyword_rules.yaml via repo-root Path instead of CWD-relative path.
"""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from configs.model_config import RES_DIR

DEFAULT_KEYWORD_RULES_YAML = RES_DIR / "keyword_rules.yaml"


def normalize_vi(text: str) -> str:
    """Lowercase, strip Vietnamese diacritics (đ -> d), and collapse punctuation/whitespace."""
    s = unicodedata.normalize("NFD", (text or "").lower().replace("đ", "d"))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _pad(text: str) -> str:
    return f" {text} "


def load_yaml_safe(path: Path) -> Dict[str, Any]:
    """Load YAML using PyYAML if installed, falling back to Ruby stdlib YAML on macOS."""
    try:
        import yaml  # type: ignore

        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level mapping in {path}")
        return data
    except ImportError:
        proc = subprocess.run(
            [
                "ruby",
                "-ryaml",
                "-rjson",
                "-e",
                f"puts JSON.generate(YAML.load_file({str(path)!r}))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(proc.stdout)


class AttributeExtractor:
    """Extracts level, role, duration bucket, and explicit DRL points from event text."""

    LEVEL_ORDER = ("quoc_te", "quoc_gia", "thanh_pho", "dhqg", "truong", "khoa", "lop")

    def __init__(self, rules_yaml_path: Path = DEFAULT_KEYWORD_RULES_YAML) -> None:
        cfg = load_yaml_safe(rules_yaml_path)
        self._attr = cfg["attributes"]

    @staticmethod
    def _contains(padded_text: str, phrase: str) -> bool:
        return _pad(normalize_vi(phrase)) in padded_text

    def extract_level(self, text: str, organizer: str = "") -> Tuple[Optional[str], List[str]]:
        """Detect highest organization scope mentioned in text + organizer."""
        full = _pad(normalize_vi(text) + " " + normalize_vi(organizer))
        found = [
            lv
            for lv in self.LEVEL_ORDER
            if any(self._contains(full, p) for p in self._attr["level"][lv])
        ]
        return (found[0] if found else None), found

    def extract_role(self, text: str) -> Tuple[List[str], bool]:
        """Detect mentioned roles (organizer/collaborator/volunteer) and recruitment cues."""
        full = _pad(normalize_vi(text))
        roles = [
            role
            for role, phrases in self._attr["role"].items()
            if any(self._contains(full, p) for p in phrases)
        ]
        is_recruiting = any(
            self._contains(full, cue) for cue in self._attr["recruitment_cues"]
        )
        return roles, bool(roles and is_recruiting)

    def extract_duration_bucket(self, text: str) -> Tuple[Optional[str], bool]:
        """Extract canonical rubric.yaml duration bucket ('lt_24h', '24h_to_lt_1w', 'ge_1w').

        Returns (bucket_or_none, is_inferred_prior).
        """
        t = normalize_vi(text)
        pattern = self._attr["duration"]["unit_patterns"][0]
        for match in re.finditer(pattern, t):
            n = int(match.group(1))
            unit = match.group(2)
            if unit in ("tuan", "thang"):
                return "ge_1w", False
            if unit == "ngay":
                if n >= 7:
                    return "ge_1w", False
                if n >= 2:
                    return "24h_to_lt_1w", False
                return "lt_24h", False
            if unit == "gio":
                if n >= 24:
                    return "24h_to_lt_1w", False
                return "lt_24h", False

        for prior in self._attr["duration"]["priors"]:
            if self._contains(_pad(t), prior["phrase"]):
                bucket = str(prior["result"]).rstrip("?")
                if bucket in ("lt_24h", "24h_to_lt_1w", "ge_1w"):
                    return bucket, True
        return None, False

    def extract_explicit_points(self, text: str) -> Dict[str, Any]:
        """Detect explicitly announced DRL points or explicit non-DRL negation."""
        t = normalize_vi(text)
        ep = self._attr["explicit_points"]
        if re.search(ep["phu_dinh"], t):
            return {"mentions_drl": True, "negated": True, "points": None}
        match = re.search(ep["cong_diem"], t) or re.search(ep["so_diem"], t)
        mention = any(self._contains(_pad(t), p) for p in ep["nhac_den_drl"])
        return {
            "mentions_drl": bool(mention or match),
            "negated": False,
            "points": int(match.group(1)) if match else None,
        }
