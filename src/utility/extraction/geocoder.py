"""Stage-3 HCMC & VNU-HCM coordinate resolver (`src/utility/extraction/geocoder.py`).

Resolves `events.coordinate` (`"lat,lng"` or `None`) for `SCH`:
1. Online-only / placeholder venues -> `None`
2. University campuses & faculty halls -> verified campus coordinates
3. Known student landmarks -> verified coordinates
4. External street addresses -> local cache -> Google Maps / Nominatim / ArcGIS
   bounded to Greater HCMC & Dĩ An `(10.35..11.20, 106.30..107.10)`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "geocode_cache.json"

BBOX_LAT = (10.35, 11.20)
BBOX_LNG = (106.30, 107.10)

SCHOOL_CAMPUS_COORDINATES: Tuple[Tuple[str, re.Pattern[str], str], ...] = (
    (
        "Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM - Cơ sở 2 (Linh Trung / Đông Hòa)",
        re.compile(
            r"(?:"
            r"(?:khoa học tự nhiên|hcmus|khtn).*(?:cơ sở\s*0?2|đông\s*h(?:òa|oà)|linh\s*trung|khu\s*đô\s*thị\s*đhqg)"
            r"|(?:cơ sở\s*0?2|cơ sở\s*(?:phường\s*)?đông\s*h(?:òa|oà)).*(?:khoa học tự nhiên|hcmus|khtn)?"
            r"|\bcơ sở\s*(?:phường\s*)?đông\s*h(?:òa|oà)\b|\bcơ sở\s*linh\s*trung\b"
            r")",
            re.IGNORECASE,
        ),
        "10.875809,106.799178",
    ),
    (
        "Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM - Cơ sở 1 (227 Nguyễn Văn Cừ)",
        re.compile(
            r"(?:"
            r"khoa học tự nhiên|\bhcmus\b|\b227\s*nguyễn\s*văn\s*cừ\b"
            r"|\bcơ sở\s*(?:phường\s*)?chợ\s*quán\b|\bcơ sở\s*nguyễn\s*văn\s*cừ\b|\bcơ sở\s*0?1\b.*(?:khtn|hcmus)"
            r")",
            re.IGNORECASE,
        ),
        "10.762622,106.682172",
    ),
    (
        "Trường ĐH Kinh tế - Luật (UEL), ĐHQG-HCM",
        re.compile(r"(?:đại học|đh|trường)\s*kinh\s*tế\s*[\-–]\s*luật|\buel\b", re.IGNORECASE),
        "10.871343,106.778523",
    ),
    (
        "Trường ĐH Công nghệ Thông tin (UIT), ĐHQG-HCM",
        re.compile(r"(?:đại học|đh|trường)\s*công\s*nghệ\s*thông\s*tin|\buit\b", re.IGNORECASE),
        "10.870009,106.803054",
    ),
    (
        "Trường ĐH Bách Khoa (HCMUT), ĐHQG-HCM - Cơ sở 2",
        re.compile(r"bách\s*khoa.*(?:cơ sở\s*0?2|đông\s*h(?:òa|oà)|dĩ\s*an|khu\s*đô\s*thị)", re.IGNORECASE),
        "10.880559,106.805376",
    ),
    (
        "Trường ĐH Bách Khoa (HCMUT), ĐHQG-HCM - Cơ sở 1 (268 Lý Thường Kiệt)",
        re.compile(r"(?:đại học|đh|trường)\s*bách\s*khoa|\bhcmut\b|268\s*lý\s*thường\s*kiệt", re.IGNORECASE),
        "10.772075,106.657902",
    ),
    (
        "Trường ĐH KHXH&NV (USSH), ĐHQG-HCM - Cơ sở 2",
        re.compile(r"(?:khoa học xã hội và nhân văn|khxh\s*&\s*nv|ussh).*(?:cơ sở\s*0?2|linh\s*trung|đông\s*h(?:òa|oà))", re.IGNORECASE),
        "10.872836,106.801865",
    ),
    (
        "Trường ĐH KHXH&NV (USSH), ĐHQG-HCM - Cơ sở 1 (10-12 Đinh Tiên Hoàng)",
        re.compile(r"(?:khoa học xã hội và nhân văn|khxh\s*&\s*nv|\bussh\b|10-12\s*đinh\s*tiên\s*hoàng)", re.IGNORECASE),
        "10.785859,106.702728",
    ),
    (
        "Trường ĐH Quốc Tế (HCMIU), ĐHQG-HCM",
        re.compile(r"(?:đại học|đh|trường)\s*quốc\s*tế|\bhcmiu\b", re.IGNORECASE),
        "10.877611,106.801614",
    ),
    (
        "Trường ĐH Sư phạm TP.HCM (280 An Dương Vương)",
        re.compile(r"(?:đại học|đh|trường)\s*sư\s*phạm\s*(?:tp\.?\s*hcm|thành\s*phố\s*hồ\s*chí\s*minh)|\bhcmue\b", re.IGNORECASE),
        "10.761288,106.682166",
    ),
    (
        "Đại học Kinh tế TP.HCM (UEH - 59C Nguyễn Đình Chiểu)",
        re.compile(r"đại\s*học\s*kinh\s*tế\s*(?:tp\.?\s*hcm|thành\s*phố\s*hồ\s*chí\s*minh)|\bueh\b", re.IGNORECASE),
        "10.782831,106.695031",
    ),
)

KNOWN_VENUE_COORDINATES: Tuple[Tuple[str, re.Pattern[str], str], ...] = (
    (
        "Nhà Văn hóa Sinh viên TP.HCM tại Khu đô thị ĐHQG-HCM",
        re.compile(r"nhà\s*văn\s*h(?:óa|oá)\s*sinh\s*viên.*(?:đhqg|lưu\s*hữu\s*phước|đông\s*h(?:òa|oà))", re.IGNORECASE),
        "10.875120,106.800720",
    ),
    (
        "Nhà Văn hóa Sinh viên TP.HCM (643 Điện Biên Phủ)",
        re.compile(r"nhà\s*văn\s*h(?:óa|oá)\s*sinh\s*viên.*643\s*điện\s*biên\s*phủ", re.IGNORECASE),
        "10.770928,106.678879",
    ),
)

HCMUS_CAMPUS_SHORTHAND_RE = re.compile(
    r"^(?:hội\s*trường\s+i\b|giảng\s*đường\s+\d+|sảnh\s+[a-z0-9]+|phòng\s+[a-z]?\d+|khoa\s+)",
    re.IGNORECASE,
)

ONLINE_ONLY_RE = re.compile(
    r"^(?:online|trực\s*tuyến|zoom|google\s*meet|ms\s*teams|microsoft\s*teams|livestream)(?:\s*\(.*\)|\s+tại\s+fanpage.*)?$",
    re.IGNORECASE,
)
INVALID_LOCATION_RE = re.compile(r"^(?:tba|tbd|đang\s*cập\s*nhật|unknown|chưa\s*xác\s*định)$", re.IGNORECASE)

INDOOR_PREFIX_RE = re.compile(
    r"^(?:phòng\s+[^,-]+|hội\s*trường\s+[^,-]+|giảng\s*đường\s+[^,-]+|sảnh\s+[^,-]+|"
    r"thư\s*viện\s+|nhà\s*hát\s*[^,-]+|tầng\s+[^,-]+|lầu\s+[^,-]+|khuôn\s*viên\s+[^,-]+|hành\s*lang\s+[^,-]+)"
    r"\s*(?:[,–\-]\s*)",
    re.IGNORECASE,
)

WARD_CLEANUP_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r",?\s*phường\s*sài\s*gòn", re.IGNORECASE), ", Quận 1"),
    (re.compile(r",?\s*vĩnh\s*hội", re.IGNORECASE), ", Quận 4"),
    (re.compile(r",?\s*phường\s*tân\s*mỹ", re.IGNORECASE), ", Quận 7"),
    (re.compile(r",?\s*bảy\s*hiền", re.IGNORECASE), ", Tân Bình"),
    (re.compile(r",?\s*phường\s*chợ\s*quán", re.IGNORECASE), ", Quận 5"),
)


def _in_bbox(lat: float, lng: float) -> bool:
    return BBOX_LAT[0] <= lat <= BBOX_LAT[1] and BBOX_LNG[0] <= lng <= BBOX_LNG[1]


def clean_address_candidates(raw_location: str) -> List[str]:
    """Strip indoor room prefixes and produce ordered geocoder query strings."""
    text = re.sub(r"\s+", " ", raw_location or "").strip().rstrip(".")
    text = re.sub(r"\((?:phòng|địa điểm)[^)]*thông báo sau[^)]*\)", "", text, flags=re.IGNORECASE).strip()
    stripped = INDOOR_PREFIX_RE.sub("", text).strip()
    for pat, repl in WARD_CLEANUP_RULES:
        stripped = pat.sub(repl, stripped)

    candidates: List[str] = []
    outside_paren = re.sub(r"\s*\([^)]*\)", "", stripped).strip(" ,.-–")
    if outside_paren:
        candidates.append(f"{outside_paren}, Thành phố Hồ Chí Minh")
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    seen: set[str] = set()
    unique: List[str] = []
    for c in candidates:
        norm = re.sub(r"\s+", " ", c).strip(" ,.")
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


class EventGeocoder:
    """Campus-first coordinate resolver with persistent JSON cache and optional HTTP geocoding."""

    def __init__(
        self,
        cache_path: Optional[Path] = DEFAULT_CACHE_PATH,
        enable_network_geocoding: bool = True,
        google_maps_api_key: Optional[str] = None,
    ) -> None:
        self.cache_path = cache_path
        self.enable_network_geocoding = enable_network_geocoding
        self.google_maps_api_key = (
            google_maps_api_key
            if google_maps_api_key is not None
            else os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
        )
        self._cache: Dict[str, Optional[str]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Optional[str]]:
        if self.cache_path and self.cache_path.is_file():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def resolve(self, location: str, sources: str = "", is_online: bool = False) -> Tuple[Optional[str], str]:
        """Return `(coordinate_str_or_none, diagnostic_reason)`."""
        loc = (location or "").strip()
        if not loc or INVALID_LOCATION_RE.match(loc) or ONLINE_ONLY_RE.match(loc):
            return None, "skipped_online_or_empty"
        if is_online and not re.search(r"\d+|cơ\s*sở|hội\s*trường|phòng|đại\s*học|trường", loc, re.I):
            return None, "skipped_online_mode"

        # 1. Known university campuses
        for label, pat, coord in SCHOOL_CAMPUS_COORDINATES:
            if pat.search(loc):
                return coord, f"school_rule:{label}"

        # 2. Indoor shorthand posted by an HCMUS source
        if HCMUS_CAMPUS_SHORTHAND_RE.search(loc) and re.search(
            r"khoa học tự nhiên|hcmus|khtn|cntt|fit", sources or "", re.IGNORECASE
        ):
            if re.search(r"linh\s*trung|đông\s*h(?:òa|oà)|nhà\s*điều\s*hành|cơ\s*sở\s*2", loc, re.I):
                return "10.875809,106.799178", "school_shorthand:HCMUS_CS2"
            return "10.762622,106.682172", "school_shorthand:HCMUS_CS1"

        # 3. Known student cultural landmarks
        for label, pat, coord in KNOWN_VENUE_COORDINATES:
            if pat.search(loc):
                return coord, f"landmark:{label}"

        # 4. Cache + optional network geocoding
        candidates = clean_address_candidates(loc)
        for q in candidates:
            if q in self._cache and self._cache[q]:
                return self._cache[q], f"cache:{q}"

        if not self.enable_network_geocoding:
            return None, "unresolved_offline"

        dirty = False
        for q in candidates:
            if q in self._cache and self._cache[q] is None:
                continue
            coord = self._query_network_geocoders(q)
            self._cache[q] = coord
            dirty = True
            if coord:
                self._save_cache()
                return coord, f"geocoded:{q}"

        if dirty:
            self._save_cache()
        return None, "unresolved"

    def _query_network_geocoders(self, query: str) -> Optional[str]:
        try:
            if self.google_maps_api_key:
                c = self._geocode_google(query)
                if c:
                    return c
            c = self._geocode_nominatim(query)
            if c:
                return c
            return self._geocode_arcgis(query)
        except Exception:
            return None

    def _geocode_google(self, query: str) -> Optional[str]:
        url = "https://maps.googleapis.com/maps/api/geocode/json?" + urllib.parse.urlencode(
            {"address": query, "region": "vn", "language": "vi", "key": self.google_maps_api_key}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "No-Le-DRL/2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        results = payload.get("results") or []
        if results:
            loc = results[0]["geometry"]["location"]
            lat, lng = float(loc["lat"]), float(loc["lng"])
            if _in_bbox(lat, lng):
                return f"{lat:.6f},{lng:.6f}"
        return None

    @staticmethod
    def _geocode_nominatim(query: str) -> Optional[str]:
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
            {
                "q": query,
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "vn",
                "viewbox": f"{BBOX_LNG[0]},{BBOX_LAT[1]},{BBOX_LNG[1]},{BBOX_LAT[0]}",
                "bounded": 1,
            }
        )
        req = urllib.request.Request(url, headers={"User-Agent": "No-Le-DRL-Extractor/2.0"})
        time.sleep(1.0)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data:
            lat, lng = float(data[0]["lat"]), float(data[0]["lon"])
            if _in_bbox(lat, lng):
                return f"{lat:.6f},{lng:.6f}"
        return None

    @staticmethod
    def _geocode_arcgis(query: str) -> Optional[str]:
        url = (
            "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates?"
            + urllib.parse.urlencode(
                {
                    "SingleLine": query,
                    "f": "json",
                    "maxLocations": 1,
                    "countryCode": "VNM",
                    "searchExtent": f"{BBOX_LNG[0]},{BBOX_LAT[0]},{BBOX_LNG[1]},{BBOX_LAT[1]}",
                }
            )
        )
        req = urllib.request.Request(url, headers={"User-Agent": "No-Le-DRL-Extractor/2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cands = data.get("candidates") or []
        if cands and cands[0].get("score", 0) >= 75:
            loc = cands[0]["location"]
            lat, lng = float(loc["y"]), float(loc["x"])
            if _in_bbox(lat, lng):
                return f"{lat:.6f},{lng:.6f}"
        return None
