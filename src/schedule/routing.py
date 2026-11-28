"""Optional HTTP routing. The schedule engine can run without credentials or network."""

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import ceil
import json
import os
from pathlib import Path
import sqlite3
from typing import Callable, Optional
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


CAMPUSES = {
    "cs1_nvc": ("227 Nguyễn Văn Cừ, TP.HCM, Việt Nam", (106.6825, 10.7628)),
    "cs2_lt": ("Trường Đại học Khoa học Tự nhiên, cơ sở Linh Trung, TP.HCM, Việt Nam", (106.7959, 10.8761)),
}


@dataclass(frozen=True)
class RouteEstimate:
    minutes: int
    source: str


def _key(value):
    return " ".join(value.casefold().split())


def _parse_lat_lng(value: Optional[str]) -> Optional[tuple[float, float]]:
    if not value or not isinstance(value, str) or "," not in value:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        return None
    try:
        lat, lng = float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    return lat, lng


class CarRouting:
    def __init__(
        self, *, google_key: Optional[str] = None,
        quota_db: Path = Path(".module4-route-quota.db"), daily_google_limit: int = 250,
        osrm_url: Optional[str] = None, geocode: Optional[Callable] = None,
        request_json: Optional[Callable] = None,
        catalog: Optional[dict] = None,
    ):
        self.google_key = google_key
        self.quota_db = quota_db
        # 250/day leaves headroom below the 10,000/month Essentials allowance.
        self.daily_google_limit = min(250, max(0, daily_google_limit))
        self.osrm_url = osrm_url
        self.geocode = geocode
        self._json = request_json or self._request_json
        self._places = dict(CAMPUSES)
        self._aliases = {}
        self._times = {}
        for place in (catalog or {}).get("places", ()):
            location_id, address = place["id"], place["address"]
            self._places[location_id] = (address, (float(place["longitude"]), float(place["latitude"])))
            for alias in (location_id, address, *place.get("aliases", ())):
                self._aliases[_key(alias)] = location_id
        for item in (catalog or {}).get("travel_minutes", ()):
            self._times[(item["origin"], item["destination"])] = int(item["minutes"])
        self.cache_signature = sha256(json.dumps({
            "catalog": catalog, "osrm": osrm_url, "geocode_enabled": geocode is not None,
            "google_enabled": bool(google_key),
        }, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_environment(cls):
        # Nominatim is opt-in. Use a private endpoint for personal calendar addresses.
        nominatim = os.getenv("MODULE4_NOMINATIM_URL")
        catalog_path = os.getenv("MODULE4_LOCATION_CATALOG")
        geocode = cls.nominatim_geocoder(nominatim) if nominatim else None
        return cls(
            google_key=os.getenv("MODULE4_GOOGLE_ROUTES_KEY"),
            osrm_url=os.getenv("MODULE4_OSRM_URL"),
            geocode=geocode,
            quota_db=Path(os.getenv("MODULE4_QUOTA_DB", ".module4-route-quota.db")),
            catalog=json.loads(Path(catalog_path).read_text(encoding="utf-8")) if catalog_path else None,
        )

    def _admit_google(self) -> bool:
        if self.daily_google_limit == 0:
            return False
        day = datetime.now(timezone.utc).date().isoformat()
        try:
            with closing(sqlite3.connect(self.quota_db)) as db, db:
                db.execute("CREATE TABLE IF NOT EXISTS daily_calls (day TEXT PRIMARY KEY, calls INTEGER NOT NULL)")
                result = db.execute(
                    "INSERT INTO daily_calls VALUES (?, 1) "
                    "ON CONFLICT(day) DO UPDATE SET calls = calls + 1 WHERE calls < ?",
                    (day, self.daily_google_limit),
                )
                return result.rowcount == 1
        except sqlite3.Error:
            return False

    @staticmethod
    def _request_json(request):
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    @staticmethod
    def nominatim_geocoder(base_url: str):
        from urllib.parse import urlsplit

        if (urlsplit(base_url).hostname or "").casefold() == "nominatim.openstreetmap.org":
            raise ValueError("Use a private Nominatim endpoint for calendar address lookup")

        def resolve(address):
            query = urlencode({"q": address, "format": "jsonv2", "limit": 2, "countrycodes": "vn"})
            request = Request(
                f"{base_url.rstrip('/')}/search?{query}", headers={"User-Agent": "No-Le-DRL-Module4/1.0"}
            )
            matches = CarRouting._request_json(request)
            if len(matches) == 1:
                return float(matches[0]["lon"]), float(matches[0]["lat"])
            return None

        return resolve

    def _coordinates(self, location):
        if location in self._places:
            return self._places[location][1]
        parsed = _parse_lat_lng(location)
        if parsed is not None:
            return parsed[1], parsed[0]
        return self.geocode(location) if self.geocode else None

    def _google_waypoint(self, location: str) -> dict:
        if location in self._places:
            return {"address": self._places[location][0]}
        parsed = _parse_lat_lng(location)
        if parsed is not None:
            return {"location": {"latLng": {"latitude": parsed[0], "longitude": parsed[1]}}}
        return {"address": location}

    def __call__(self, origin: str, destination: str) -> Optional[int]:
        result = self.estimate(origin, destination)
        return result.minutes if result is not None else None

    def estimate(self, origin: str, destination: str) -> Optional[RouteEstimate]:
        origin, destination = (self._aliases.get(_key(item), item) for item in (origin, destination))
        minutes = self._times.get((origin, destination))
        if minutes is not None:
            return RouteEstimate(max(0, minutes), "catalog")

        if self.osrm_url:
            try:
                first, second = self._coordinates(origin), self._coordinates(destination)
                if first and second:
                    points = quote(f"{first[0]},{first[1]};{second[0]},{second[1]}", safe=",;.")
                    request = Request(f"{self.osrm_url.rstrip('/')}/route/v1/driving/{points}?overview=false")
                    return RouteEstimate(max(0, ceil(float(self._json(request)["routes"][0]["duration"]) / 60)), "osrm")
            except (URLError, OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
                pass

        if self.google_key and self._admit_google():
            try:
                body = {
                    "origin": self._google_waypoint(origin),
                    "destination": self._google_waypoint(destination),
                    "travelMode": "DRIVE",
                    "routingPreference": "TRAFFIC_UNAWARE",
                }
                request = Request(
                    "https://routes.googleapis.com/directions/v2:computeRoutes",
                    data=json.dumps(body).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": self.google_key,
                        "X-Goog-FieldMask": "routes.duration",
                    },
                )
                duration = self._json(request)["routes"][0]["duration"]
                return RouteEstimate(max(0, ceil(float(duration.removesuffix("s")) / 60)), "google")
            except (URLError, OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
                pass

        return None
