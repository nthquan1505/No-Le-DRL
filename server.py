import contextlib
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from credits import CreditCalculationModule, DEFAULT_COMPLETED_EVENTS, DRL_KHTN_RULES

BASE_DIR = Path(__file__).resolve().parent
ENGINE = CreditCalculationModule(DRL_KHTN_RULES)

SCRAPE_FILE = BASE_DIR / "scrape_1311.json"
SCRAPED_CATALOG = [
    {k: v for k, v in item.items() if k != "source_event"}
    for item in (ENGINE.load_and_map_scraped_events(SCRAPE_FILE) if SCRAPE_FILE.exists() else [])
]

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/walkthrough": ("credits_walkthrough.html", "text/html; charset=utf-8"),
    "/drl2025.pdf": ("drl2025.pdf", "application/pdf"),
    "/logo.jpg": ("images (1).jpg", "image/jpeg"),
}


def evaluate_events(events):
    """Chạy trực tiếp CreditCalculationModule trong credits.py cho danh sách sự kiện."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        payload = ENGINE.calculate_earned_points(events)

    mapped_events = []
    for idx, ev in enumerate(events):
        mapped = ENGINE.map_event_to_drl_criteria(ev if isinstance(ev, dict) else {"event_category": ev})
        mapped_events.append({
            "index": idx,
            "title": ev.get("title", "") if isinstance(ev, dict) else str(ev),
            "event_category": mapped.get("matched_key", ""),
            "category_id": mapped["category_id"],
            "sub_code": mapped["sub_code"],
            "article_ref": mapped.get("article_ref", ""),
            "points": ev.get("points", mapped["default_points"]) if isinstance(ev, dict) else mapped["default_points"],
        })

    categories_detail = []
    for cat_id, info in DRL_KHTN_RULES["categories"].items():
        raw_val = payload["raw_by_category"].get(cat_id, 0)
        capped_val, max_cap, is_capped = ENGINE.validate_max_cap_limit(cat_id, raw_val)
        categories_detail.append({
            "cat_id": cat_id,
            "name": info["name"],
            "article_ref": info.get("article_ref", ""),
            "base_points": info.get("base_points", 0),
            "raw_points": raw_val,
            "capped_points": capped_val,
            "max_cap": max_cap,
            "is_capped": is_capped,
        })

    return {
        "payload": payload,
        "mapped_events": mapped_events,
        "categories_detail": categories_detail,
        "console_log": buf.getvalue(),
    }


class MinimalHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[VA-DRL] {fmt % args}")

    def _send(self, status, content_type, data_bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data_bytes)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data_bytes)

    def _send_json(self, status, obj):
        self._send(status, "application/json; charset=utf-8", json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path in STATIC_FILES:
            filename, mime = STATIC_FILES[path]
            file_path = BASE_DIR / filename
            if not file_path.exists():
                return self._send_json(404, {"error": f"Không tìm thấy {filename}"})
            return self._send(200, mime, file_path.read_bytes())

        if path == "/api/init":
            return self._send_json(200, {
                "rules": DRL_KHTN_RULES,
                "default_events": DEFAULT_COMPLETED_EVENTS,
                "result": evaluate_events(DEFAULT_COMPLETED_EVENTS),
                "scraped_events": SCRAPED_CATALOG,
            })

        return self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/calculate":
            return self._send_json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            return self._send_json(200, evaluate_events(body.get("events", [])))
        except Exception as exc:
            return self._send_json(400, {"error": str(exc)})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    req_port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else int(os.environ.get("PORT", "8080"))
    server = None
    for port in (req_port, 8888, 18080, 5000):
        try:
            server = ThreadingHTTPServer((host, port), MinimalHandler)
            print(f"🚀 VA DRL Server: http://127.0.0.1:{port} (host={host}, {len(SCRAPED_CATALOG)} sự kiện)", flush=True)
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError("Không thể mở cổng mạng.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
