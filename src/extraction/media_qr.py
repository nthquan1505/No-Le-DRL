"""Stage-3 local QR-only poster link decoder (`src/extraction/media_qr.py`).

Triggered ONLY when `registration_url` is `None` and `images_url` is non-empty.
Uses local OpenCV (`cv2.QRCodeDetector`) with zero LLM vision calls.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence
from urllib.parse import urlparse
import urllib.request

from .heuristics import NON_REGISTRATION_FB_PATH_RE, clean_registration_url

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB safety limit per poster
DEFAULT_MAX_IMAGES = 3
DEFAULT_DOWNLOAD_TIMEOUT = 6.0

ImageFetcherCallable = Callable[[str, float], Optional[bytes]]
QrBytesDecoderCallable = Callable[[bytes], List[str]]


def _default_image_fetcher(url: str, timeout: float = DEFAULT_DOWNLOAD_TIMEOUT) -> Optional[bytes]:
    if not url or not url.startswith(("http://", "https://")):
        return None
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_len = resp.headers.get("Content-Length")
            if content_len and content_len.isdigit() and int(content_len) > MAX_IMAGE_BYTES:
                return None
            data = resp.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                return None
            return data
    except Exception as exc:
        logger.debug("Poster download skipped for %s: %s", url, exc)
        return None


def _opencv_decode_qr_bytes(image_bytes: bytes) -> List[str]:
    """Decode one or more QR strings from raw image bytes using OpenCV if installed."""
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return []

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    detector = cv2.QRCodeDetector()
    decoded: List[str] = []

    def _collect(mat: object) -> None:
        try:
            ok, texts, _, _ = detector.detectAndDecodeMulti(mat)
            if ok and texts:
                for t in texts:
                    if t and t.strip() and t.strip() not in decoded:
                        decoded.append(t.strip())
                return
        except Exception:
            pass
        try:
            single, _, _ = detector.detectAndDecode(mat)
            if single and single.strip() and single.strip() not in decoded:
                decoded.append(single.strip())
        except Exception:
            pass

    # Pass 1: original color image
    _collect(img)
    if decoded:
        return decoded

    # Pass 2: grayscale + 1.5x upscale for small corner QR codes on student posters
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _collect(gray)
        if not decoded:
            upscaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            _collect(upscaled)
    except Exception:
        pass

    return decoded


def _is_usable_registration_qr_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "facebook.com" in host or "fb.com" in host:
        segments = [seg for seg in (parsed.path or "").split("/") if seg]
        # Reject fanpage profile QR links (/slug), post/photo/reel permalinks, or bare facebook.com
        if not segments or segments[0].lower() != "events" or NON_REGISTRATION_FB_PATH_RE.search(parsed.path or ""):
            return False
    return True


def resolve_registration_url_from_posters(
    images_url: Sequence[str],
    max_images: int = DEFAULT_MAX_IMAGES,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    image_fetcher: Optional[ImageFetcherCallable] = None,
    qr_decoder: Optional[QrBytesDecoderCallable] = None,
) -> Optional[str]:
    """Scan up to `max_images` poster URLs and return the first valid registration URL."""
    if not images_url:
        return None

    fetcher = image_fetcher or _default_image_fetcher
    decoder = qr_decoder or _opencv_decode_qr_bytes

    for img_url in list(images_url)[: max(1, max_images)]:
        raw_bytes = fetcher(img_url, timeout)
        if not raw_bytes:
            continue
        for raw_qr in decoder(raw_bytes):
            cleaned = clean_registration_url(raw_qr)
            if cleaned and _is_usable_registration_qr_url(cleaned):
                return cleaned
    return None
