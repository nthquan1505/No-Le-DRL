"""MVC View layer: HTML template loader and response helpers."""
from pathlib import Path

from fastapi.responses import HTMLResponse

VIEW_DIR = Path(__file__).resolve().parent


def load_view_html(template_name: str) -> str:
    """Read an HTML template file from `src/view/`."""
    target = (VIEW_DIR / template_name).resolve()
    if VIEW_DIR not in target.parents or not target.is_file():
        raise FileNotFoundError(f"View template '{template_name}' not found in {VIEW_DIR}.")
    return target.read_text(encoding="utf-8")


def render_view(template_name: str, status_code: int = 200) -> HTMLResponse:
    """Render an HTML template file from `src/view/` as a FastAPI `HTMLResponse`."""
    return HTMLResponse(content=load_view_html(template_name), status_code=status_code)


__all__ = ["VIEW_DIR", "load_view_html", "render_view"]
