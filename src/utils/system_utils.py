import sys
from pathlib import Path
from datetime import datetime

def get_resource_path(relative_path: str | Path) -> Path:
    """
    Get absolute path to a project resource, compatible with dev and PyInstaller.

    Args:
        relative_path: Resource path relative to the project root.

    Returns:
        Absolute path to the resource.
    """
    base_path = getattr(sys, "_MEIPASS", None)
    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent.parent

    return Path(base_path) / relative_path


def get_downloads_path() -> Path:
    """Return the user's Downloads folder path."""
    return Path.home() / "Downloads"


def save_bytes_to_downloads(file_name: str, content: bytes) -> Path:
    """Save bytes content into Downloads and return final path."""
    downloads_path = get_downloads_path()
    downloads_path.mkdir(parents=True, exist_ok=True)

    target_path = downloads_path / file_name
    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = downloads_path / f"{stem}_{timestamp}{suffix}"

    target_path.write_bytes(content)
    return target_path
