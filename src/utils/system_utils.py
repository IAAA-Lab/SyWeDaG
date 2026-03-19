import sys
from pathlib import Path

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
