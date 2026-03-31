import sys
from pathlib import Path
from datetime import datetime


def safe_print(*args, **kwargs) -> None:
    """
    Print safely when running windowed executables where stdout/stderr may be None.
    """
    try:
        output_stream = kwargs.pop("file", None)
        if output_stream is None:
            output_stream = sys.stdout if sys.stdout is not None else sys.stderr

        if output_stream is None:
            return

        try:
            print(*args, file=output_stream, **kwargs)
        except UnicodeEncodeError:
            # Fallback for Windows charmap consoles that cannot encode emoji/symbols.
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            flush = kwargs.get("flush", False)

            text = sep.join(str(arg) for arg in args) + end
            encoding = getattr(output_stream, "encoding", None) or "utf-8"

            safe_text = text.encode(encoding, errors="replace").decode(
                encoding, errors="replace"
            )
            output_stream.write(safe_text)
            if flush:
                output_stream.flush()
    except Exception:
        # Never let logging/printing break application flow.
        return

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
