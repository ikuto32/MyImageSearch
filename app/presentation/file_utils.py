from __future__ import annotations

import pathlib

# Allowed extensions for different file categories
ALLOWED_IMAGE = {
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'
}
ALLOWED_VIDEO = {
    'mp4', 'avi', 'mov', 'mkv', 'webm'
}
ALLOWED_TEXT = {
    'txt', 'md', 'json', 'csv', 'html', 'htm'
}

def file_category(filename: str) -> str | None:
    """Return the category of the file based on its extension.

    Parameters
    ----------
    filename: str
        Name of the file to inspect.

    Returns
    -------
    str | None
        ``"image"`` if the extension is recognised as an image,
        ``"video"`` if it is a video,
        ``"text"`` if it is a text file, otherwise ``None``.
    """
    ext = pathlib.Path(filename).suffix.lower().lstrip('.')
    if ext in ALLOWED_IMAGE:
        return 'image'
    if ext in ALLOWED_VIDEO:
        return 'video'
    if ext in ALLOWED_TEXT:
        return 'text'
    return None
