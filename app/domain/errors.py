"""User-facing failures with explicit handling at the HTTP boundary."""


class ResourceLimitError(Exception):
    """The requested image or archive exceeds the configured resource budget."""


class SearchInputError(ValueError):
    """The selected model or an input image cannot be used for search."""


class ImageUnavailableError(Exception):
    """The image is unknown, outside the image root, or cannot be read."""
