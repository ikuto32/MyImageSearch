"""Bounded regular-expression matching shared by the three metadata searches."""
from time import monotonic

import regex


class SearchPatternError(ValueError):
    """Invalid or overly expensive search expression."""


class SearchMatcher:
    MAX_PATTERN_LENGTH = 512
    TIMEOUT_SECONDS = 3.0

    def __init__(self, text: str, is_regexp: bool, *, timeout: float = TIMEOUT_SECONDS):
        self.text = text
        self.pattern = None
        self.deadline = None
        self.timeout = timeout
        if is_regexp:
            if len(text) > self.MAX_PATTERN_LENGTH:
                raise SearchPatternError(f"正規表現は{self.MAX_PATTERN_LENGTH}文字以内で指定してください。")
            try:
                self.pattern = regex.compile(text, regex.VERSION0)
            except (regex.error, OverflowError, RecursionError) as error:
                raise SearchPatternError(f"正規表現を確認してください: {error}") from None

    def __call__(self, value: str | None) -> bool:
        value = value or ""
        if self.pattern is None:
            return self.text in value
        if self.deadline is None:
            self.deadline = monotonic() + self.timeout
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise SearchPatternError("正規表現の処理時間を超えました。式を簡単にして再試行してください。")
        try:
            return self.pattern.search(value, timeout=remaining) is not None
        except TimeoutError:
            raise SearchPatternError("正規表現の処理時間を超えました。式を簡単にして再試行してください。") from None
