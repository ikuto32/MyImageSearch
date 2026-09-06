"""Lossless saved search queries, including the settings needed for replay."""

from dataclasses import dataclass
import json
import math
import re

from app.domain.domain_object import ModelId


class InvalidSearchQuery(ValueError):
    """A query supplied by the caller cannot be used for this search."""


FLOAT32_MAX = 3.4028234663852886e38
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _vector(values) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise InvalidSearchQuery("search_query must contain a non-empty numeric vector")
    result = []
    for value in values:
        if type(value) not in (int, float):
            raise InvalidSearchQuery("search_query vector must contain finite float32 numbers")
        try:
            valid = math.isfinite(value) and abs(value) <= FLOAT32_MAX
        except OverflowError:
            valid = False
        if not valid:
            raise InvalidSearchQuery("search_query vector must contain finite float32 numbers")
        result.append(float(value))
    return tuple(result)


@dataclass(frozen=True)
class SearchQuery:
    vector: tuple[float, ...]
    mean_centering: bool = True
    model_id: ModelId | None = None

    def to_text(self) -> str:
        if self.model_id is None:
            # Existing vector-only queries retain their historical interpretation.
            return json.dumps(self.vector, allow_nan=False)
        return json.dumps({
            "version": 1,
            "model": {
                "model_name": self.model_id.model_name,
                "pretrained": self.model_id.pretrained,
            },
            "mean_centering": self.mean_centering,
            "vector": self.vector,
        }, allow_nan=False, separators=(",", ":"))


def parse_search_query(text: str, expected_model: ModelId | None = None) -> SearchQuery:
    if not isinstance(text, str) or not text.strip():
        raise InvalidSearchQuery("search_query must be a non-empty string")
    text = text.strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            raise InvalidSearchQuery("search_query must contain valid saved query JSON") from None
        if type(data.get("version")) is not int or data["version"] != 1:
            raise InvalidSearchQuery("search_query version is unsupported; expected version 1")
        if type(data.get("mean_centering")) is not bool:
            raise InvalidSearchQuery("search_query mean_centering must be true or false")
        model = data.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("model_name"), str) or not model["model_name"].strip() or not isinstance(model.get("pretrained"), str):
            raise InvalidSearchQuery("search_query model must specify model_name and pretrained")
        model_id = ModelId(model["model_name"], model["pretrained"])
        if expected_model is not None and model_id != expected_model:
            raise InvalidSearchQuery(
                "search_query was saved for a different model; select "
                f"{model_id.model_name} ({model_id.pretrained or 'no pretrained tag'})"
            )
        return SearchQuery(_vector(data.get("vector")), data["mean_centering"], model_id)

    # NumPy's previous output uses numeric literals such as "1." and "-.1".
    # Read only numeric tokens, never evaluate Python expressions.
    for _ in range(2):
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
    values = []
    for token in text.split(","):
        if not _NUMBER.fullmatch(token.strip()):
            raise InvalidSearchQuery("search_query must be a non-empty numeric vector")
        values.append(float(token))
    return SearchQuery(_vector(values))
