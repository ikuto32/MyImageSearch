"""Lossless saved search queries, including the settings needed for replay."""

from dataclasses import dataclass
import json
import math

from app.domain.domain_object import ModelId


class InvalidSearchQuery(ValueError):
    """A query supplied by the caller cannot be used for this search."""


FLOAT32_MAX = 3.4028234663852886e38
QUERY_VERSION = 1


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


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """検索に使った特徴量と、同じ順位を再現するためのモデル・中心化条件。"""

    vector: tuple[float, ...]
    mean_centering: bool
    model_id: ModelId

    def to_text(self) -> str:
        return json.dumps({
            "version": QUERY_VERSION,
            "model": {
                "model_name": self.model_id.model_name,
                "pretrained": self.model_id.pretrained,
            },
            "mean_centering": self.mean_centering,
            "vector": self.vector,
        }, allow_nan=False, separators=(",", ":"))


def parse_search_query(text, expected_model: ModelId | None = None) -> SearchQuery:
    if not isinstance(text, str) or not text.strip():
        raise InvalidSearchQuery("search_query must be a non-empty string")
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        raise InvalidSearchQuery("search_query must contain valid saved query JSON") from None
    if not isinstance(data, dict):
        raise InvalidSearchQuery("search_query must contain valid saved query JSON")
    if type(data.get("version")) is not int or data["version"] != QUERY_VERSION:
        raise InvalidSearchQuery(f"search_query version is unsupported; expected version {QUERY_VERSION}")
    if type(data.get("mean_centering")) is not bool:
        raise InvalidSearchQuery("search_query mean_centering must be true or false")
    model = data.get("model")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("model_name"), str)
        or not model["model_name"].strip()
        or not isinstance(model.get("pretrained"), str)
    ):
        raise InvalidSearchQuery("search_query model must specify model_name and pretrained")
    model_id = ModelId(model["model_name"], model["pretrained"])
    if expected_model is not None and model_id != expected_model:
        raise InvalidSearchQuery(
            "search_query was saved for a different model; select "
            f"{model_id.model_name} ({model_id.pretrained or 'no pretrained tag'})"
        )
    return SearchQuery(_vector(data.get("vector")), data["mean_centering"], model_id)
