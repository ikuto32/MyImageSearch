"""画像区分(rating)と、その絞り込み条件。"""

from app.domain.errors import SearchInputError

KNOWN_RATINGS = ("general", "sensitive", "questionable", "explicit")
UNCLASSIFIED = "unclassified"
RATING_CATEGORIES = (*KNOWN_RATINGS, UNCLASSIFIED)

# None は絞り込みなし。空のタプルはどの画像にも一致しない。
RatingFilter = tuple[str, ...] | None


def rating_category(rating: str | None) -> str:
    """未設定や未知のratingを「未分類」にまとめる。"""
    return rating if rating in KNOWN_RATINGS else UNCLASSIFIED


def normalize_rating_filter(ratings) -> RatingFilter:
    """区分の一覧を重複のない整列済みタプルにする。全区分の指定は絞り込みなしと同じ。"""
    if ratings is None:
        return None
    if not isinstance(ratings, (list, tuple)) or any(
        not isinstance(value, str) or value not in RATING_CATEGORIES for value in ratings
    ):
        raise SearchInputError("ratings must contain supported image categories")
    normalized = tuple(sorted(set(ratings)))
    return None if len(normalized) == len(RATING_CATEGORIES) else normalized
