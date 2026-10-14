"""Bounded regular-expression matching shared by the metadata searches."""
from collections.abc import Sequence
from time import monotonic

import numpy as np
import regex


class SearchPatternError(ValueError):
    """Invalid or overly expensive search expression."""


# 繰り返しを展開した後の要素数の上限。
MAX_EXPANDED_SIZE = 10_000
# バックスラッシュに続く1文字、または \p{...} のような名前の指定。
_ESCAPED = regex.compile(r"[pPN]\{[^}]*\}|.", regex.DOTALL)
# [ に続く文字クラスの残り。先頭の ] は文字として扱われる。[:alpha:] などの名前はエンジンと同じくASCIIだけ。
_CLASS_REST = regex.compile(
    r"\^?\]?(?:\[:\^?[A-Za-z0-9 &_.-]*(?:[:=][A-Za-z0-9 &_./-]*)?:\]|\\.|[^\]\\])*\]?", regex.DOTALL
)
# ( に続く、空白とコメントを読み飛ばすフラグ x の指定。(?x) や (?ix:...) など。
_VERBOSE = regex.compile(r"\?[aiLmsubefprwV01]*x")
# ( に続く、それ自体は要素にならない指定。コメント (?#...) と、以降のフラグを切り替える (?i) など。
# コメントの中のバックスラッシュは、続く1文字をコメントの一部にする。
_NOT_AN_ELEMENT = regex.compile(
    r"\?(?:#(?:\\.|[^)\\])*|(?:[aiLmsubefprw]|V[01])*(?:-(?:[aiLmsubefprw]|V[01])*)?)\)", regex.DOTALL
)
# { に続く回数の指定。{3} {3,} {3,5} {,5} {,}。数字もカンマもない {} は、エンジンでは文字として扱われる。
_COUNT = regex.compile(r"(?=[0-9]|,)([0-9]*)(,?)[0-9]*\}")


def _check_expansion(text: str) -> None:
    """正規表現エンジンに渡すと、メモリを使い尽くしたりプロセスごと異常終了したりする式を拒否する。

    エンジンは ``x{1000}`` を1000個の複製として、``(...)+`` を2個の複製として組み立てる。入れ子に
    すると複製は掛け算で増え、短い式でも数GBのメモリを使う。数万個を超えるとエンジンの内部で
    スタックがあふれ、例外として捕捉できない。そのため、展開後の要素数を見積もって上限を設ける。

    回数ではない ``{...}`` (あいまい一致の指定)と ``(?x)`` も受け付けない。前者は再帰と組み合わせると
    エンジンが異常終了し、後者は空白やコメントを挟んで繰り返しの対象を見積もれなくする。
    """
    sizes = [0]  # 開いているグループごとの、展開後の要素数
    last = 0  # 直前の要素の大きさ。量指定子はこれを複製する
    position = 0
    while position < len(text):
        char = text[position]
        position += 1
        if char in "+{":
            copies = 2
            if char == "{":
                count = _COUNT.match(text, position)
                if count is None:
                    raise SearchPatternError(
                        "正規表現を確認してください: {} には繰り返しの回数だけを指定できます。"
                        "文字として検索する場合は \\{ と書いてください。"
                    )
                position = count.end()
                copies = max(int(count[1] or 0) + bool(count[2]), 1)
            sizes[-1] += last * (copies - 1)
            last = 0  # 続く + や ? は、繰り返し方を変えるだけで複製しない
        elif char == "(":
            if _VERBOSE.match(text, position):
                raise SearchPatternError("正規表現を確認してください: (?x) の指定には対応していません。")
            skipped = _NOT_AN_ELEMENT.match(text, position)
            if skipped is not None:
                # 続く量指定子は、コメントやフラグの指定を飛び越えて、その前の要素に掛かる。
                position = skipped.end()
            else:
                sizes.append(0)
                last = 0
        else:
            if char == "\\":
                escaped = _ESCAPED.match(text, position)
                position = escaped.end() if escaped else position
            elif char == "[":
                position = _CLASS_REST.match(text, position).end()
            # グループを閉じると、グループ自体と中身の全体が直前の要素になる。空のグループも複製される。
            last = sizes.pop() + 1 if char == ")" and len(sizes) > 1 else 1
            sizes[-1] += last
        if sizes[-1] > MAX_EXPANDED_SIZE:
            raise SearchPatternError("正規表現を確認してください: 繰り返しの回数が大きすぎます。")


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
            _check_expansion(text)
            try:
                self.pattern = regex.compile(text, regex.VERSION0)
            except regex.error as error:
                raise SearchPatternError(f"正規表現を確認してください: {error}") from None
            except (ValueError, KeyError, OverflowError, RecursionError):
                # 式の中のフラグ指定((?V1)や(?au)など)が受け付けられない場合。例外の文面は利用者向けではない。
                raise SearchPatternError("正規表現を確認してください: 対応していない指定が含まれています。") from None
            except MemoryError:
                raise SearchPatternError("正規表現が複雑すぎます。式を簡単にして再試行してください。") from None

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
        except (MemoryError, RuntimeError, regex.error):
            # 終わりのない再帰((?R)など)や、エンジンが実行できない式は、照合の途中で初めて失敗する。
            raise SearchPatternError("正規表現が複雑すぎます。式を簡単にして再試行してください。") from None

    def matching_positions(self, values: Sequence[str]) -> np.ndarray:
        """Positions of every matching value, in order. The deadline covers the whole scan."""
        if self.pattern is None:
            text = self.text
            matched = [position for position, value in enumerate(values) if text in value]
        else:
            matched = [position for position, value in enumerate(values) if self(value)]
        return np.asarray(matched, dtype=np.int64)
