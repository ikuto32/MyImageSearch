import unittest
from time import monotonic
from unittest.mock import patch

from app.application.text_matching import SearchMatcher, SearchPatternError


class TextMatchingTests(unittest.TestCase):
    def test_literal_matching_keeps_regex_characters_literal(self):
        matches = SearchMatcher('[cat]', False)
        self.assertTrue(matches('prefix [cat] suffix'))
        self.assertFalse(matches('cat'))
        self.assertFalse(matches(None))

    def test_regular_expression_supports_japanese_and_groups(self):
        matches = SearchMatcher(r'^(猫|犬)\d+\.png$', True)
        self.assertTrue(matches('猫12.png'))
        self.assertFalse(matches('鳥12.png'))

    def test_invalid_and_overlong_expressions_are_rejected(self):
        for pattern in ('[', 'a' * 513, 'a{9999999999999999999999999}'):
            with self.subTest(pattern=pattern[:30]), self.assertRaises(SearchPatternError):
                SearchMatcher(pattern, True)

    def test_backtracking_is_stopped_by_the_engine(self):
        started = monotonic()
        with self.assertRaises(SearchPatternError):
            SearchMatcher(r'(a|aa)+$', True, timeout=0.02)('a' * 5000 + '!')
        self.assertLess(monotonic() - started, 1)

    def test_deadline_is_shared_across_items(self):
        matches = SearchMatcher('cat', True, timeout=1)
        with patch('app.application.text_matching.monotonic', side_effect=[0, 0.1, 1.1]):
            self.assertTrue(matches('cat'))
            with self.assertRaises(SearchPatternError):
                matches('cat')
