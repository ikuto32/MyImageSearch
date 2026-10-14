import unittest
from time import monotonic
from unittest.mock import Mock, patch

import numpy as np

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
        # The engine reports unsupported inline flags with errors of its own kinds.
        for pattern in ('[', 'a' * 513, 'a{9999999999999999999999999}', '(?V1)[[a-z]--[aeiou]]', '(?au)cat'):
            with self.subTest(pattern=pattern[:30]), self.assertRaises(SearchPatternError):
                SearchMatcher(pattern, True)

    def test_unbounded_recursion_is_rejected_while_matching(self):
        with self.assertRaises(SearchPatternError):
            SearchMatcher('(?R)', True).matching_positions(['cat'])

    def test_expressions_that_would_expand_beyond_the_limit_never_reach_the_engine(self):
        # Larger versions of these exhaust memory or crash the engine natively. These sizes are harmless
        # to compile, so a missing check shows as a failed assertion instead of a dead test process.
        nested = '(' * 14 + 'a+' + ')+' * 14
        for pattern in (
            '(?:|a){20000}', '(?:ab|cd){12000}', '(?:(?:|a){150}){150}', 'a{10001}', nested,
            # A quantifier reaches over comments and flag changes to the element before them.
            '(?:(?:|a){100})(?#note){100}', '(?:(?:|a){100})(?i){100}', '(?:(?:|a){100})(?#a)(?-i)(?#b){100}',
            # Empty groups are copied too; compiling many copies of one takes minutes.
            '(){20000}',
            # Places where the engine reads braces, set names and comments differently from a naive scan.
            '{}{20000}', '[[:é:](?:ab){12000}]',
            # Approximate matching, and the mode that makes whitespace and comments invisible.
            '(?:cat){e<=1}', 'cat{i,d}', '(?x)c a t', '(?ix: c a t )',
        ):
            with self.subTest(pattern=pattern[:40]), patch('app.application.text_matching.regex.compile') as compile_:
                with self.assertRaises(SearchPatternError):
                    SearchMatcher(pattern, True)
                compile_.assert_not_called()

    def test_ordinary_repeats_and_literal_braces_are_accepted(self):
        for pattern, value in (
            (r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '123e4567-e89b-12d3-a456-426614174000'),
            (r'^(?:[0-9a-f]{2}){16}$', 'ab' * 16), (r'\p{Han}{2,}', '富士山の写真'), (r'(\w+_)+\d+', 'a_b_12'),
            (r'^.{0,10000}$', 'cat'), (r'cat\{1000000\}', 'cat{1000000}'), (r'[{}]+', '{}'), (r'(?i)CAT(?#note)', 'cat'),
            (r'(?s)a.b(?-s)', 'a\nb'), (r'^.{100}$', 'x' * 100), ('(' * 10 + 'a+' + ')+' * 10, 'aaa'),
            (r'[[:alpha:]]{3}', 'cat'), (r'[[:^digit:]]+', 'cat'), (r'a{,3}b', 'aab'), (r'(?#a\)b)cat', 'cat'),
        ):
            with self.subTest(pattern=pattern[:40]):
                self.assertTrue(SearchMatcher(pattern, True)(value))

    def test_engine_failures_are_reported_as_pattern_errors(self):
        with patch('app.application.text_matching.regex.compile', side_effect=MemoryError):
            with self.assertRaises(SearchPatternError):
                SearchMatcher('cat', True)
        matcher = SearchMatcher('cat', True)
        matcher.pattern = Mock(search=Mock(side_effect=RuntimeError('invalid RE code')))
        with self.assertRaises(SearchPatternError):
            matcher('cat')

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

    def test_matching_positions_lists_every_match_in_order(self):
        values = ['cat.png', 'dog.png', '', 'black [cat].jpg']
        self.assertEqual(SearchMatcher('cat', False).matching_positions(values).tolist(), [0, 3])
        self.assertEqual(SearchMatcher('[cat]', False).matching_positions(values).tolist(), [3])
        self.assertEqual(SearchMatcher(r'^\w+\.png$', True).matching_positions(values).tolist(), [0, 1])
        empty = SearchMatcher('bird', False).matching_positions(values)
        self.assertEqual((empty.tolist(), empty.dtype), ([], np.int64))

    def test_position_scan_shares_one_deadline(self):
        matcher = SearchMatcher('cat', True, timeout=1)
        with patch('app.application.text_matching.monotonic', side_effect=[0, 0.1, 1.1]):
            with self.assertRaises(SearchPatternError):
                matcher.matching_positions(['cat', 'cat'])
