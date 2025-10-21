import unittest


# Programmer Code
class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        def greedy_partitions(t: str) -> int:
            cnt = {}
            parts = 0
            for ch in t:
                cnt[ch] = cnt.get(ch, 0) + 1
                if len(cnt) > k:
                    # start new partition at ch
                    parts += 1
                    cnt = {ch: 1}
            return parts + 1  # last partition
        # no-change baseline
        best = greedy_partitions(s)
        # try one change at every position
        for i in range(n):
            orig = s[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == orig:
                    continue
                t = s[:i] + c + s[i+1:]
                best = max(best, greedy_partitions(t))
        return best

# Tester Code
class TestMaxPartitionsAfterOperations(unittest.TestCase):
    def setUp(self):
        from solution import Solution  # adjust import path as needed
        self.sol = Solution()

    def test_single_char_k1(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("a", 1), 1)

    def test_two_same_chars_k1(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("aa", 1), 1)

    def test_two_diff_chars_k1(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("ab", 1), 2)

    def test_three_same_chars_k1(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("aaa", 1), 1)

    def test_alternating_two_chars_k1(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("ababab", 1), 6)

    def test_alternating_two_chars_k2(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("ababab", 2), 1)

    def test_three_distinct_k2(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("abc", 2), 2)

    def test_three_distinct_k3(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("abc", 3), 1)

    def test_all_distinct_k1(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("abcdef", 1), 6)

    def test_all_distinct_k2(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("abcdef", 2), 3)

    def test_all_distinct_k3(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("abcdef", 3), 2)

    def test_all_same_long_k26(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("zzzzzz", 26), 1)

    def test_mix_with_repeats_k2(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("aabacada", 2), 5)

    def test_mix_with_repeats_k3(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("aabacada", 3), 3)

    def test_edge_k_equals_26_distinctish(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("abcxyz", 26), 1)

    def test_change_can_reduce_prefix_k2(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("aaabbb", 2), 2)

    def test_change_not_needed_large_k(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("abacabadabacaba", 5), 1)

    def test_change_best_in_middle_k1(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("baaaaab", 1), 6)

    def test_long_run_then_switch_k1(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("aaaaab", 1), 2)

    def test_long_run_then_switch_k2(self):
        self.assertEqual(self.sol.maxPartitionsAfterOperations("aaaaab", 2), 1)



if __name__ == "__main__":
    unittest.main(verbosity=2)    unittest.main(verbosity=2)