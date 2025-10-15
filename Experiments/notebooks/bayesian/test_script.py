import unittest


# Programmer Code
class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        ch_cnt = [0] * 26
        ch_dp = [[0] * (k + 1) for _ in range(26)]

        for i in range(1, n + 1):
            ch_cnt[ord(s[i - 1]) - ord("a")] += 1
            ch_dp = [
                [
                    max(
                        ch_dp[ch][j],
                        ch_dp[ch][j - 1]
                        + (ch == ord(s[i - 1]) - ord("a") and ch_cnt[ch] == j),
                    )
                    for j in range(k + 1)
                ]
                for ch in range(26)
            ]
            dp[i] = [
                max(dp[i - 1][j], max(ch_dp[ch][j] for ch in range(26)))
                for j in range(k + 1)
            ]

        return dp[-1][-1]


# Tester Code
class TestSolution(unittest.TestCase):
    def test_small_string(self):
        solution = Solution()
        result = solution.maxPartitionsAfterOperations("abc", 2)
        self.assertEqual(result, 2)

    def test_single_char_string(self):
        solution = Solution()
        result = solution.maxPartitionsAfterOperations("a", 1)
        self.assertEqual(result, 1)

    def test_same_char_string(self):
        solution = Solution()
        result = solution.maxPartitionsAfterOperations("aaaa", 1)
        self.assertEqual(result, 4)

    def test_large_k(self):
        solution = Solution()
        result = solution.maxPartitionsAfterOperations("abcd", 4)
        self.assertEqual(result, 1)

    def test_large_string_and_k(self):
        solution = Solution()
        random_string = "a" * 10**4
        result = solution.maxPartitionsAfterOperations(random_string, 26)
        self.assertEqual(result, 10**4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
