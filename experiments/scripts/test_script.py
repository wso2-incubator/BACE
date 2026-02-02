class Solution:
    def sol(self, input_str: str) -> str:
        S = input_str.strip()
        n = len(S)

        if n == 0:
            return ""

        rev_S = S[::-1]
        T = rev_S + "#" + S

        # Compute KMP prefix function
        m = len(T)
        pi = [0] * m
        for i in range(1, m):
            j = pi[i - 1]
            while j > 0 and T[i] != T[j]:
                j = pi[j - 1]
            if T[i] == T[j]:
                j += 1
            pi[i] = j

        # L = length of longest palindromic suffix of S
        L = pi[m - 1]

        # Append reverse of the prefix that's not part of palindromic suffix
        return S + rev_S[L:]
