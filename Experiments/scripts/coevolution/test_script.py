class Solution:
    def sol(self, input_str: str) -> str:
        data = list(map(int, input_str.split()))
        it = iter(data)
        try:
            N = next(it)
        except StopIteration:
            return "0"
        A = [next(it) for _ in range(N)]
        B = [next(it) for _ in range(N)]
        C = [next(it) for _ in range(N)]
        mismatches = [i for i in range(N) if A[i] != B[i]]
        m = len(mismatches)
        if m == 0:
            return "0"
        pos = mismatches
        Apos = [A[p] for p in pos]
        Cpos = [C[p] for p in pos]
        init_sum = sum((A[i] * C[i] for i in range(N)))
        delta = [(1 - 2 * Apos[j]) * Cpos[j] for j in range(m)]
        total_states = 1 << m
        sum_delta = [0] * total_states
        for state in range(1, total_states):
            lsb = state & -state
            j = lsb.bit_length() - 1
            prev = state ^ lsb
            sum_delta[state] = sum_delta[prev] + delta[j]
        INF = 10**30
        dp = [INF] * total_states
        dp[0] = 0
        for state in range(total_states):
            if dp[state] == INF:
                continue
            cur_sum = init_sum + sum_delta[state]
            rem = ~state & total_states - 1
            s = rem
            while s:
                lsb = s & -s
                j = lsb.bit_length() - 1
                next_state = state | lsb
                cost_of_op = cur_sum + delta[j]
                new_cost = dp[state] + cost_of_op
                if new_cost < dp[next_state]:
                    dp[next_state] = new_cost
                s ^= lsb
        ans = dp[total_states - 1]
        return str(ans)


# Execute solution
sol = Solution()
print(
    sol.sol("""20
1 1 1 1 0 0 1 1 0 0 0 1 0 1 0 1 1 0 1 0
0 0 0 1 1 1 0 1 1 0 0 0 0 0 0 1 0 1 0 0
52 73 97 72 54 15 79 67 13 55 65 22 36 90 84 46 1 2 27 8""")
)
