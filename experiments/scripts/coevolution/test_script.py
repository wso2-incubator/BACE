import random


def generate_test_inputs(num_inputs: int) -> list[dict]:
    """
    Generate test inputs for the teeth-fitting problem that aim to differentiate between two solution variants.
    Each test is returned as a dict with key "input_str" containing the full stdin string.

    Strategy to produce diverse and challenging cases (not hardcoded):
    - Randomly choose N across small and larger values within constraints.
    - Choose X from small (1) to very large (1e9).
    - Generate U_i and D_i with several patterns:
        * Uniform S = U_i + D_i for all i (already fits) and slight perturbations.
        * Large values near upper bound to test 64-bit sums.
        * Alternating large/small to stress adjacency constraints.
        * Monotonic sequences to force propagation of +/-X constraints.
        * Randomized with occasional forced impossible H for some S values.
    - No test is hardcoded; all are generated via pseudorandom choices.
    """
    tests = []
    rng = random.Random(1234567)

    def make_input(N, X, U, D):
        parts = [f"{N} {X}"]
        for u, d in zip(U, D):
            parts.append(f"{u} {d}")
        return {"input_str": "\n".join(parts) + "\n"}

    for t in range(num_inputs):
        r = rng.random()
        if r < 0.5:
            N = rng.randint(2, 6)
        elif r < 0.85:
            N = rng.randint(7, 50)
        else:
            N = rng.randint(100, 300)
        X_choice = rng.choice([1, rng.randint(1, 10), rng.randint(10, 1000), 10**9])
        X = X_choice
        U = [0] * N
        D = [0] * N
        pattern = rng.choice(
            [
                "uniform_S",
                "all_large",
                "alternating",
                "monotonic",
                "random",
                "edge_case_small_D_or_U",
                "tight_X_constraints",
            ]
        )
        if pattern == "uniform_S":
            S_val = (
                rng.randint(1, 10**6)
                if rng.random() < 0.7
                else rng.randint(10**6, 10**9)
            )
            for i in range(N):
                u = rng.randint(1, S_val - 1) if S_val > 1 else 1
                U[i] = u
                D[i] = S_val - u
            for _ in range(max(1, N // 4)):
                idx = rng.randrange(N)
                delta = rng.randint(0, min(10, 10**5))
                U[idx] = min(10**9, U[idx] + delta)
                D[idx] = max(1, D[idx] - delta)
        elif pattern == "all_large":
            for i in range(N):
                u = rng.randint(10**8, 10**9)
                d = rng.randint(10**8, 10**9)
                U[i] = u
                D[i] = d
            if rng.random() < 0.5:
                idx = rng.randrange(N)
                U[idx] = rng.randint(1, 10)
                D[idx] = rng.randint(1, 10)
        elif pattern == "alternating":
            big = rng.randint(10**6, 10**9)
            small = rng.randint(1, 100)
            for i in range(N):
                if i % 2 == 0:
                    U[i] = big
                    D[i] = rng.randint(1, 1000)
                else:
                    U[i] = rng.randint(1, 1000)
                    D[i] = big
        elif pattern == "monotonic":
            start = rng.randint(1, 1000)
            up = rng.choice([True, False])
            step = rng.randint(0, 1000)
            for i in range(N):
                if up:
                    val = start + i * step + rng.randint(0, 10)
                else:
                    val = max(1, start + (N - i - 1) * step - rng.randint(0, 10))
                U[i] = min(10**9, val)
                D[i] = rng.randint(1, 10**6)
        elif pattern == "random":
            for i in range(N):
                U[i] = rng.randint(1, 10**9)
                D[i] = rng.randint(1, 10**9)
        elif pattern == "edge_case_small_D_or_U":
            for i in range(N):
                if rng.random() < 0.6:
                    U[i] = 1
                    D[i] = rng.randint(1, 10**6)
                else:
                    U[i] = rng.randint(1, 10**6)
                    D[i] = 1
            for _ in range(max(1, N // 5)):
                idx = rng.randrange(N)
                U[idx] = rng.randint(1, 10**9)
                D[idx] = rng.randint(1, 10**9)
        elif pattern == "tight_X_constraints":
            if X == 10**9:
                X = rng.randint(1, 5)
            base = rng.randint(1, 1000)
            for i in range(N):
                U[i] = base + i * rng.randint(0, max(1, X))
                D[i] = rng.randint(1, 1000)
        for i in range(N):
            if U[i] < 1:
                U[i] = 1
            if D[i] < 1:
                D[i] = 1
            if U[i] > 10**9:
                U[i] = 10**9
            if D[i] > 10**9:
                D[i] = 10**9
        tests.append(make_input(N, X, U, D))
    return tests


print(generate_test_inputs(100))
