import unittest


# Programmer Code
class Solution:
    def sol(self, input_str):
        import sys

        data = list(map(int, input_str.strip().split()))
        it = iter(data)
        N = next(it)
        M = next(it)
        groups = []
        for _ in range(M):
            K = next(it)
            C = next(it)
            S = [next(it) - 1 for _ in range(K)]
            groups.append((C, S))

        # DSU helper
        class DSU:
            def __init__(self, n):
                self.p = list(range(n))
                self.r = [0] * n

            def find(self, x):
                while self.p[x] != x:
                    self.p[x] = self.p[self.p[x]]
                    x = self.p[x]
                return x

            def union(self, a, b):
                a = self.find(a)
                b = self.find(b)
                if a == b:
                    return False
                if self.r[a] < self.r[b]:
                    a, b = b, a
                self.p[b] = a
                if self.r[a] == self.r[b]:
                    self.r[a] += 1
                return True

        # Build candidate edges: for each group, we connect its vertices by K_i-1 edges
        # to represent the clique for Kruskal (safe because MST won't need more than K_i-1 from that clique).
        edges = []
        for C, S in groups:
            if not S:
                continue
            base = S[0]
            for v in S[1:]:
                edges.append((C, base, v))

        # Kruskal over N nodes, edges sorted by weight, but need to allow merging components
        edges.sort()
        dsu = DSU(N)
        total = 0
        cnt = 0

        for w, u, v in edges:
            if dsu.union(u, v):
                total += w
                cnt += 1

        # After connecting via these backbone edges, it may still be disconnected.
        # However, cliques provide additional edges with same cost; but our construction
        # already captures all necessary connections to achieve connectivity if possible.
        if cnt < N - 1:
            # Try to add more edges by using dynamic representatives: For each group, we
            # attempt to connect different DSU components with minimal extra edges.
            # We will connect the DSU components of vertices in S using additional edges of cost C.
            # This is equivalent to running Kruskal where edges exist between all pairs in S at cost C,
            # but we only consider edges that connect distinct components and avoid O(K^2).
            for C, S in sorted(groups):
                # Map comp -> representative vertex
                comp_rep = {}
                for v in S:
                    r = dsu.find(v)
                    if r not in comp_rep:
                        comp_rep[r] = v
                reps = list(comp_rep.values())
                # Connect these components in a chain using cost C
                for i in range(1, len(reps)):
                    if dsu.union(reps[0], reps[i]):
                        total += C
                        cnt += 1
                        if cnt == N - 1:
                            break
                if cnt == N - 1:
                    break

        if cnt != N - 1:
            return "-1"
        return str(total)


# Tester Code
def run_solution(input_str):
    return Solution().sol(input_str).strip()


class TestGraphMST(unittest.TestCase):
    def test_simple_triangle(self):
        # 3 nodes; single operation forms triangle with equal weights
        # MST weight = 2 edges * weight 5 = 10
        input_str = """3 1
3 5
1 2 3
"""
        self.assertEqual(run_solution(input_str), "10")

    def test_disconnected_two_components(self):
        # 4 nodes; edges only among {1,2} and {3,4}, not connected => -1
        input_str = """4 2
2 3
1 2
2 4
3 4
"""
        self.assertEqual(run_solution(input_str), "-1")

    def test_chain_via_cliques(self):
        # 4 nodes; clique on {1,2} cost 1, clique on {2,3} cost 2, clique on {3,4} cost 3
        # Resulting graph connected. MST can take edges: (1-2)=1, (2-3)=2, (3-4)=3 => total 6
        input_str = """4 3
2 1
1 2
2 2
2 3
2 3
3 4
"""
        self.assertEqual(run_solution(input_str), "6")

    def test_multiple_operations_same_pair_keep_min(self):
        # Pair (1,2) added with weight 10 then 1; also 2-3 with 2; triangle 1-3 not present
        # MST edges: (1-2)=1, (2-3)=2 => 3
        input_str = """3 2
2 10
1 2
2 1
1 2
"""
        self.assertEqual(run_solution(input_str), "3")

    def test_full_clique_two_ops(self):
        # N=4; first op adds clique on {1,2,3} weight 4; second op adds clique on {2,3,4} weight 1
        # Graph is fully connected. MST should pick low edges: connect all with weight 1 where possible
        # Available edges weight 1: (2,3), (2,4), (3,4). Need to connect node 1 using (1,2) or (1,3) weight 4.
        # MST: (2,3)=1, (2,4)=1, (1,2)=4 => total 6
        input_str = """4 2
3 4
1 2 3
3 1
2 3 4
"""
        self.assertEqual(run_solution(input_str), "6")

    def test_star_like_connectivity(self):
        # N=5; connect center 3 to others through pairs added by multiple ops
        # Ops: clique on {3,4,5} weight 2; pairs (1,3) weight 5; (2,3) weight 1
        # MST: (3,4)=2, (3,5)=2, (2,3)=1, (1,3)=5 => total 10
        input_str = """5 3
3 2
3 4 5
2 5
1 3
2 1
2 3
"""
        self.assertEqual(run_solution(input_str), "10")

    def test_only_one_op_large_K(self):
        # N=5; single op K=5 clique weight 7; MST=4 edges * 7 = 28
        input_str = """5 1
5 7
1 2 3 4 5
"""
        self.assertEqual(run_solution(input_str), "28")

    def test_redundant_edges_higher_cost(self):
        # Build a path with low costs, then add a high cost clique that shouldn't affect MST
        # Path: 1-2 (1), 2-3 (1), 3-4 (1)
        # High-cost clique {1,2,3,4} weight 100 ignored
        input_str = """4 4
2 1
1 2
2 1
2 3
2 1
3 4
4 100
1 2 3 4
"""
        self.assertEqual(run_solution(input_str), "3")

    def test_disconnected_isolated_node(self):
        # N=4; edges only among 1,2,3; node 4 isolated => -1
        input_str = """4 1
3 3
1 2 3
"""
        self.assertEqual(run_solution(input_str), "-1")

    def test_duplicate_vertices_across_ops(self):
        # Multiple ops adding overlapping cliques with varying weights.
        # Op1: {1,2,3} w=5; Op2: {2,3,4} w=2; Op3: {4,5} w=1
        # To connect node 1 cheapest via 1-2 (5). Then 2-3 (2 via Op2 cheaper than 5), 3-4 (2), 4-5 (1)
        # MST total: 5 + 2 + 2 + 1 = 10
        input_str = """5 3
3 5
1 2 3
3 2
2 3 4
2 1
4 5
"""
        self.assertEqual(run_solution(input_str), "10")

    def test_min_edge_updates_within_clique(self):
        # Same clique appears twice with different weights; use smaller
        # N=3; first clique {1,2,3} w=9, then same clique w=2
        # MST = 2*2 = 4
        input_str = """3 2
3 9
1 2 3
3 2
1 2 3
"""
        self.assertEqual(run_solution(input_str), "4")

    def test_sparse_then_dense(self):
        # Initially disconnected pair, then dense operation connects all cheaply
        # Op1: (1,2) w=50; Op2: clique {1,2,3,4} w=3
        # MST uses 3 edges with w=3 => 9
        input_str = """4 2
2 50
1 2
4 3
1 2 3 4
"""
        self.assertEqual(run_solution(input_str), "9")

    def test_multi_component_then_bridge(self):
        # Two cliques then a bridge via a pair
        # {1,2,3} w=4, {4,5} w=7, bridge (3,4) w=1
        # MST: within first clique pick two edges of w=4 -> 8, connect to second via 1 -> 1, within second pick none (since 4-5 already connected? Need 5 nodes => 4 edges total)
        # Actually MST edges: (1-2)=4, (2-3)=4, (3-4)=1, (4-5)=7 => total 16
        input_str = """5 3
3 4
1 2 3
2 7
4 5
2 1
3 4
"""
        self.assertEqual(run_solution(input_str), "16")

    def test_all_pairs_repeated_varied_costs(self):
        # Build several pairs with multiple costs; ensure min is used per pair in MST selection
        # Edges: (1,2):10 then 3, (2,3):2, (1,3):5, (3,4):1, (4,5):2, (1,5):100
        # MST: (3,4)=1, (2,3)=2, (4,5)=2, (1,2)=3 => total 8
        input_str = """5 6
2 10
1 2
2 2
2 3
2 5
1 3
2 1
3 4
2 2
4 5
2 100
1 5
2 3
1 2
"""
        self.assertEqual(run_solution(input_str), "8")

    def test_minimal_n_two_nodes(self):
        # N=2; single pair added cost 42 => MST 42
        input_str = """2 1
2 42
1 2
"""
        self.assertEqual(run_solution(input_str), "42")

    def test_n_two_nodes_disconnected(self):
        # N=2; no operation connects them? But constraints say K_i >=2; create op on other nodes impossible. Use M=0? Not allowed.
        # Instead, create operation with unrelated pair in larger N. For N=2 must connect; shift to N=3 with only pair 1-2.
        input_str = """3 1
2 7
1 2
"""
        self.assertEqual(run_solution(input_str), "-1")

    def test_large_K_with_extra_pairs(self):
        # N=6; clique on {1,2,3,4} w=10; clique on {3,4,5,6} w=1
        # To connect 1,2 to the rest, must use edges of weight 10 into {3,4}
        # MST: pick cheap clique edges to connect 3,4,5,6: need 3 edges of w=1
        # Then add two edges from {1,2} to connect: e.g., (1,3)=10 and (2,3)=10 => total 3*1 + 2*10 = 23
        input_str = """6 2
4 10
1 2 3 4
4 1
3 4 5 6
"""
        self.assertEqual(run_solution(input_str), "23")

    def test_multiple_small_pairs_connect_all(self):
        # N=5; pairs chain 1-2 (3), 2-3 (1), 3-4 (1), 4-5 (1)
        # MST should be sum = 6
        input_str = """5 4
2 3
1 2
2 1
2 3
2 1
3 4
2 1
4 5
"""
        self.assertEqual(run_solution(input_str), "6")

    def test_overlapping_cliques_with_mixed_costs(self):
        # N=5; {1,2,3} w=3, {2,3,4} w=8, {3,4,5} w=2
        # Best edges: from last clique, pick (3,4)=2, (4,5)=2; from first clique pick (1,2)=3; need connect 2 to 3: from first clique (2,3)=3 better than 8
        # Total: 2 + 2 + 3 + 3 = 10
        input_str = """5 3
3 3
1 2 3
3 8
2 3 4
3 2
3 4 5
"""
        self.assertEqual(run_solution(input_str), "10")

    def test_edge_case_many_ops_same_pair(self):
        # Repeated (1,2) decreasing costs, plus connect to 3
        # Final min cost for (1,2) = 1; add (2,3)=4
        # MST: 1 + 4 = 5
        input_str = """3 5
2 10
1 2
2 7
1 2
2 3
1 2
2 2
1 2
2 4
2 3
"""
        self.assertEqual(run_solution(input_str), "5")

    def test_connectivity_via_single_high_cost_bridge(self):
        # Two dense cheap groups connected by one high-cost pair
        # {1,2,3} w=1, {4,5,6} w=1, bridge (3,4) w=50
        # MST: within each group for 3 nodes need 2 edges each: 2*1 + 2*1 = 4, plus bridge 50 => 54
        input_str = """6 3
3 1
1 2 3
3 1
4 5 6
2 50
3 4
"""
        self.assertEqual(run_solution(input_str), "54")

    def test_disconnected_with_unused_nodes(self):
        # N=6; connect among {1,2,3,4}, leave {5,6} unconnected
        input_str = """6 1
4 2
1 2 3 4
"""
        self.assertEqual(run_solution(input_str), "-1")

    def test_mixed_pairs_and_clique_prefer_pairs(self):
        # N=4; clique {1,2,3,4} w=10; additional cheap pairs (1,2)=1, (2,3)=2
        # MST: choose (1,2)=1, (2,3)=2, and one edge to connect node 4: pick any clique edge 10
        # total 13
        input_str = """4 3
4 10
1 2 3 4
2 1
1 2
2 2
2 3
"""
        self.assertEqual(run_solution(input_str), "13")


if __name__ == "__main__":
    unittest.main(verbosity=2)
