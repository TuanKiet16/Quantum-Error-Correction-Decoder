import numpy as np
import networkx as nx
from itertools import combinations
 
 
def build_check_graph(check_qubit_adj, n_checks):
    G = nx.Graph()
    G.add_nodes_from(range(n_checks))
    G.add_node("B")
    for q, checks in enumerate(check_qubit_adj):
        if len(checks) == 2:
            c1, c2 = checks
            if G.has_edge(c1, c2):
                continue
            G.add_edge(c1, c2, weight=1, qubit=q)
        elif len(checks) == 1:
            c1 = checks[0]
            if G.has_edge(c1, "B"):
                continue
            G.add_edge(c1, "B", weight=1, qubit=q)
    return G
 
 
class MWPMDecoder:
    def __init__(self, check_qubit_adj, n_checks, n_qubits):
        self.G = build_check_graph(check_qubit_adj, n_checks)
        self.n_checks = n_checks
        self.n_qubits = n_qubits
        self.dist, self.path = dict(nx.all_pairs_dijkstra(self.G, weight="weight"))[0] \
            if False else (None, None)
        self._apsp_dist = dict(nx.all_pairs_dijkstra_path_length(self.G, weight="weight"))
        self._apsp_path = dict(nx.all_pairs_dijkstra_path(self.G, weight="weight"))
 
    def decode(self, syndrome):
        triggered = list(np.nonzero(syndrome)[0])
        if len(triggered) == 0:
            return np.zeros(self.n_qubits, dtype=np.uint8)
 
        nodes = []
        H = nx.Graph()
        for i, c in enumerate(triggered):
            nodes.append(("c", c))
            H.add_node(("c", c))
        for i, c in enumerate(triggered):
            H.add_node(("b", i))
            H.add_edge(("c", c), ("b", i), weight=self._apsp_dist[c]["B"])
        for i, j in combinations(range(len(triggered)), 2):
            H.add_edge(("b", i), ("b", j), weight=0)
        for i, j in combinations(range(len(triggered)), 2):
            c1, c2 = triggered[i], triggered[j]
            w = self._apsp_dist[c1][c2]
            H.add_edge(("c", c1), ("c", c2), weight=w)
 
        matching = nx.algorithms.matching.min_weight_matching(H, weight="weight")
 
        correction = np.zeros(self.n_qubits, dtype=np.uint8)
        seen_check_pairs = set()
        for (a, b) in matching:
            if a[0] == "c" and b[0] == "c":
                c1, c2 = a[1], b[1]
                path = self._apsp_path[c1][c2]
                for u, v in zip(path[:-1], path[1:]):
                    if u == "B" or v == "B":
                        continue
                    q = self.G[u][v]["qubit"]
                    correction[q] ^= 1
            elif (a[0] == "c" and b[0] == "b") or (a[0] == "b" and b[0] == "c"):
                cnode = a if a[0] == "c" else b
                c1 = cnode[1]
                path = self._apsp_path[c1]["B"]
                for u, v in zip(path[:-1], path[1:]):
                    q = self.G[u][v]["qubit"]
                    correction[q] ^= 1
        return correction
 
 
class UnionFindDecoder:
    def __init__(self, check_qubit_adj, n_checks, n_qubits):
        self.n_checks = n_checks
        self.n_qubits = n_qubits
        nbr = {c: [] for c in range(n_checks)}
        nbr["B"] = []
        for q, checks in enumerate(check_qubit_adj):
            if len(checks) == 2:
                c1, c2 = checks
                nbr[c1].append((q, c2))
                nbr[c2].append((q, c1))
            elif len(checks) == 1:
                c1 = checks[0]
                nbr[c1].append((q, "B"))
                nbr["B"].append((q, c1))
        self.nbr = nbr
 
    def _is_valid(self, cluster, triggered):
        if "B" in cluster:
            return True
        return sum(1 for x in cluster if x in triggered) % 2 == 0
 
    def decode(self, syndrome):
        triggered = set(np.nonzero(syndrome)[0].tolist())
        if not triggered:
            return np.zeros(self.n_qubits, dtype=np.uint8)
 
        clusters = [set([c]) for c in triggered]
 
        max_iter = self.n_checks + 2
        it = 0
        while it < max_iter:
            it += 1
            invalid = [cl for cl in clusters if not self._is_valid(cl, triggered)]
            if not invalid:
                break
            grown = []
            for cl in clusters:
                if self._is_valid(cl, triggered):
                    grown.append(cl)
                    continue
                new_nodes = set(cl)
                for node in cl:
                    for q, other in self.nbr[node]:
                        new_nodes.add(other)
                grown.append(new_nodes)
            merged = []
            for cl in grown:
                placed_into = None
                for m in merged:
                    if cl & m:
                        placed_into = m
                        break
                if placed_into is None:
                    merged.append(set(cl))
                else:
                    placed_into |= cl
                    changed = True
                    while changed:
                        changed = False
                        for i in range(len(merged)):
                            for j in range(i + 1, len(merged)):
                                if merged[i] & merged[j]:
                                    merged[i] |= merged[j]
                                    del merged[j]
                                    changed = True
                                    break
                            if changed:
                                break
            clusters = merged
 
        correction = np.zeros(self.n_qubits, dtype=np.uint8)
        for cluster in clusters:
            if len(cluster) <= 1 and not (cluster & triggered):
                continue
            G = nx.Graph()
            G.add_nodes_from(cluster)
            for node in cluster:
                for q, other in self.nbr[node]:
                    if other in cluster:
                        G.add_edge(node, other, qubit=q)
            if G.number_of_edges() == 0:
                continue
            T = nx.minimum_spanning_tree(G)
            parity = {node: (1 if node in triggered else 0) for node in T.nodes}
            T2 = T.copy()
            while T2.number_of_edges() > 0:
                leaves = [x for x in T2.nodes if T2.degree(x) == 1 and x != "B"]
                if not leaves:
                    break
                leaf = leaves[0]
                (other,) = T2.neighbors(leaf)
                q = T2[leaf][other]["qubit"]
                if parity[leaf] % 2 == 1:
                    correction[q] ^= 1
                    parity[other] ^= 1
                T2.remove_node(leaf)
        return correction
 
 
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from surface_code import SurfaceCode
    from noise_and_dataset import sample_errors
 
    sc = SurfaceCode(5)
    rng = np.random.default_rng(1)
    mwpm = MWPMDecoder(sc.Zcheck_qubit_adj, sc.mX, sc.n)
    uf = UnionFindDecoder(sc.Zcheck_qubit_adj, sc.mX, sc.n)
 
    n_trials = 300
    fails_mwpm = 0
    fails_uf = 0
    for _ in range(n_trials):
        eX, eZ = sample_errors(sc.n, "iid_phaseflip", rng, p=0.06)
        synd = sc.syndrome_X_from_Zerr(eZ)
        for name, dec, fails in [("mwpm", mwpm, "fails_mwpm"), ("uf", uf, "fails_uf")]:
            corr = dec.decode(synd)
            residual = (corr ^ eZ) % 2
            logical_err = sc.logical_flip_Z(residual)
            if name == "mwpm" and logical_err:
                fails_mwpm += 1
            if name == "uf" and logical_err:
                fails_uf += 1
    print(f"d=5, p=0.06, phase-flip noise, {n_trials} trials")
    print("MWPM logical error rate:", fails_mwpm / n_trials)
    print("UF   logical error rate:", fails_uf / n_trials)
 