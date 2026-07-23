import numpy as np
import itertools
 
 
# GF(2)
 
def repetition_H(d):
    H = np.zeros((d - 1, d), dtype=np.uint8)
    for i in range(d - 1):
        H[i, i] = H[i, i + 1] = 1
    return H
 
 
def hypergraph_product(H1, H2):
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    Im1, In1 = np.eye(m1, dtype=np.uint8), np.eye(n1, dtype=np.uint8)
    Im2, In2 = np.eye(m2, dtype=np.uint8), np.eye(n2, dtype=np.uint8)
    HX = np.concatenate([np.kron(H1, In2), np.kron(Im1, H2.T)], axis=1) % 2
    HZ = np.concatenate([np.kron(In1, H2), np.kron(H1.T, Im2)], axis=1) % 2
    return HX.astype(np.uint8), HZ.astype(np.uint8)
 
 
def gf2_rref(M):
    M = M.copy() % 2
    rows, cols = M.shape
    pivot_row, pivots = 0, []
    for col in range(cols):
        pivot = next((r for r in range(pivot_row, rows) if M[r, col]), None)
        if pivot is None:
            continue
        M[[pivot_row, pivot]] = M[[pivot, pivot_row]]
        for r in range(rows):
            if r != pivot_row and M[r, col]:
                M[r] = (M[r] + M[pivot_row]) % 2
        pivots.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break
    return M, pivots
 
 
def gf2_nullspace(M):
    rows, cols = M.shape
    R, pivots = gf2_rref(M)
    free_cols = [c for c in range(cols) if c not in pivots]
    basis = []
    for fc in free_cols:
        vec = np.zeros(cols, dtype=np.uint8)
        vec[fc] = 1
        for i, pc in enumerate(pivots):
            if R[i, fc]:
                vec[pc] = 1
        basis.append(vec)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, cols), dtype=np.uint8)
 
 
def gf2_rank(M):
    return 0 if M.shape[0] == 0 else len(gf2_rref(M)[1])
 
 
def in_rowspace(vec, M):
    aug = np.concatenate([M, vec.reshape(1, -1)], axis=0) % 2
    return gf2_rank(aug) == gf2_rank(M)
 
 
# Main class 
 
class SurfaceCode:
    def __init__(self, d):
        assert d >= 3 and d % 2 == 1, 
        self.d = d
        Hrep = repetition_H(d)
        self.HX, self.HZ = hypergraph_product(Hrep, Hrep)
        self.n  = self.HX.shape[1]
        self.mX = self.HX.shape[0]
        self.mZ = self.HZ.shape[0]
 
        self.logical_Z = self._find_logical(self.HX, self.HZ)
        self.logical_X = self._find_logical(self.HZ, self.HX)
 
        # legacy attributes used by decoders / benchmark
        self.block1_shape      = (d, d)
        self.block2_shape      = (d - 1, d - 1)
        self.Xcheck_qubit_adj  = self._check_adjacency(self.HZ)
        self.Zcheck_qubit_adj  = self._check_adjacency(self.HX)
 
        # Build coordinate maps 
        self.grid_shape = (2 * d - 1, 2 * d - 1)   # rows × cols  (y, x)
        self._q_xy  = self._build_qubit_coords()
        self._xc_xy = self._build_xcheck_coords()
        self._zc_xy = self._build_zcheck_coords()
 
    # coordinate builders 
 
    def _build_qubit_coords(self):
        d   = self.d
        coords = {}
        for i in range(d):
            for j in range(d):
                q = i * d + j
                coords[q] = (2 * j, 2 * i)
        for i in range(d - 1):
            for j in range(d - 1):
                q = d * d + i * (d - 1) + j
                coords[q] = (2 * j + 1, 2 * i + 1)
        return coords
 
    def _build_xcheck_coords(self):
        d      = self.d
        coords = {}
        for i in range(d - 1):
            for j in range(d):
                c = i * d + j
                coords[c] = (2 * j, 2 * i + 1)
        return coords
 
    def _build_zcheck_coords(self):
        d      = self.d
        coords = {}
        for i in range(d):
            for j in range(d - 1):
                c = i * (d - 1) + j
                coords[c] = (2 * j + 1, 2 * i)
        return coords
 
 
    def qubit_coords(self):
        return [self._q_xy[q] for q in range(self.n)]
 
    def check_coords(self):
        xc = [self._xc_xy[c] for c in range(self.mX)]
        zc = [self._zc_xy[c] for c in range(self.mZ)]
        return xc, zc
 
    def lattice_summary(self):
        print(f"\n{'='*56}")
        print(f"  Surface code  d={self.d}   n={self.n}   "
              f"mX={self.mX}   mZ={self.mZ}")
        print(f"  grid_shape (rows×cols) = {self.grid_shape}")
        print(f"{'='*56}")
        print("\nData qubits  (index : x, y) :")
        for q in range(self.n):
            x, y = self._q_xy[q]
            tag = "B1" if q < self.d**2 else "B2"
            print(f"  q{q:3d} [{tag}]  x={x}  y={y}")
        print("\nX-stabilisers  (index : x, y) :")
        for c in range(self.mX):
            x, y = self._xc_xy[c]
            print(f"  X{c:3d}  x={x}  y={y}")
        print("\nZ-stabilisers  (index : x, y) :")
        for c in range(self.mZ):
            x, y = self._zc_xy[c]
            print(f"  Z{c:3d}  x={x}  y={y}")
 
    # 2-D grid mappings 
    def syndrome_to_grid(self, syndrome, check_type="X"):
        H, W = self.grid_shape
        grid = np.zeros((H, W), dtype=np.float32)
        if check_type == "X":
            for c, val in enumerate(syndrome):
                x, y = self._xc_xy[c]
                grid[y, x] = float(val)
        else:
            for c, val in enumerate(syndrome):
                x, y = self._zc_xy[c]
                grid[y, x] = float(val)
        return grid
 
    def error_to_grid(self, error):
        H, W = self.grid_shape
        grid = np.zeros((H, W), dtype=np.float32)
        for q, val in enumerate(error):
            x, y = self._q_xy[q]
            grid[y, x] = float(val)
        return grid
 
    def batch_syndrome_to_grid(self, syndromes, check_type="X"):
        N = syndromes.shape[0]
        H, W = self.grid_shape
        batch = np.zeros((N, H, W), dtype=np.float32)
        if check_type == "X":
            coord_map = self._xc_xy
            m = self.mX
        else:
            coord_map = self._zc_xy
            m = self.mZ
        for c in range(m):
            x, y = coord_map[c]
            batch[:, y, x] = syndromes[:, c].astype(np.float32)
        return batch
 
    # syndrome 
 
    def syndrome_X_from_Zerr(self, eZ):
        return (self.HX @ eZ) % 2
 
    def syndrome_Z_from_Xerr(self, eX):
        return (self.HZ @ eX) % 2
 
    def logical_flip_X(self, eX):
        return int((self.logical_Z @ eX) % 2)
 
    def logical_flip_Z(self, eZ):
        return int((self.logical_X @ eZ) % 2)
 
    # visualisation 
 
    def visualize_lattice(self, error_X=None, error_Z=None,
                          syndrome_X=None, syndrome_Z=None,
                          title=None, ax=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
 
        if error_X  is None: error_X  = np.zeros(self.n,  dtype=np.uint8)
        if error_Z  is None: error_Z  = np.zeros(self.n,  dtype=np.uint8)
        if syndrome_X is None: syndrome_X = np.zeros(self.mX, dtype=np.uint8)
        if syndrome_Z is None: syndrome_Z = np.zeros(self.mZ, dtype=np.uint8)
 
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(max(6, self.d * 1.5),
                                            max(6, self.d * 1.5)))
 
        # draw stabiliser–qubit edges
        for c in range(self.mX):
            cx, cy = self._xc_xy[c]
            for q in np.nonzero(self.HX[c])[0]:
                qx, qy = self._q_xy[q]
                ax.plot([cx, qx], [cy, qy], color="#f4a261", lw=0.8,
                        alpha=0.6, zorder=1)
 
        for c in range(self.mZ):
            cx, cy = self._zc_xy[c]
            for q in np.nonzero(self.HZ[c])[0]:
                qx, qy = self._q_xy[q]
                ax.plot([cx, qx], [cy, qy], color="#52b788", lw=0.8,
                        alpha=0.6, zorder=1)
 
        # draw X-stabilisers 
        for c in range(self.mX):
            x, y = self._xc_xy[c]
            triggered = bool(syndrome_X[c])
            fc = "#f4a261" if triggered else "#fff3e0"
            ax.scatter(x, y, marker="s", s=220, color=fc,
                       edgecolors="#e07b39", linewidths=1.5, zorder=3)
            ax.text(x, y, f"X{c}", ha="center", va="center",
                    fontsize=5.5, fontweight="bold", color="#7b3a10", zorder=4)
 
        # draw Z-stabilisers 
        for c in range(self.mZ):
            x, y = self._zc_xy[c]
            triggered = bool(syndrome_Z[c])
            fc = "#52b788" if triggered else "#d8f3dc"
            ax.scatter(x, y, marker="D", s=220, color=fc,
                       edgecolors="#1b7a40", linewidths=1.5, zorder=3)
            ax.text(x, y, f"Z{c}", ha="center", va="center",
                    fontsize=5.5, fontweight="bold", color="#1b4332", zorder=4)
 
        # draw data qubits
        for q in range(self.n):
            x, y     = self._q_xy[q]
            has_X    = bool(error_X[q])
            has_Z    = bool(error_Z[q])
            if   has_X and has_Z: fc, ec = "#9b59b6", "#6c3483"   # Y error
            elif has_X:           fc, ec = "#e74c3c", "#922b21"   # X error
            elif has_Z:           fc, ec = "#2980b9", "#1a5276"   # Z error
            else:                 fc, ec = "#ecf0f1", "#7f8c8d"   # no error
            tag = "B1" if q < self.d**2 else "B2"
            ax.scatter(x, y, marker="o", s=300, color=fc,
                       edgecolors=ec, linewidths=1.5, zorder=5)
            ax.text(x, y, f"q{q}", ha="center", va="center",
                    fontsize=5, color="#2c3e50", zorder=6)
 
        ax.set_aspect("equal")
        ax.set_xlim(-1, 2 * self.d - 1)
        ax.set_ylim(-1, 2 * self.d - 1)
        ax.set_xticks(range(2 * self.d - 1))
        ax.set_yticks(range(2 * self.d - 1))
        ax.grid(True, alpha=0.15)
        ax.set_xlabel("x coordinate"); ax.set_ylabel("y coordinate")
        t = title or f"Surface code  d={self.d}  (n={self.n} qubits)"
        ax.set_title(t, fontsize=10, fontweight="bold")
 
        legend = [
            mpatches.Patch(color="#f4a261", label="X-stab (triggered)"),
            mpatches.Patch(color="#52b788", label="Z-stab (triggered)"),
            mpatches.Patch(color="#e74c3c", label="X error"),
            mpatches.Patch(color="#2980b9", label="Z error"),
            mpatches.Patch(color="#9b59b6", label="Y error"),
        ]
        ax.legend(handles=legend, fontsize=6, loc="upper right")
 
        if standalone:
            plt.tight_layout()
            return fig, ax
        return ax
 
    
 
    @staticmethod
    def _find_logical(H_detecting, H_same_type):
        ker = gf2_nullspace(H_detecting)
        for vec in ker:
            if vec.sum() and not in_rowspace(vec, H_same_type):
                return vec
        raise RuntimeError("no logical operator found – check construction")
 
    @staticmethod
    def _check_adjacency(H):
        m, n = H.shape
        return [np.nonzero(H[:, q])[0].tolist() for q in range(n)]
 
 
# CLI self-test
 
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    for d in [3, 5, 7]:
        sc = SurfaceCode(d)
        print(f"d={d}: n={sc.n}  mX={sc.mX}  mZ={sc.mZ}  "
              f"wt(logX)={sc.logical_X.sum()}  wt(logZ)={sc.logical_Z.sum()}")
 
    sc = SurfaceCode(3)
    sc.lattice_summary()
 
    # Grid mapping check
    syn_test = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)[:sc.mX]
    grid = sc.syndrome_to_grid(syn_test, check_type="X")
    print(f"\nSyndrome grid shape: {grid.shape}\n{grid}")
 
    # Visualise a small example with a random error
    rng = np.random.default_rng(0)
    eX  = (rng.random(sc.n) < 0.15).astype(np.uint8)
    eZ  = (rng.random(sc.n) < 0.15).astype(np.uint8)
    sX  = sc.syndrome_X_from_Zerr(eZ)
    sZ  = sc.syndrome_Z_from_Xerr(eX)
    fig, ax = sc.visualize_lattice(eX, eZ, sX, sZ,
                                   title=f"d=3 lattice with random errors")
    