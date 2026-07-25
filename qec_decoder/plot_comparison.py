"""Plot decoder comparison (p_L vs physical error rate) from a sweep result.

Reads results/comparison.json (written by qec_decoder.evaluate) and draws one
panel per code distance: logical error rate against physical error rate p, on a
log y-axis, with binomial error bars, one line per decoder. Saved as vector PDF.
"""
import argparse
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

OUT = "figures/decoder_comparison.pdf"
_MARK = {"mwpm": "o", "qcnn_cong": "s", "qcnn_hybrid": "^", "cnn": "D"}


def plot(result: dict, out: str = OUT) -> str:
    pts = result["points"]
    ds = sorted({r["d"] for r in pts})
    lam = result.get("lambda", {})

    fig, axes = plt.subplots(1, len(ds), figsize=(6 * len(ds), 5), squeeze=False)
    for ax, d in zip(axes[0], ds):
        series = defaultdict(list)
        for r in pts:
            if r["d"] == d:
                series[r["decoder"]].append(r)
        for name, rows in sorted(series.items()):
            rows.sort(key=lambda r: r["p"])
            xs = [r["p"] for r in rows]
            ys = [r["logical_error_rate"] for r in rows]
            es = [r["uncertainty"] for r in rows]
            ax.errorbar(xs, ys, yerr=es, marker=_MARK.get(name, "x"),
                        capsize=3, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("physical error rate  p")
        ax.set_ylabel("logical error rate  $p_L$")
        title = f"d = {d}"
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    lam_txt = "  ".join(f"{k} Λ={v:.2f}" for k, v in sorted(lam.items()))
    fig.suptitle(f"Decoder comparison  (shots={result.get('shots','?')})   {lam_txt}",
                 fontsize=13)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/comparison.json")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    result = json.load(open(a.inp))
    print(plot(result, a.out))


if __name__ == "__main__":
    main()
