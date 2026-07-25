"""Render the QEC surface-code circuit to PDF.

Stim emits circuit diagrams as SVG (vector); we rasterize-free convert to PDF
with cairosvg. Default is the d=3 rotated_memory_z timeline — the smallest
distance, so the diagram stays legible. Larger d works but the timeline grows
wide fast.
"""
import argparse
import os
import cairosvg

from qec_decoder.data_gen import build_circuit

OUT_DIR = "figures"


def draw(d: int, p: float, kind: str, fname: str):
    circuit = build_circuit(d, p)
    svg = str(circuit.diagram(kind))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, fname)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=path)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--kind", default="timeline-svg",
                    help="stim diagram type, e.g. timeline-svg, timeslice-svg")
    args = ap.parse_args()
    draw(args.d, args.p, args.kind,
         f"qec_surface_d{args.d}_{args.kind.replace('-svg', '')}.pdf")


if __name__ == "__main__":
    main()
