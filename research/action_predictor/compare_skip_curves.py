"""Overlay success-rate vs EFFECTIVE-skip-rate curves from several closed_loop_eval.json runs.

Fair comparison of skip policies: each run already reports, per setting, success_rate and the MEASURED
effective_skip_rate. We plot success vs effective_skip (the honest x-axis -- a gate's q/target is not its
realized skip rate), one line per run, so policies are compared at matched compute savings.

Run (from research/):
    ../.venv/bin/python action_predictor/compare_skip_curves.py \
        --runs results/closed_loop/retrieval_n1_random:random \
               results/closed_loop/retrieval_n1_distgate:distgate \
        --out results/closed_loop/retrieval_n1_distgate/compare_vs_random.png
"""

from __future__ import annotations

import argparse
import json
import math
import os


def load(path):
    d = json.load(open(os.path.join(path, "closed_loop_eval.json")))
    rows = [(r["effective_skip_rate"], r["success_rate"], r["n_success"], r["n_episodes"], r["setting"])
            for r in d["by_skip_rate"]]
    rows.sort()
    return d, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="each is path[:label]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=None,
                    help="path[:label] of a full-VLA (skip-0) run -> horizontal ceiling line")
    ap.add_argument("--baseline-value", type=float, default=None,
                    help="manual full-VLA ceiling value (e.g. 0.56) when there is no skip-0 run file; "
                         "draws the same ceiling line + fraction bands as --baseline")
    ap.add_argument("--baseline-label", default="full VLA", help="legend label for --baseline-value")
    ap.add_argument("--title", default="Closed-loop: success vs effective skip rate")
    ap.add_argument("--no-error-bars", action="store_true", help="hide the binomial +/-1 SE bars")
    ap.add_argument("--baseline-fracs", default="0.95,0.9",
                    help="with --baseline, shade bands at these fractions of the baseline (e.g. 0.95,0.9)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    print(f"{'run':12s} {'setting':10s} {'eff_skip':>8s} {'success':>8s}  n")
    for spec in args.runs:
        path, _, label = spec.partition(":")
        label = label or os.path.basename(path.rstrip("/"))
        d, rows = load(path)
        xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
        yerr = None if args.no_error_bars else [math.sqrt(max(p * (1 - p), 0) / n) for _, p, _, n, _ in rows]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", ms=5, capsize=3, label=f"{label} ({d.get('task','?')})")
        for es, sr, ns, ne, setting in rows:
            print(f"{label:14s} {setting:10s} {es:8.3f} {sr:8.3f}  {ns}/{ne}")
    bp, blabel, bnote = None, args.baseline_label, ""   # full-VLA ceiling (skip-0 run OR manual value)
    if args.baseline_value is not None:
        bp = float(args.baseline_value)
    elif args.baseline:
        bpath, _, lab = args.baseline.partition(":")
        bd, brows = load(bpath)
        bp = brows[0][1]; blabel = lab or "full VLA"; bnote = f"  [{brows[0][2]}/{brows[0][3]}]"
        print(f"{blabel:14s} {'skip0':10s} {brows[0][0]:8.3f} {bp:8.3f}  {brows[0][2]}/{brows[0][3]}")
    if bp is not None:
        # shaded bands at fractions of the baseline (e.g. 95% / 90% of full-VLA accuracy)
        fracs = sorted({float(x) for x in args.baseline_fracs.split(",") if x.strip()}, reverse=True)
        edges = [1.0] + fracs                            # [1.0, 0.95, 0.9]
        band_colors = ["tab:green", "gold", "tab:orange", "tab:red"]
        for i, f in enumerate(fracs):
            hi, lo, c = edges[i] * bp, f * bp, band_colors[i % len(band_colors)]
            ax.axhspan(lo, hi, color=c, alpha=0.13, zorder=0,
                       label=f"{int(round(f*100))}-{int(round(edges[i]*100))}% of full VLA  ({lo:.3f}-{hi:.3f})")
            ax.axhline(lo, ls=":", color=c, lw=1, zorder=1)
        ax.axhline(bp, ls="--", color="black", lw=1.5, label=f"{blabel} (skip 0) = {bp:.2f}{bnote}")
        for f in fracs:
            print(f"{'  baseline x'+format(f,'.2f'):24s} -> {f*bp:.3f}")
    ax.set_xlabel("effective skip rate (measured fraction of VLA calls saved)")
    ax.set_ylabel("success rate")
    ax.set_title(args.title)
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130); plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
