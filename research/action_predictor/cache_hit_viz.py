"""Visualize WHICH cached frame the retrieval policy hit at each skip, for a real closed-loop rollout.

The closed-loop trace now records, per skip, the executed cache entry's provenance (source episode + the
decision-point frame index) and the key distance (closed_loop.py -> retrieval_policy.last_match). With the
env var DUMP_HIT_IMAGES=1 the eval also saves the LIVE decision-point frame of each skip
(hits_ep*.npz), so we can show the live view beside the cached view it matched.

Produces, for one rollout episode:
  * a FILMSTRIP -- each skip is a column: live decision frame (top) over the matched CACHE frame (bottom),
    captioned with the source episode/frame and the key distance;
  * a TIMELINE  -- rollout decision step vs the source episode hit (coloured by distance), so you can see
    whether retrieval stays on one demo or hops between demos;
  * a source-episode HISTOGRAM (how concentrated the hits are).

Run (from research/):
    ../.venv/bin/python action_predictor/cache_hit_viz.py \
        --run results/closed_loop/PnPSinkToCounter/retrieval_fused_pnp_sink_to_counter \
        --data data/pnp_sink_to_counter_dense_img [--tag even0.5] [--episode 5005] [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _read_traces(tag_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(tag_dir, "skip_trace_ep*.jsonl"))):
        with open(p) as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    return rows


def _read_hits(tag_dir):
    """Concatenate all hits_ep*.npz -> dict of arrays (live frames + provenance), or None if absent."""
    files = sorted(glob.glob(os.path.join(tag_dir, "hits_ep*.npz")))
    if not files:
        return None
    keys = ["live_image", "ep", "decision_idx", "step", "src_ep", "src_imgidx", "src_t", "cache_idx", "dist"]
    acc = {k: [] for k in keys}
    for p in files:
        d = np.load(p)
        for k in keys:
            if k in d.files:
                acc[k].append(d[k])
    return {k: (np.concatenate(v) if v else None) for k, v in acc.items()}


def _ep_file_map(data_dir, _maps={}):
    """{episode_number -> npz path}, robust to zero-padded names (ep0046_success=1.npz)."""
    if data_dir not in _maps:
        m = {}
        for f in glob.glob(os.path.join(data_dir, "ep*_success=*.npz")):
            try:
                m[int(os.path.basename(f).split("ep")[1].split("_")[0])] = f
            except (IndexError, ValueError):
                pass
        _maps[data_dir] = m
    return _maps[data_dir]


def _cache_frame(data_dir, src_ep, imgidx, _imgs={}):
    """Decision-point RGB of cache entry (src_ep, imgidx) = that episode's cur_image[imgidx]."""
    if src_ep is None or src_ep < 0 or imgidx is None or imgidx < 0:
        return None
    if src_ep not in _imgs:
        f = _ep_file_map(data_dir).get(int(src_ep))
        d = np.load(f) if f else None
        _imgs[src_ep] = d["cur_image"] if (d is not None and "cur_image" in d.files) else None
    arr = _imgs[src_ep]
    return None if arr is None or imgidx >= len(arr) else arr[int(imgidx)]


def _ep_length(data_dir, ep, _len={}):
    """Total executed steps T of source episode `ep` -> normalizer so a hit's progress (src step / T) is
    ~1 at the end of that demo."""
    if ep not in _len:
        f = _ep_file_map(data_dir).get(int(ep)) if (ep is not None and ep >= 0) else None
        if f:
            d = np.load(f)
            _len[ep] = (int(d["realized_actions"].shape[0]) if "realized_actions" in d.files
                        else int(len(d["cur_image"])) if "cur_image" in d.files else None)
        else:
            _len[ep] = None
    return _len[ep]


def _rollout_lengths(tag_dir):
    """{episode -> realized rollout length} from the eval_ep*.json sidecars (for live rollout progress)."""
    out = {}
    for p in glob.glob(os.path.join(tag_dir, "eval_ep*.json")):
        try:
            for e in json.load(open(p)).get("results", {}).get("episodes", []):
                out[int(e["ep"])] = int(e["length"])
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="closed-loop eval dir (contains videos/<tag>/) OR a tag dir directly")
    ap.add_argument("--data", required=True, help="source data dir (ep*_success=1.npz) for the cached frames")
    ap.add_argument("--tag", default=None, help="skip setting, e.g. even0.5 (default: the tag with the most hits)")
    ap.add_argument("--episode", type=int, default=None, help="episode to visualize (default: the one with most skips)")
    ap.add_argument("--max-cols", type=int, default=12, help="max filmstrip columns (subsample skips evenly if more)")
    ap.add_argument("--out-dir", default=None, help="where to write the PNG (default: the tag dir)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # locate the tag dir (videos/<tag>/) with skip traces
    if glob.glob(os.path.join(args.run, "skip_trace_ep*.jsonl")):
        tag_dirs = [args.run]
    else:
        tag_dirs = sorted(d for d in glob.glob(os.path.join(args.run, "videos", "*")) if os.path.isdir(d))
    if args.tag:
        tag_dirs = [d for d in tag_dirs if os.path.basename(d) == args.tag]
    assert tag_dirs, f"no skip_trace under {args.run} (looked in videos/<tag>/)"
    # pick the tag with the most skips if not specified
    def n_skips(d):
        return sum(1 for r in _read_traces(d) if r.get("skip"))
    tag_dir = args.tag and tag_dirs[0] or max(tag_dirs, key=n_skips)
    tag = os.path.basename(tag_dir)
    traces = _read_traces(tag_dir)
    hits = _read_hits(tag_dir)

    skips = [r for r in traces if r.get("skip") and r.get("hit_src_ep") is not None]
    assert skips, (f"no skips with provenance in {tag_dir} -- re-run the eval after the instrumentation "
                   f"(and set DUMP_HIT_IMAGES=1 for live frames).")
    # choose episode (default: most skips)
    eps, counts = np.unique([r["ep"] for r in skips], return_counts=True)
    episode = args.episode if args.episode is not None else int(eps[int(np.argmax(counts))])
    ep_skips = sorted((r for r in skips if r["ep"] == episode), key=lambda r: r["step"])
    assert ep_skips, f"episode {episode} has no skips with provenance (have {list(eps)})"

    # live frames (optional) keyed by (ep, step)
    live_by = {}
    if hits is not None and hits.get("live_image") is not None:
        for i in range(len(hits["live_image"])):
            live_by[(int(hits["ep"][i]), int(hits["step"][i]))] = hits["live_image"][i]

    # ---- per-hit progress within its source demo (end of demo = 1) + the live rollout length ----
    for r in ep_skips:
        T = _ep_length(args.data, r.get("hit_src_ep"))
        r["_prog"] = (r["hit_src_t"] / T) if (T and r.get("hit_src_t") is not None) else np.nan
    L_roll = _rollout_lengths(tag_dir).get(episode) or (max(r["step"] for r in ep_skips) + 16)

    # ---- subsample columns for the filmstrip ----
    K = len(ep_skips)
    sel = ep_skips if K <= args.max_cols else [ep_skips[i] for i in
                                               np.linspace(0, K - 1, args.max_cols).round().astype(int)]
    ncol = len(sel)
    have_live = any((episode, r["step"]) in live_by for r in sel)
    img_rows = 2 if have_live else 1

    # Layout: a filmstrip region (tight image spacing) ABOVE a separate analysis region (loose panel
    # spacing) -- nested gridspecs so the bottom panels never overlap.
    fig = plt.figure(figsize=(max(11, 1.45 * ncol), 3.0 * img_rows + 5.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[img_rows * 1.05, 1.7], hspace=0.34)
    gtop = outer[0].subgridspec(img_rows, ncol, wspace=0.05, hspace=0.34)
    gbot = outer[1].subgridspec(1, 3, width_ratios=[1.25, 1.15, 0.75], wspace=0.5)

    def _yl(ax, txt):
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(txt, rotation=0, ha="right", va="center", fontsize=9, labelpad=20)

    for c, r in enumerate(sel):
        se, si, d, pg = r.get("hit_src_ep"), r.get("hit_src_imgidx"), r.get("hit_dist", np.nan), r["_prog"]
        if have_live:
            ax = fig.add_subplot(gtop[0, c]); lf = live_by.get((episode, r["step"]))
            ax.imshow(lf if lf is not None else np.zeros((224, 224, 3), np.uint8)); ax.axis("off")
            ax.set_title(f"step {r['step']}", fontsize=8)
            if c == 0:
                _yl(ax, "LIVE")
        axc = fig.add_subplot(gtop[img_rows - 1, c]); cf = _cache_frame(args.data, se, si)
        axc.imshow(cf if cf is not None else np.zeros((224, 224, 3), np.uint8)); axc.axis("off")
        axc.set_title(f"ep{se} f{si}\nprog={pg:.2f}  d={d:.2f}", fontsize=8)
        if c == 0:
            _yl(axc, "CACHE")

    steps = [r["step"] for r in ep_skips]; ses = [r.get("hit_src_ep") for r in ep_skips]
    ds = np.array([r.get("hit_dist", np.nan) for r in ep_skips])

    # panel 1: which source demo each skip hit, over the rollout (coloured by key distance)
    ax1 = fig.add_subplot(gbot[0, 0])
    ax1.plot(steps, ses, "-", color="0.8", lw=1, zorder=1)
    sc1 = ax1.scatter(steps, ses, c=ds, cmap="viridis_r", s=55, zorder=2, edgecolor="k", linewidth=0.4)
    ax1.set_xlabel("rollout step"); ax1.set_ylabel("source episode hit")
    ax1.set_title("retrieval source over rollout", fontsize=9); ax1.grid(alpha=0.3)
    fig.colorbar(sc1, ax=ax1, label="key dist", fraction=0.046, pad=0.02)

    # panel 2: temporal alignment -- live rollout progress vs the hit cache-frame's progress (end=1)
    ax2 = fig.add_subplot(gbot[0, 1])
    lp = [r["step"] / L_roll for r in ep_skips]; cp = [r["_prog"] for r in ep_skips]
    ax2.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="aligned (y=x)")
    sc2 = ax2.scatter(lp, cp, c=ds, cmap="viridis_r", s=55, edgecolor="k", linewidth=0.4, zorder=3)
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("rollout progress (step / length)")
    ax2.set_ylabel("hit cache-frame progress\n(src step / demo length, end=1)")
    ax2.set_title("temporal alignment of hits", fontsize=9); ax2.grid(alpha=0.3); ax2.legend(fontsize=7, loc="upper left")
    fig.colorbar(sc2, ax=ax2, label="key dist", fraction=0.046, pad=0.02)

    # panel 3: top source episodes
    ax3 = fig.add_subplot(gbot[0, 2])
    u, cnt = np.unique([r.get("hit_src_ep") for r in ep_skips], return_counts=True)
    order = np.argsort(cnt)[::-1][:12]
    ax3.barh([str(int(u[i])) for i in order][::-1], [cnt[i] for i in order][::-1], color="tab:blue")
    ax3.set_xlabel("# hits"); ax3.set_ylabel("src ep"); ax3.set_title("top source demos", fontsize=9)
    ax3.grid(alpha=0.3, axis="x")

    n_distinct = len({r.get("hit_src_ep") for r in ep_skips})
    mean_prog = np.nanmean([r["_prog"] for r in ep_skips])
    fig.suptitle(f"Cache hits  |  {os.path.basename(os.path.dirname(os.path.dirname(tag_dir)))} / {tag} / "
                 f"episode {episode}  |  {len(ep_skips)} skips, {n_distinct} distinct source demos, "
                 f"mean key-dist {np.nanmean(ds):.3f}, mean hit-progress {mean_prog:.2f}"
                 + ("" if have_live else "  (no live frames: set DUMP_HIT_IMAGES=1)"), fontsize=11)
    out_dir = args.out_dir or tag_dir
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"cache_hits_{tag}_ep{episode}.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")
    print(f"  episode {episode}: {len(ep_skips)} skips, {n_distinct} distinct source demos, "
          f"mean key-dist {np.nanmean(ds):.3f}, mean hit-progress {mean_prog:.2f}, "
          f"live_frames={'yes' if have_live else 'no'}")


if __name__ == "__main__":
    main()
