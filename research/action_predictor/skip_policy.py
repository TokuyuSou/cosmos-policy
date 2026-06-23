"""Skip policies for closed-loop control: at each decision point, decide whether to
CALL the cloud VLA (Cosmos) or SKIP it and use the local action predictor.

Extensible: add new policies (envelope gate, risk head, VLA-disagreement, ...) by
subclassing SkipPolicy. The caller only asks `decide(ctx)` when skipping is *allowed*
(e.g. a cloud cache and action history exist); policies need not re-check that.
"""

from __future__ import annotations

import os
from fractions import Fraction

import numpy as np

GRIP = 6      # action dim of the (near-binary) gripper command
EPS = 1e-6


class SkipPolicy:
    name = "base"

    def reset(self) -> None:
        """Called at the start of each episode."""

    def decide(self, ctx: dict) -> bool:
        """Return True to SKIP the cloud (use local predictor), False to CALL the cloud.

        `ctx` carries info for smarter future policies, e.g.:
          decision_idx, step, cached_future_proprio, cached_future_img, prev_actions.
        """
        raise NotImplementedError


class RandomSkipPolicy(SkipPolicy):
    """Skip the cloud with a fixed probability (independent per decision)."""

    name = "random"

    def __init__(self, skip_rate: float, seed: int = 0):
        assert 0.0 <= skip_rate <= 1.0
        self.skip_rate = float(skip_rate)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)

    def reset(self) -> None:
        # Fresh, deterministic stream per episode is set by the caller via a new instance
        # or seed; keep the running stream here so repeated episodes differ.
        pass

    def decide(self, ctx: dict) -> bool:
        r = float(self._rng.random())
        self.last = {"score": r, "tau": self.skip_rate}   # skip iff score < tau
        return bool(r < self.skip_rate)


class EvenSkipPolicy(SkipPolicy):
    """Deterministic, MAXIMALLY-EVEN skip pattern at a target rate -- the anti-clustering counterpart of
    RandomSkipPolicy. The skip rate p/q (the target rationalised with a bounded denominator) is realised by
    a Bresenham / error-diffusion accumulator: `acc += p`, and we SKIP whenever `acc >= q` (then `acc -= q`).
    This spreads the p skips as evenly as possible over every q decisions -- consecutive-CALL gaps are only
    floor(q/p) or ceil(q/p), so skips NEVER cluster the way random ones do by chance (the failure mode that
    compounds open-loop drift). r=1/3 -> 0,0,1,0,0,1...; r=1/2 -> 0,1,0,1...; integer arithmetic makes the
    pattern exact (no float drift). The accumulator advances only on can-skip decisions and resets each
    episode (identical phase per episode). Same maximally-even guarantee as a Euclidean/Bjorklund rhythm,
    reached more simply. Same interface as the other gates: decide(ctx)->bool, `self.last` trace dict, `name`.
    """

    name = "even"

    def __init__(self, skip_rate: float, max_denom: int = 1000):
        assert 0.0 <= float(skip_rate) <= 1.0
        self.skip_rate = float(skip_rate)
        fr = Fraction(self.skip_rate).limit_denominator(int(max_denom))  # skip p out of every q (exact, even)
        self.p, self.q = fr.numerator, fr.denominator
        self._acc = 0

    def reset(self) -> None:
        self._acc = 0  # restart the even pattern at the start of each episode

    def decide(self, ctx: dict) -> bool:
        self._acc += self.p
        skip = self._acc >= self.q
        if skip:
            self._acc -= self.q
        self.last = {"acc": int(self._acc), "p": self.p, "q": self.q, "tau": self.skip_rate}
        return bool(skip)


class GateSkipPolicy(SkipPolicy):
    """Confidence gate (no VLA call): SKIP -- adopt the predictor output -- only when that output is
    corroborated by state-similarity. Concretely, the cache chunk nearest the predictor's output (its
    L1 pick) must also lie within the top-K of the proprio+prev-actions (N1) retrieval for the current
    state. Offline this isolates a near-oracle-accuracy subset (see research/retrieval_oracle/gate_eval).

    `predictor` is the same PredictorPolicy executed on a skip; the cache (chunks + N1 keys) is built
    from the train split of `data_dir` (identical split to the action-predictor eval).
    """

    name = "gate"

    def __init__(self, predictor, data_dir, K, val_frac: float = 0.15, seed: int = 0):
        from dataset import build_samples, list_success_episodes, split_episode_files
        self.predictor = predictor
        self.K = int(K)
        tr, _ = split_episode_files(list_success_episodes(data_dir), val_frac, seed)
        s = build_samples(tr, [], ["vla_future_proprio", "actual_next_proprio"])
        cact = np.stack([x.target for x in s]).astype(np.float32)               # (Nc,16,7)
        cur = np.stack([x.states["actual_next_proprio"] for x in s]).astype(np.float32)  # (Nc,9)
        prev = np.stack([x.prev_actions for x in s]).astype(np.float32)         # (Nc,16,7)
        self.c_flat = cact.reshape(cact.shape[0], -1)                           # (Nc,112) for L1
        self.pm, self.ps = cur.mean(0), cur.std(0) + 1e-6
        cc = np.concatenate([prev, cact], 0).reshape(-1, 7)
        self.am, self.asd = cc.mean(0), cc.std(0) + 1e-6
        self.c_key = self._n1(cur, prev)                                        # (Nc,121) N1 keys

    def _n1(self, cur, prev):
        cur = np.atleast_2d(np.asarray(cur, np.float32))
        prev = np.asarray(prev, np.float32).reshape(cur.shape[0], -1, 7)
        return np.concatenate([(cur - self.pm) / self.ps,
                               ((prev - self.am) / self.asd).reshape(cur.shape[0], -1)], axis=1).astype(np.float32)

    def decide(self, ctx: dict) -> bool:
        chat = self.predictor.predict_chunk(ctx["prev_actions"], ctx["current_proprio"],
                                            ctx["cached_future_proprio"], ctx["cached_future_img"])
        jstar = int(np.linalg.norm(self.c_flat - chat.reshape(-1)[None, :], axis=1).argmin())  # L1 pick
        q = self._n1(ctx["current_proprio"], ctx["prev_actions"])[0]
        dstate = np.linalg.norm(self.c_key - q[None, :], axis=1)
        kk = min(self.K, dstate.shape[0] - 1)
        in_topk = bool(jstar in set(np.argpartition(dstate, kk)[:kk].tolist()))  # L1 pick within N1 top-K?
        self.last = {"in_topk": in_topk, "K": self.K}
        return in_topk


_DISTGATE_CACHE = {}  # one (RetrievalPolicy, calib-d1) per (data_dir, key, knn, state_source, cache_episodes, split)


def _build_distgate(data_dir, key, knn, state_source, cache_episodes, val_frac, seed, device):
    """Build the SAME cache+key the deployed N1 retrieval uses, plus the held-out (val) d1 distribution
    used to calibrate the threshold. Returns (RetrievalPolicy, calib_d1[Nval]). Cached across q values."""
    import torch
    from retrieval_policy import RetrievalPolicy

    from dataset import build_samples, list_success_episodes, split_episode_files

    rp = RetrievalPolicy(data_dir, key=key, k=knn, state_source=state_source,
                         cache_episodes=(cache_episodes or None))
    val_files = split_episode_files(list_success_episodes(data_dir), val_frac, seed)[1]
    vs = build_samples(val_files, rp.views, [state_source])
    vkeys = np.stack([rp._key(s.prev_actions, s.states[state_source], s.future_img)
                      for s in vs]).astype(np.float32)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    calib = torch.cdist(torch.as_tensor(vkeys, device=dev),
                        torch.as_tensor(rp.keys, device=dev)).min(1).values.cpu().numpy()  # held-out d1
    return rp, calib


class DistGateSkipPolicy(SkipPolicy):
    """Out-of-support gate (no VLA call): SKIP -- trust the N1 retrieval -- only when the query is
    IN-SUPPORT of the cache, i.e. the nearest cache-key distance d1 is small. When d1 is large the
    state is off the success manifold (drift/failure-prone, where even the oracle match degrades --
    see research/retrieval_oracle/dist_gate_probe.py), so CALL the VLA. d1 is literally the deployed
    retrieval's own nearest-neighbour distance (this reuses RetrievalPolicy's cache + key), so the
    gate adds no model and nothing task/VLA-specific.

    tau = q-quantile of d1 over the held-out (val) decision points, so q ~= the in-support skip rate
    (skip the closest fraction q). The closed-loop EFFECTIVE skip rate is measured (typically < q,
    because rollouts also visit off-support states the gate routes to the VLA -- exactly the point).
    Same interface/convention as the other gates: decide(ctx)->bool, `self.last` trace dict, `name`.
    """

    name = "distgate"

    def __init__(self, data_dir, q, key="prev_state", knn=1, state_source="actual_next_proprio",
                 cache_episodes=0, val_frac: float = 0.15, seed: int = 0, device=None):
        ck = (os.path.abspath(data_dir), key, int(knn), state_source, int(cache_episodes or 0), val_frac, seed)
        if ck not in _DISTGATE_CACHE:
            _DISTGATE_CACHE[ck] = _build_distgate(data_dir, key, knn, state_source, cache_episodes,
                                                  val_frac, seed, device)
        self.rp, calib = _DISTGATE_CACHE[ck]
        self.q, self.state_source = float(q), state_source
        self.tau = float(np.quantile(calib, self.q))  # skip the closest (smallest-d1) fraction q

    def reset(self) -> None:
        pass

    def decide(self, ctx: dict) -> bool:
        state = (ctx["current_proprio"] if self.state_source == "actual_next_proprio"
                 else ctx["cached_future_proprio"])
        qk = self.rp._key(np.asarray(ctx["prev_actions"], np.float32), np.asarray(state, np.float32),
                          ctx.get("cached_future_img"))
        d1 = float(np.linalg.norm(self.rp.keys - qk[None, :], axis=1).min())  # the retrieval's own NN distance
        self.last = {"score": d1, "tau": self.tau, "q": self.q, "n1_dist": d1}  # skip iff score < tau
        return bool(d1 < self.tau)   # SKIP (trust retrieval) iff in-support (small d1)


_DISAGREE_CACHE = {}  # one (RetrievalPolicy, calib) per (data_dir, key, knn, state_source, cache_eps, K, w, split)


def _build_disagree(data_dir, key, knn, state_source, cache_episodes, K, w_grip, val_frac, seed, device):
    """Build the SAME cache the deployed N1 retrieval uses (RetrievalPolicy), plus the held-out (val)
    disagree/grip gate distribution used to calibrate the z-stats and tau. For each VAL decision point
    (disjoint from the train cache -> leakage-free, exactly the deploy condition):
        disagree = RMSE(top-1 chunk, mean of the top-K neighbour chunks)   -- multimodality / ambiguity
        grip     = gripper-action range over the top-1 chunk               -- grasp/release criticality
    z-stats (mean/std) and gate = z(disagree)+w*z(grip) are returned. Cached across q values."""
    import torch
    from retrieval_policy import RetrievalPolicy

    from dataset import build_samples, list_success_episodes, split_episode_files

    rp = RetrievalPolicy(data_dir, key=key, k=knn, state_source=state_source,
                         cache_episodes=(cache_episodes or None))
    val_files = split_episode_files(list_success_episodes(data_dir), val_frac, seed)[1]
    vs = build_samples(val_files, rp.views, [state_source])
    vkeys = np.stack([rp._key(s.prev_actions, s.states[state_source], s.future_img)
                      for s in vs]).astype(np.float32)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    D = torch.cdist(torch.as_tensor(vkeys, device=dev), torch.as_tensor(rp.keys, device=dev))
    topk = D.topk(min(int(K), rp.keys.shape[0]), dim=1, largest=False).indices.cpu().numpy()  # (Nval,K) nearest-first
    nb = rp.values[topk]                                       # (Nval,K,16,7) neighbour chunks (incl. top-1)
    t1 = rp.values[topk[:, 0]]                                 # (Nval,16,7) top-1 = chunk the retrieval executes
    dis = np.sqrt(((t1 - nb.mean(1)) ** 2).reshape(topk.shape[0], -1).mean(1))   # (Nval,)
    grip = t1[..., GRIP].max(-1) - t1[..., GRIP].min(-1)                          # (Nval,)
    dm, ds = float(dis.mean()), float(dis.std() + EPS)
    gm, gs = float(grip.mean()), float(grip.std() + EPS)
    gate = (dis - dm) / ds + float(w_grip) * (grip - gm) / gs
    return rp, {"dm": dm, "ds": ds, "gm": gm, "gs": gs, "gate": gate.astype(np.float32),
                "K": int(K), "w": float(w_grip), "n_cache": int(rp.keys.shape[0]), "n_val": int(topk.shape[0])}


class DisagreeGateSkipPolicy(SkipPolicy):
    """Neighbour-DISAGREEMENT safety gate for N1 top-1 retrieval -- the best offline selective-risk gate
    (research/skip_v2 P9: corr 0.80-0.83 with action error; halves the safe-50% error, near oracle).
    SKIP -- trust the retrieved chunk -- only when the retrieval is UNAMBIGUOUS:
        disagree = RMSE(top-1 chunk, mean of the top-K neighbour chunks)  -- multimodality / ambiguity
        grip     = gripper-action range over the top-1 chunk              -- grasp/release criticality
        gate     = z(disagree) + w*z(grip)
    SKIP iff gate < tau, else CALL the VLA. High disagreement = the cache offers conflicting futures here
    (a fork / contact phase the local lookup can't resolve) -> defer to the fresh-perception VLA. Unlike
    distgate (key-space coverage d1, which offline does NOT track action error), disagree reads conflict
    in the VALUE space. The top-1 and its K neighbours come from RetrievalPolicy's own cache + key, so the
    gate scores EXACTLY the chunk the retrieval executes; it adds no model and nothing task/VLA-specific.

    Calibration (held-out val, leakage-free -- same convention as DistGate): z-stats (dm,ds,gm,gs) AND
    tau = q-quantile of the gate are fit on the val decision points scored against the disjoint train
    cache, so q ~= the in-distribution skip fraction. The closed-loop EFFECTIVE skip rate is measured
    (typically < q: rollouts also visit ambiguous states the gate routes to the VLA -- the point). A drift
    budget (runner-level --max-skips) caps consecutive skips (ambiguity clusters in time -- campaign P2b).
    Same interface as the other gates: decide(ctx)->bool, `self.last` trace dict, `name`.
    """

    name = "disagree"

    def __init__(self, data_dir, q, key="prev_state", knn=1, state_source="actual_next_proprio",
                 cache_episodes=0, K=10, w_grip=1.0, val_frac: float = 0.15, seed: int = 0, device=None):
        ck = (os.path.abspath(data_dir), key, int(knn), state_source, int(cache_episodes or 0),
              int(K), float(w_grip), val_frac, seed)
        if ck not in _DISAGREE_CACHE:
            _DISAGREE_CACHE[ck] = _build_disagree(data_dir, key, knn, state_source, cache_episodes,
                                                  int(K), float(w_grip), val_frac, seed, device)
        self.rp, c = _DISAGREE_CACHE[ck]
        self.q, self.state_source = float(q), state_source
        self.K, self.w = c["K"], c["w"]
        self.dm, self.ds, self.gm, self.gs = c["dm"], c["ds"], c["gm"], c["gs"]
        self.n_cache, self.n_val = c["n_cache"], c["n_val"]
        self.tau = float(np.quantile(c["gate"], self.q))  # skip the lowest-gate (most unambiguous) fraction q

    def reset(self) -> None:
        pass

    def decide(self, ctx: dict) -> bool:
        state = (ctx["current_proprio"] if self.state_source == "actual_next_proprio"
                 else ctx["cached_future_proprio"])
        qk = self.rp._key(np.asarray(ctx["prev_actions"], np.float32), np.asarray(state, np.float32),
                          ctx.get("cached_future_img"))
        d = np.linalg.norm(self.rp.keys - qk[None, :], axis=1)
        topk = np.argpartition(d, min(self.K, d.shape[0] - 1))[: self.K]  # K nearest (unordered)
        top1 = self.rp.values[int(topk[d[topk].argmin()])]               # the chunk the retrieval executes
        dis = float(np.sqrt(((top1 - self.rp.values[topk].mean(0)) ** 2).mean()))
        grip = float(top1[:, GRIP].max() - top1[:, GRIP].min())
        gate = (dis - self.dm) / self.ds + self.w * (grip - self.gm) / self.gs
        self.last = {"score": float(gate), "tau": self.tau, "q": self.q, "disagree": dis, "grip": grip}
        return bool(gate < self.tau)   # SKIP (trust retrieval) iff low disagreement+grip


class OracleRmseGateSkipPolicy(SkipPolicy):
    """Oracle upper bound on any retrieval-vs-VLA disagreement gate -- it isolates the ceiling that gate
    engineering can reach for N1 top-1 retrieval. At EVERY decision point the VLA is ALSO run (no compute
    saved -- diagnostic only, like the oracle_query path / research/retrieval_oracle), and the gate SKIPS --
    executing the N1 top-1 retrieval chunk -- iff that chunk already agrees with the VLA's own fresh chunk:

        rmse = sqrt(mean_over[all 7 dims]( z(retrieval_chunk - vla_chunk[:H]) ** 2 ))   (z = /act_std)
        SKIP iff rmse < tau, else CALL the VLA.

    This is the *oracle* counterpart of research/retrieval_error/rmse_skip_policy.py (rmsegate), which
    PREDICTS this retrieval error from an offline k-NN dictionary; here the true error is measured against
    the live VLA chunk, so the (effective-skip-rate vs success) frontier it traces is the limit a perfect
    retrieval-error gate could reach. RMSE is the per-dim z-scored distance over ALL 7 action dims --
    gripper INCLUDED -- so grasp/release disagreement counts but no single dim's physical scale dominates
    (the per-dim-std normalization is the repo's `norm` metric, cf. OracleRetrievalPolicy / evaluate.py).

    `needs_vla = True` signals closed_loop.run_closed_loop_episode to run the VLA up-front and pass its
    chunk in ctx["vla_chunk"] (reused on a CALL -> no double query). `predictor` is the SAME RetrievalPolicy
    executed on a skip, so the gate scores EXACTLY the chunk that will run (same trick as GateSkipPolicy).
    Same interface as the other gates: decide(ctx)->bool, `self.last` trace dict, `name`.
    """

    name = "oraclermse"
    needs_vla = True

    def __init__(self, predictor, tau: float):
        self.predictor = predictor
        self.tau = float(tau)
        std = getattr(getattr(predictor, "norm", None), "act_std", None)  # per-dim action std (z-score divisor)
        self.act_std = np.ones(7, np.float32) if std is None else np.asarray(std, np.float32)

    def decide(self, ctx: dict) -> bool:
        ret = self.predictor.predict_chunk(ctx["prev_actions"], ctx["current_proprio"],
                                           ctx["cached_future_proprio"], ctx["cached_future_img"])  # (H,7) N1 top-1
        vla = np.asarray(ctx["vla_chunk"], np.float32)[: ret.shape[0]]                               # (H,7) VLA chunk
        err = (ret - vla) / self.act_std                            # per-dim z-scored, all 7 dims (grip incl.)
        rmse = float(np.sqrt((err ** 2).mean()))
        self.last = {"score": rmse, "tau": self.tau}
        return bool(rmse < self.tau)   # SKIP (trust N1 retrieval) iff it agrees with the live VLA chunk


class FusionGateSkipPolicy(SkipPolicy):
    """N1-agreement gate for the FUSION policy, used as a VLA-fallback VETO on top of random skipping.

    This is GateSkipPolicy's "L1 pick within the N1 top-K" agreement test specialised to FusionRetrievalPolicy
    and REUSING that policy's own predictor, cache and key (nothing is rebuilt). Let p = the fusion predictor's
    chunk, topK = the `gate_k` state-nearest cache chunks, and the "L1 pick" jstar = argmin over the WHOLE cache
    of ||chunk - p|| (the predictor's unconstrained best match). Two cases:

      - jstar NOT in topK : the predictor's best match lies OUTSIDE the gate's state-neighbourhood (off-support)
                            -> the gate is TRIPPED -> CALL the VLA (fallback); never skip here.
      - jstar IN topK     : the predictor's best match is state-corroborated -> the gate passes -> SKIP with the
                            caller-given probability `skip_rate` (else CALL).

    So decide() = (jstar in topK) AND (rand < skip_rate): `skip_rate` is the random skip probability AMONG the
    gate-passing (safe) decisions, while the gate only ever REMOVES skips (it vetoes off-support ones), so the
    MEASURED effective skip rate is <= skip_rate.

    `gate_k` (the corroboration neighbourhood) is DECOUPLED from fusion's own selection K (--fusion-k): None/<=0
    reuses fusion.K (the default; gate-pass then means "fusion executes the predictor's global L1 pick"), while
    a LARGER gate_k relaxes the test to "the L1 pick lies within the gate_k state-neighbourhood" -- a looser
    support check that raises gate-pass coverage WITHOUT changing what fusion executes (a bigger gate_k admits
    more decisions as skip-eligible; coverage grows monotonically in gate_k). Composition (not inheritance) over
    the live FusionRetrievalPolicy keeps cache/key/predictor identical to the executed policy and unable to
    drift. Same interface/convention as the other gates: decide(ctx)->bool, `self.last` trace dict, `name`.
    """

    name = "fusiongate"

    def __init__(self, fusion, skip_rate: float, gate_k: int | None = None, seed: int = 0):
        assert 0.0 <= float(skip_rate) <= 1.0, "skip_rate (the random skip probability) must be in [0, 1]"
        assert all(hasattr(fusion, a) for a in ("predictor", "retr", "K")), \
            "fusiongate needs a FusionRetrievalPolicy-like object (predictor + retr + K); use --policy fusion"
        self.fusion = fusion
        self.skip_rate = float(skip_rate)
        # gate_k decouples the corroboration neighbourhood from fusion's selection K; None/<=0 -> reuse fusion.K.
        self.gate_k = int(gate_k) if (gate_k is not None and int(gate_k) > 0) else int(fusion.K)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)
        self._cflat = fusion.retr.values.reshape(fusion.retr.values.shape[0], -1)  # (Nc,112) for the L1 pick

    def reset(self) -> None:
        pass

    def decide(self, ctx: dict) -> bool:
        f = self.fusion
        p = f.predictor.predict_chunk(ctx["prev_actions"], ctx["current_proprio"],
                                      ctx["cached_future_proprio"], ctx["cached_future_img"],
                                      current_image=ctx.get("current_image"),
                                      current_wrist=ctx.get("current_wrist"))             # predictor chunk (16,7)
        q = f.retr._query_key(ctx["prev_actions"], ctx["current_proprio"],
                              ctx["cached_future_proprio"], ctx["cached_future_img"])      # state key (same as fusion)
        dstate = np.linalg.norm(f.retr.keys - q[None, :], axis=1)                         # (Nc,) state-space distance
        K = min(self.gate_k, dstate.shape[0])
        topk = np.argpartition(dstate, K - 1)[:K]                                         # gate_k state-nearest chunks
        jstar = int(np.linalg.norm(self._cflat - p.reshape(-1)[None, :], axis=1).argmin())  # global L1 pick
        in_topk = bool(jstar in set(topk.tolist()))                                       # corroborated by state?
        r = float(self._rng.random())
        self.last = {"in_topk": in_topk, "gate_k": int(K), "score": r, "tau": self.skip_rate}  # skip iff in_topk & r<tau
        return bool(in_topk and r < self.skip_rate)


# Default gate_K grid swept during offline calibration (clamped to the cache size).
_FUSIONGATE_K_GRID = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000]


def calibrate_fusion_gate_k(fusion, data_dir, skip_ceiling: float, k_grid=None,
                            val_frac: float = 0.15, seed: int = 0):
    """Offline-calibrate gate_K for FusionGateSkipPolicy (no VLA / simulator needed).

    coverage(K) = fraction of HELD-OUT (val) decision points where the fusion predictor's global L1 pick
    falls in the K state-nearest cache chunks -- i.e. the gate-PASS rate at gate_K, which is the OFFLINE
    ceiling on the achievable effective skip rate (online effective skip ~= coverage(K) * skip_rate, so the
    max reachable skip at skip_rate->1 is ~coverage(K)). We return the SMALLEST gate_K whose coverage >=
    `skip_ceiling`: smallest-K-meeting-the-floor keeps the gate as SELECTIVE as possible (best success per
    skip) while still letting the --skip-rates sweep reach ~skip_ceiling effective skip. Set skip_ceiling
    > 0.5 to be able to sweep past 50% skip (and sweep --skip-rates up toward 1.0 to realise it).

    Deterministic (fixed val split); uses the SAME cache/key/predictor as the deployed gate, so no second
    model is built. Returns (k_star, coverage_curve{K:cov}, n_val)."""
    from dataset import build_samples, list_success_episodes, split_episode_files

    ss = fusion.retr.state_source
    use_img = bool(getattr(fusion.predictor, "use_obs_emb", False))  # obs-emb predictors need the live frame
    val_files = split_episode_files(list_success_episodes(data_dir), val_frac, seed)[1]
    vs = build_samples(val_files, [], [ss], with_image=use_img)
    cflat = fusion.retr.values.reshape(fusion.retr.values.shape[0], -1)
    N = cflat.shape[0]
    ranks = np.empty(len(vs), dtype=np.int64)
    for i, s in enumerate(vs):
        prev, cur = s.prev_actions.astype(np.float32), s.states[ss].astype(np.float32)
        p = fusion.predictor.predict_chunk(prev, cur, cur, None,
                                           current_image=s.image, current_wrist=s.wrist)
        q = fusion.retr._query_key(prev, cur, cur, None)
        d = np.linalg.norm(fusion.retr.keys - q[None, :], axis=1)
        jstar = int(np.linalg.norm(cflat - p.reshape(-1)[None, :], axis=1).argmin())
        ranks[i] = int((d < d[jstar]).sum()) + 1                  # rank of the L1 pick in the state ranking
    grid = sorted({int(k) for k in (k_grid or _FUSIONGATE_K_GRID) if 1 <= k <= N} | {N})
    cov = {int(k): float((ranks <= k).mean()) for k in grid}
    k_star = next((k for k in grid if cov[k] >= skip_ceiling), N)  # smallest K meeting the coverage floor
    return int(k_star), cov, int(len(vs))


_ENVELOPE_CACHE = {}  # (data_dir, predictor.run_dir, state_source, cache_eps, val_frac, seed) -> envelope dict


def _build_envelope(predictor, data_dir, state_source, cache_episodes, val_frac, seed):
    """Build the success ENVELOPE for EnvelopeGateSkipPolicy (learning-free, no risk head).

    sigma  = per-dim std of the executed actions over the success (demo) episodes -- "how much each action
             dim normally moves" (the guide's sigma-kind=value). cont = continuous control dims, GRIP excluded.
    grip_mid/grip_deadband = the gripper open/close boundary (mean of the demo gripper action) and a deadband:
             0.0 when the gripper actually crosses the boundary in the demos (open+close present), else 0.05
             (a constant/one-sided gripper would otherwise be false-vetoed by the predictor's near-zero noise).
    Then we run the predictor over the HELD-OUT (val) decision points and store the per-decision envelope
    SCORE (max over steps x cont dims of |pred - last_executed| / sigma) with grip-changed decisions set to
    +inf (they always CALL), so a q-quantile threshold => ~q of decisions pass. Cached across q values."""
    from dataset import build_samples, list_success_episodes, split_episode_files

    files = list_success_episodes(data_dir)
    train_files, val_files = split_episode_files(files, val_frac, seed)
    sig_files = files[:cache_episodes] if cache_episodes else train_files
    acts = np.concatenate([np.load(f, allow_pickle=True)["realized_actions"].astype(np.float32)
                           for f in sig_files], axis=0)                  # (N,A) executed demo actions
    sigma = np.maximum(acts.std(0), 1e-6).astype(np.float32)            # (A,) per-dim scale
    cont = np.array([d for d in range(acts.shape[1]) if d != GRIP], dtype=np.int64)  # continuous dims (GRIP out)
    grip_mid = float(acts[:, GRIP].mean())                              # open/close decision boundary
    grip_db = 0.0 if (acts[:, GRIP].max() - acts[:, GRIP].min()) > 0.1 else 0.05  # deadband iff ~constant gripper

    ss = getattr(predictor, "state_source", state_source)
    use_img = bool(getattr(getattr(predictor, "predictor", predictor), "use_obs_emb", False))
    vs = build_samples(val_files, [], [ss], with_image=use_img)
    scores = np.empty(len(vs), np.float32)
    for i, s in enumerate(vs):
        prev = s.prev_actions.astype(np.float32)
        st = s.states[ss].astype(np.float32)
        chunk = predictor.predict_chunk(prev, st, st, None, current_image=s.image, current_wrist=s.wrist)
        last = prev[-1]
        score = float((np.abs(chunk - last[None, :])[:, cont] / sigma[cont]).max())
        gb = grip_mid + grip_db
        grip_changed = bool(((chunk[:, GRIP] > gb) != (last[GRIP] > gb)).any())
        scores[i] = np.inf if grip_changed else score                  # grip-changed -> never passes
    return {"sigma": sigma, "cont": cont, "grip_mid": grip_mid, "grip_db": grip_db,
            "scores_eff": scores, "n_val": int(len(vs)), "max_skip_frac": float(np.isfinite(scores).mean())}


class EnvelopeGateSkipPolicy(SkipPolicy):
    """Success-ENVELOPE gate (learning-free, no risk head, no probabilities): SKIP -- adopt the predictor's
    chunk -- only when that chunk stays INSIDE the demos' success envelope. Two independent conditions, both
    required (the guide's wrapper._envelope_score + gripper veto):

        score = max over the K predicted steps x continuous dims of |pred - last_executed| / sigma   (<= tau)
        grip_changed = the gripper open/close decision flips anywhere in the chunk                    (must be False)

    i.e. every continuous dim of every predicted step must stay within `tau`*sigma of the last EXECUTED action
    (the residual-from-last structure makes sigma the natural scale), and the safety-critical gripper switch is
    NEVER left to the predictor. `max` (not mean) + the gripper veto are what make it conservative. The decision
    materials are only the predictor output and the demo per-dim sigma -- no model, nothing VLA-specific.

    tau is calibrated to the q-quantile of the held-out (val) envelope scores (grip-changed = +inf), so q ~= the
    in-distribution skip fraction (skip the lowest-score fraction q); the closed-loop EFFECTIVE skip rate is
    measured (typically < q under covariate shift -- the guide's pass-2 online recalibration; here, as for the
    other gates, we report the measured rate instead). Structural gates from the guide (warmup / cooldown /
    max-consecutive) map onto the runner: `can_skip` (no skip until history+cache exist) and `--max-skips` (the
    consecutive-skip drift budget). `predictor` is the SAME policy executed on a skip, so the gate scores EXACTLY
    the chunk that will run. Same interface as the other gates: decide(ctx)->bool, `self.last` trace dict, `name`.
    """

    name = "envelope"

    def __init__(self, predictor, data_dir, q, state_source="actual_next_proprio",
                 cache_episodes=0, val_frac: float = 0.15, seed: int = 0):
        ck = (os.path.abspath(data_dir), getattr(predictor, "run_dir", id(predictor)),
              state_source, int(cache_episodes or 0), val_frac, seed)
        if ck not in _ENVELOPE_CACHE:
            _ENVELOPE_CACHE[ck] = _build_envelope(predictor, data_dir, state_source, cache_episodes, val_frac, seed)
        c = _ENVELOPE_CACHE[ck]
        self.predictor = predictor
        self.sigma, self.cont = c["sigma"], c["cont"]
        self.grip_mid, self.grip_db = c["grip_mid"], c["grip_db"]
        self.q, self.max_skip_frac = float(q), c["max_skip_frac"]
        self.tau = float(np.quantile(c["scores_eff"], self.q))  # skip the lowest-score fraction q (grip-changed = inf)

    def reset(self) -> None:
        pass

    def decide(self, ctx: dict) -> bool:
        chunk = self.predictor.predict_chunk(ctx["prev_actions"], ctx["current_proprio"],
                                             ctx["cached_future_proprio"], ctx["cached_future_img"],
                                             current_image=ctx.get("current_image"),
                                             current_wrist=ctx.get("current_wrist"))   # (K,A) the chunk a skip runs
        last = np.asarray(ctx["prev_actions"], np.float32)[-1]                          # last EXECUTED raw action
        score = float((np.abs(chunk - last[None, :])[:, self.cont] / self.sigma[self.cont]).max())
        gb = self.grip_mid + self.grip_db
        grip_changed = bool(((chunk[:, GRIP] > gb) != (last[GRIP] > gb)).any())
        self.last = {"score": score, "tau": self.tau, "grip_changed": grip_changed, "q": self.q}
        return bool(score <= self.tau and not grip_changed)  # inside the envelope AND no gripper switch


def make_skip_policy(name: str, skip_rate: float, seed: int = 0) -> SkipPolicy:
    if name == "random":
        return RandomSkipPolicy(skip_rate, seed=seed)
    if name == "even":
        return EvenSkipPolicy(skip_rate)
    raise ValueError(f"unknown skip policy {name!r} (available: random, even)")
