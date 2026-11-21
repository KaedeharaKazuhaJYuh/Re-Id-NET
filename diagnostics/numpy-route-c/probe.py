"""Synthetic experiment for route C. No images or real identity data."""
import argparse
import json
from pathlib import Path

import numpy as np

MODES = ("none", "ratr_proxy", "cov", "cov_id")
WEIGHTS = {"rank": .25, "cov": .50, "id": .25, "triplet": .25}


def unit(x):
    norm = np.sqrt((x * x).sum(axis=1, keepdims=True) + 1e-12)
    return x / norm, norm


def unit_grad(z, norm, g):
    return (g - z * (g * z).sum(axis=1, keepdims=True)) / norm


def ranking(z, y, refs=None):
    """Per-query ranking score differences and index pairs, never mix pos/neg."""
    n = len(z)
    classes = np.unique(y)
    masks = (classes[:, None] == y[None, :]).astype(float)
    masks /= masks.sum(axis=1, keepdims=True)
    basis = np.concatenate((np.eye(n), masks))
    if refs is None:
        queries, left, right, groups = [], [], [], []
        for i in range(n):
            pos = np.flatnonzero((y == y[i]) & (np.arange(n) != i))
            neg = n + np.flatnonzero(classes != y[i])
            for group, idx in enumerate((pos, neg)):
                a, b = np.triu_indices(len(idx), 1)
                queries.extend([i] * len(a))
                left.extend(idx[a])
                right.extend(idx[b])
                groups.extend([group] * len(a))
        refs = tuple(np.asarray(v, dtype=int) for v in (queries, left, right, groups))
    q, a, b, _ = refs
    coords = basis @ z
    dif = coords[a] - coords[b]
    return (z[q] * dif).sum(1), (refs, basis, dif)


def rank_grad(z, cache, grad):
    (q, a, b, _), basis, dif = cache
    out = np.zeros_like(z)
    np.add.at(out, q, grad[:, None] * dif)
    v = np.zeros((len(basis), z.shape[1]), dtype=z.dtype)
    np.add.at(v, a, grad[:, None] * z[q])
    np.add.at(v, b, -grad[:, None] * z[q])
    return out + basis.T @ v


def rank_penalty(za, zb, y, tau=.15):
    a, ca = ranking(za, y)
    b, cb = ranking(zb, y, ca[0])
    sa, sb = np.tanh(a / tau), np.tanh(b / tau)
    loss = 0.
    da, db = np.zeros_like(a), np.zeros_like(b)
    for group in (0, 1):
        idx = ca[0][-1] == group
        count = idx.sum()
        if count:
            loss += .5 * np.mean(sa[idx] * sb[idx])
            da[idx] = .5 * sb[idx] * (1 - sa[idx]**2) / (tau * count)
            db[idx] = .5 * sa[idx] * (1 - sb[idx]**2) / (tau * count)
    return loss, rank_grad(za, ca, da), rank_grad(zb, cb, db)


def cov_penalty(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    n = len(a) - 1
    c = a.T @ b / n
    g = 2 * c / (c.size * n)
    ga, gb = b @ g.T, a @ g
    return (c**2).mean(), ga - ga.mean(0), gb - gb.mean(0)


def cross_entropy(logits, y):
    scores = logits - logits.max(1, keepdims=True)
    e = np.exp(scores)
    p = e / e.sum(1, keepdims=True)
    rows = np.arange(len(y))
    loss = np.mean(np.log(e.sum(1)) - scores[rows, y])
    p[rows, y] -= 1
    return loss, p / len(y)


def triplet(z, y, margin=.25):
    diff = z[:, None, :] - z[None, :, :]
    dist = (diff**2).sum(2)
    same = y[:, None] == y[None, :]
    pos = same & ~np.eye(len(y), dtype=bool)
    j = np.where(pos, dist, -np.inf).argmax(1)
    k = np.where(~same, dist, np.inf).argmin(1)
    rows = np.arange(len(y))
    v = margin + dist[rows, j] - dist[rows, k]
    a = rows[v > 0]
    p = 2 * diff[a, j[a]] / len(y)
    n = 2 * diff[a, k[a]] / len(y)
    g = np.zeros_like(z)
    np.add.at(g, a, p - n)
    np.add.at(g, j[a], -p)
    np.add.at(g, k[a], n)
    return np.maximum(v, 0).mean(), g


def synth(seed):
    rng = np.random.default_rng(seed)
    mats = [rng.normal(size=(8, 12)) / np.sqrt(8) for _ in range(2)]
    bias = [rng.normal(0, .5, (2, 12)) for _ in range(2)]

    def split(ids, offset):
        lat = rng.normal(size=(ids, 8))
        y = np.repeat(np.arange(ids), 8)
        camera = np.tile(np.repeat([0, 1], 4), ids)
        lat = lat[y]
        x1 = lat.copy()
        x2 = lat.copy()
        x1[:, 6:] = 0
        x2[:, :2] = 0
        x = tuple(l @ m + b[camera] + rng.normal(0, .9, (len(y), 12))
                  for l, m, b in zip((x1, x2), mats, bias))
        return x, y + offset, camera

    return split(24, 0), split(12, 24), split(12, 36)


def train(data, mode, seed, steps):
    rng = np.random.default_rng(seed)
    (x1, x2), y, _ = data
    classes = len(np.unique(y))
    w1, w2 = (rng.normal(0, .30, (12, 8)) for _ in range(2))
    h1, h2 = (rng.normal(0, .22, (8, classes)) for _ in range(2))
    for step in range(steps):
        ids = rng.choice(classes, 6, replace=False)
        batch = np.concatenate([rng.choice(np.flatnonzero(y == c), 4, replace=False) for c in ids])
        a, b, yy = x1[batch], x2[batch], y[batch]
        za, na = unit(a @ w1)
        zb, nb = unit(b @ w2)
        logits_a, logits_b = 4 * (za @ h1), 4 * (zb @ h2)
        _, la = cross_entropy(logits_a, yy)
        _, lb = cross_entropy(logits_b, yy)
        if mode == "cov_id":
            d = logits_a - logits_b
            d -= d.mean(1, keepdims=True)
            gg = WEIGHTS["id"] * 2 * d / d.size
            la += gg
            lb -= gg
        ga, gb = 4 * (la @ h1.T), 4 * (lb @ h2.T)
        for z, g in ((za, ga), (zb, gb)):
            _, dt = triplet(z, yy)
            g += WEIGHTS["triplet"] * dt
        if mode != "none":
            if mode == "ratr_proxy":
                _, da, db = rank_penalty(za, zb, yy)
                weight = WEIGHTS["rank"]
            else:
                _, da, db = cov_penalty(za, zb)
                weight = WEIGHTS["cov"]
            ga += weight * da
            gb += weight * db
        lr = .23 * (1 - .7 * step / max(1, steps - 1))
        w1 -= lr * (a.T @ unit_grad(za, na, ga))
        w2 -= lr * (b.T @ unit_grad(zb, nb, gb))
        h1 -= lr * (4 * za.T @ la)
        h2 -= lr * (4 * zb.T @ lb)
    return w1, w2


def retrieval(scores, qy, gy):
    order = np.argsort(-scores, axis=1, kind="stable")
    hits = gy[order] == qy[:, None]
    found = hits.sum(1)
    if not np.all(found):
        raise ValueError("Every query must have a gallery positive")
    precision = hits.cumsum(1) / np.arange(1, hits.shape[1] + 1)
    return {"mAP": float(((precision * hits).sum(1) / found).mean()),
            "rank1": float(hits[:, 0].mean())}


def evaluate(data, weights):
    x, y, camera = data
    a, b = [unit(v @ w)[0] for v, w in zip(x, weights)]
    q, g = camera == 0, camera == 1
    sa, sb = a[q] @ a[g].T, b[q] @ b[g].T
    scores = {k: retrieval(s, y[q], y[g])
              for k, s in (("branch_a", sa), ("branch_b", sb), ("fusion", (sa+sb)/2))}
    d1, c1 = ranking(a[g], y[g])
    d2, _ = ranking(b[g], y[g], c1[0])
    diag = {}
    for idx, name in enumerate(("positive", "negative_centroid")):
        mask = c1[0][-1] == idx
        signs = np.sign(d1[mask]) * np.sign(d2[mask])
        diag[name] = {"tau_a": float(signs.mean()), "discordant_fraction": float((signs < 0).mean())}
    return {**scores, "within_gallery_ranking_diagnostic": diag,
            "within_gallery_cross_covariance_mse": float(cov_penalty(a[g], b[g])[0]),
            "fusion_minus_best_single_map": scores["fusion"]["mAP"]
            - max(scores["branch_a"]["mAP"], scores["branch_b"]["mAP"])}


def summarize(out):
    if isinstance(out[0], dict):
        return {k: summarize([v[k] for v in out]) for k in out[0]}
    return {"mean": float(np.mean(out)),
            "sd": float(np.std(out, ddof=1)) if len(out) > 1 else None}


def run(seeds, steps):
    rows = {}
    for seed in seeds:
        tr, va, te = synth(seed)
        rows[str(seed)] = {}
        for mode in MODES:
            w = train(tr, mode, seed + 100, steps)
            rows[str(seed)][mode] = {"validation": evaluate(va, w), "test": evaluate(te, w)}
    return {"synthetic": True, "seeds": seeds, "steps": steps, "P": 6, "K": 4,
            "weights": WEIGHTS, "per_seed": rows,
            "aggregate": {m: summarize([row[m] for row in rows.values()]) for m in MODES}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21])
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    if args.steps < 1 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide positive steps and distinct non-negative seeds.")
    result = run(args.seeds, args.steps)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Synthetic test only: fusion mAP / Rank-1 (%, mean across seeds)")
    for m, stats in result["aggregate"].items():
        f = stats["test"]["fusion"]
        print(f"{m:12s} {100*f['mAP']['mean']:6.2f} / {100*f['rank1']['mean']:6.2f}")
