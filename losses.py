"""Eqs. 15–24 and experimental route-C objectives.

Exact Eq.19 uses a product of smooth signs and a binomial denominator.
Minimizing its signed result can reward inversions; the absolute/hinged
alternatives below are experimental, not claimed to reproduce the paper.
The PDF gives the two loss weights (.3, 1.) but not the metric/mining policy,
margin or temperature. Undisclosed defaults are reconstruction assumptions.
"""
import torch
import torch.nn.functional as F


def batch_hard_triplet(features, labels, margin=.3):
    """Eq.17 with Euclidean batch-hard mining (an explicit assumption)."""
    d = torch.cdist(features.float(), features.float())
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
    negative = ~same
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any():
        return d.sum() * 0
    dp = d.masked_fill(~positive, -torch.inf).amax(dim=1)
    dn = d.masked_fill(~negative, torch.inf).amin(dim=1)
    return (dp[valid] - dn[valid] + margin).clamp_min(0).mean()


def differentiable_kendall(x, y, tau=.1):
    """Eq.19 elementwise over leading dimensions; last dim is ranking."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if x.shape != y.shape:
        raise ValueError("ranking inputs must have identical shapes")
    n = x.shape[-1]
    if n < 2:
        return (x + y).sum(dim=-1) * 0
    a, b = torch.triu_indices(n, n, offset=1, device=x.device)
    return (((x[..., a] - x[..., b]) / tau).tanh()
            * ((y[..., a] - y[..., b]) / tau).tanh()).mean(dim=-1)


def _ranking_components(g, f, labels, tau, transform):
    if g.shape[0] != f.shape[0] or g.shape[0] != labels.numel():
        raise ValueError("branches and labels must share a batch")
    g, f = g.float(), f.float()
    gn, fn = F.normalize(g, dim=1), F.normalize(f, dim=1)
    idx = torch.unique(labels, sorted=True, return_inverse=True)[1]
    ids = idx.unique(sorted=True)
    member = idx[:, None].eq(ids[None, :])
    # Eq.21: normalize the mean of raw embeddings, not the mean of unit embeddings.
    m = member.to(g.dtype)
    counts = m.sum(0)
    gc = F.normalize((m.T @ g) / counts[:, None], dim=1)
    fc = F.normalize((m.T @ f) / counts[:, None], dim=1)
    pg, pf = gn @ gn.T, fn @ fn.T
    ng, nf = gn @ gc.T, fn @ fc.T
    pos = idx[:, None].eq(idx[None, :]) & ~torch.eye(
        idx.numel(), device=idx.device, dtype=torch.bool)
    neg = ~member

    def reduce_rows(x, y, mask):
        # Balanced P×K is the paper setting. For unbalanced batches, group
        # anchors by comparison count and average only eligible anchors.
        terms = []
        lengths = mask.sum(-1)
        for n in lengths.unique().tolist():
            if n < 2:
                continue
            r = lengths.eq(n)
            terms.append(transform(differentiable_kendall(
                x[r][mask[r]].reshape(-1, n), y[r][mask[r]].reshape(-1, n), tau)))
        return torch.cat(terms).mean() if terms else (x.sum() + y.sum()) * 0

    return reduce_rows(pg, pf, pos), reduce_rows(ng, nf, neg)


def ranking_regularizer(g, f, labels, tau=.1, mode="signed", rank_target=.2):
    """Eqs.19–23, or route-C per-anchor absolute / upper-bound variants."""
    if mode == "signed":
        transform = lambda x: x
    elif mode == "absolute":
        transform = lambda x: x.abs()
    elif mode == "hinge":
        transform = lambda x: (x - rank_target).clamp_min(0)
    else:
        raise ValueError(f"unknown Kendall mode {mode!r}")
    return _ranking_components(g, f, labels, tau, transform)


def cross_covariance(g, f):
    """Experimental squared cross-covariance with per-feature normalization."""
    g, f = g.float(), f.float()
    if g.shape[0] < 2:
        return (g.sum() + f.sum()) * 0
    g, f = g - g.mean(dim=0), f - f.mean(dim=0)
    g = g / (g.square().mean(dim=0) + 1e-6).sqrt()
    f = f / (f.square().mean(dim=0) + 1e-6).sqrt()
    return ((g.T @ f) / g.shape[0]).square().mean()


def identity_consistency(logits_g, logits_f):
    """Experimental symmetric JS divergence on main identity probabilities."""
    a, b = logits_g.float().log_softmax(-1), logits_f.float().log_softmax(-1)
    mix = torch.logaddexp(a, b) - 0.6931471805599453
    return .5 * ((a.exp() * (a - mix)).sum(-1) + (b.exp() * (b - mix)).sum(-1)).mean()


def objective(outputs, labels, variant="ratr", tau=.1, margin=.3, aux_weight=.3,
              ratr_weight=1., cov_weight=.1, consistency_weight=.1, rank_target=.2):
    """Each branch is (pre-BN embedding, logits); aux is a sequence of branches.

    'ratr': paper's signed loss; 'base'/'none': Eq.18 only.
    'ratr_abs' and 'ratr_hinge': route C alternatives to signed rank.
    'cov', 'consistency', and 'cov_consistency': route C alternatives.
    'ratr_cov' and 'ratr_cov_consistency': optional controlled additions.
    """
    aliases = {"no_ratr": "base", "baseline": "base", "absolute": "ratr_abs",
               "hinge": "ratr_hinge", "route_c": "cov_consistency", "cov-only": "cov"}
    variant = aliases.get(variant, variant)
    modes = {"base", "none", "ratr", "ratr_abs", "ratr_hinge", "cov",
             "consistency", "cov_consistency", "ratr_cov", "ratr_cov_consistency"}
    if variant not in modes:
        raise ValueError(f"unknown objective variant {variant!r}; choose from {sorted(modes)}")
    labels = labels.reshape(-1).long()
    g, g_logits = outputs["backbone"]
    f, f_logits = outputs["fusion"]

    def branch_loss(embedding, logits):
        return (F.cross_entropy(logits.float(), labels, label_smoothing=.1),
                batch_hard_triplet(embedding, labels, margin))

    gid, gtri = branch_loss(g, g_logits)
    fid, ftri = branch_loss(f, f_logits)
    id_loss, triplet = gid + fid, gtri + ftri
    zero = id_loss * 0
    aux = sum((sum(branch_loss(*branch)) for branch in outputs.get("aux", ())), zero)
    intra = inter = cov = consistency = zero
    if variant.startswith("ratr") and ratr_weight != 0:
        mode = {"ratr_abs": "absolute", "ratr_hinge": "hinge"}.get(variant, "signed")
        intra, inter = ranking_regularizer(g, f, labels, tau, mode, rank_target)
    if "cov" in variant and cov_weight != 0:
        cov = cross_covariance(g, f)
    if "consistency" in variant and consistency_weight != 0:
        consistency = identity_consistency(g_logits, f_logits)
    base = id_loss + triplet + aux_weight * aux
    total = base + ratr_weight * (intra + inter) + cov_weight * cov + consistency_weight * consistency
    return dict(loss=total, base=base, id=id_loss, triplet=triplet, aux=aux,
                ratr=intra + inter, ratr_intra=intra, ratr_inter=inter,
                cov=cov, consistency=consistency)
