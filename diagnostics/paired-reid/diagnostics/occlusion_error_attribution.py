"""Aggregate, evaluation-only decomposition of paired clean/masked retrieval errors."""
from collections import Counter
import numpy as np

BRANCHES = ("backbone", "fusion")
MODES = BRANCHES + ("equal",)
BUCKETS = ("branch_still_rank1", "branch_positive_rank2_to_k",
           "both_branches_below_k_clean_had_candidate",
           "both_branches_below_k_clean_had_no_candidate")

def decompose(clean, masked, query_ids, query_cams, gallery_ids, gallery_cams, *, k=5):
    """Aligned scores only; output aggregate counts, never candidate or query IDs.

    Restrict the primary denominator to clean-equal hits that masked equal misses.
    The buckets describe evidence in observed ranks, not causal attribution.
    Scorer filters apply only to assessment, after sorting the full gallery.
    """
    qid, qc, gid, gc = map(np.asarray, (query_ids, query_cams, gallery_ids, gallery_cams))
    if isinstance(k, bool) or not isinstance(k, (int,np.integer)) or k < 2 or any(a.ndim != 1 for a in (qid,qc,gid,gc)) or len(qid)!=len(qc) or len(gid)!=len(gc):
        raise ValueError("Invalid k or evaluation metadata")
    shape = (len(qid),len(gid))
    phases = [dict((m,np.asarray(group[m])) for m in MODES) for group in (clean,masked)]
    if any(x.shape!=shape or not np.isfinite(x).all() for group in phases for x in group.values()):
        raise ValueError("Mismatched, malformed, or nonfinite scores")
    counts=Counter()
    buckets=Counter(dict.fromkeys(BUCKETS,0))
    transitions={key:Counter() for key in MODES}
    margins={bucket:{m:[] for m in MODES} for bucket in BUCKETS}
    for i,(ident,cam) in enumerate(zip(qid,qc)):
        if ident == -1:
            continue
        valid = (gid!=-1) & ~((gid==ident)&(gc==cam))
        matches = gid==ident
        if not (valid&matches).any():
            continue
        def first_hit(row):
            order=np.argsort(-row,kind="stable")
            hits=matches[order[valid[order]]]
            return int(np.flatnonzero(hits)[0])+1
        before,after=({key:first_hit(x[i]) for key,x in group.items()} for group in phases)
        counts["valid_queries"]+=1
        for key in MODES:
            a,b=before[key],after[key]
            transitions[key][f"rank1_{int(a==1)}_to_{int(b==1)}"]+=1
            transitions[key][f"topk_{int(a<=k)}_to_{int(b<=k)}"]+=1
        if before["equal"] != 1 or after["equal"] == 1:
            continue
        counts["new_equal_rank1_misses"]+=1
        best_clean=min(before[key] for key in BRANCHES)
        best_masked=min(after[key] for key in BRANCHES)
        if best_masked == 1:
            bucket=BUCKETS[0]
        elif best_masked <= k:
            bucket=BUCKETS[1]
        elif best_clean <= k:
            bucket=BUCKETS[2]
        else:
            bucket=BUCKETS[3]
        buckets[bucket]+=1
        negatives = valid & ~matches
        positives = valid & matches
        if negatives.any():
            for key in MODES:
                paired = [float(g[key][i,positives].max()) - float(g[key][i,negatives].max()) for g in phases]
                margins[bucket][key].append(paired)
    if not counts["valid_queries"]:
        raise ValueError("No queries with a valid cross-camera positive")
    if sum(buckets.values())!=counts["new_equal_rank1_misses"]:
        raise RuntimeError("Attribution bucket accounting mismatch")
    def margin_summary(pairs):
        if not pairs:
            return dict(n=0,clean_mean=None,masked_mean=None,delta_mean=None)
        a,b=np.asarray(pairs,dtype=np.float64).T
        return dict(n=len(pairs),clean_mean=float(a.mean()),masked_mean=float(b.mean()),delta_mean=float((b-a).mean()))
    return dict(k=int(k),valid_queries=counts["valid_queries"],new_equal_rank1_misses=counts["new_equal_rank1_misses"],
                buckets=dict(buckets),mean_positive_minus_negative={bucket:{key:margin_summary(x) for key,x in v.items()} for bucket,v in margins.items()},transitions={key:{f"{scope}_{a}_to_{b}":v[f"{scope}_{a}_to_{b}"] for scope in ("rank1","topk") for a in (0,1) for b in (0,1)} for key,v in transitions.items()})
