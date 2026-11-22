"""Frozen aggregate diagnostics: clean/masked branch substitutions and candidate coverage."""
from collections import Counter
import numpy as np
from diagnostics.occlusion_error_attribution import BUCKETS, decompose

BRANCHES = ("backbone", "fusion")
CUTS = (5, 10, 20)
CC, CM, MC, MM = ("clean_both", "clean_backbone_masked_fusion",
                  "masked_backbone_clean_fusion", "masked_both")
FACTORS = (CC, CM, MC, MM)
MISS_BUCKETS = (
    "filtered_branch_rank1_survived",
    "raw_shortlist_valid_positive_survived",
    "raw_shortlist_lost_valid_positive",
    "raw_shortlist_no_valid_positive_in_either",
)


def _finite_scores(values, shape):
    x = np.asarray(values)
    if (x.shape != shape or x.dtype.kind not in "fiu" or
            not np.all(np.isfinite(x))):
        raise ValueError("Expected finite, aligned Q-by-G cosine scores")
    return x.astype(np.float64, copy=False)


def _order(row):
    return np.argsort(-row, kind="stable")


def _union(left_order, right_order, k):
    """Select from the actual full gallery, before evaluator-only exclusions."""
    return np.union1d(left_order[:k], right_order[:k])


def _outcome(order, pos, valid):
    hits = pos[order[valid[order]]]
    ranks = np.flatnonzero(hits) + 1
    return float(hits[0]), float(np.mean(np.arange(1, len(ranks)+1)/ranks))


def _counts(mask):
    return int(np.count_nonzero(mask))


def _score_summary(values):
    return dict(n=len(values), mean=float(np.mean(values)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None)


def _stats(matrix):
    return dict(rank1=float(matrix[:, 0].mean()), mAP=float(matrix[:, 1].mean()))


def _contrasts(base, restored):
    delta = restored - base
    return dict(rank1=float(delta[:, 0].mean()), mAP=float(delta[:, 1].mean()),
                rescues=_counts(delta[:, 0] > 0), harms=_counts(delta[:, 0] < 0))


def analyze(clean, masked, query_ids, query_cams, gallery_ids, gallery_cams,
            *, mixed, condition):
    """Return aggregate results only, using exact native ranking matrices.

    Inputs are from one fixed checkpoint, one clean gallery, and identical
    ordered queries (clean and the named masked variant). Caller must verify
    row keys/checkpoint/native archive before invoking. Native branch vectors
    must be nonzero so exact normalized concat agrees algebraically with the
    two branch cosines averaged. Fail closed on missing/malformed inputs.
    """
    if condition not in ("heavy", "light"):
        raise ValueError("Expected heavy (primary) or light (descriptive)")
    qi, qc, gi, gc = [np.asarray(v) for v in
                      (query_ids, query_cams, gallery_ids, gallery_cams)]
    if (any(v.ndim != 1 for v in (qi, qc, gi, gc)) or len(qi) != len(qc)
            or len(gi) != len(gc) or not len(qi) or not len(gi)):
        raise ValueError("Expected nonempty aligned identity and camera vectors")
    shape = (len(qi), len(gi))
    clean, masked = [{k:_finite_scores(group[k],shape) for k in (*BRANCHES,"equal")}
                     for group in (clean, masked)]
    if set(mixed) != {CM, MC}:
        raise ValueError("Both exact crossed native score matrices are required")
    factors = {CC: clean["equal"], CM: _finite_scores(mixed[CM],shape),
               MC: _finite_scores(mixed[MC],shape), MM: masked["equal"]}
    bounds = {CC: (clean, clean), CM: (clean, masked),
              MC: (masked, clean), MM: (masked, masked)}
    error = {}
    for key, arr in factors.items():
        bb, fu = bounds[key]
        error[key] = float(np.abs(arr - .5*(bb["backbone"]+fu["fusion"])).max())
    if any(e > 2e-6 for e in error.values()):
        raise ValueError("Native normalized-concatenation score parity failed")

    observations = {key: [] for key in FACTORS}
    branch_observations = {phase: {k: [] for k in BRANCHES} for phase in ("clean","masked")}
    factorial_margins = {key: [] for key in FACTORS}
    branch_margins = {phase: {k: [] for k in BRANCHES} for phase in ("clean","masked")}
    # For each k collect raw candidate generation followed by evaluator checks;
    # don't refill discarded identities/cameras or report per-query rows.
    transitions = {k: Counter(dict.fromkeys(("0_to_0","0_to_1","1_to_0","1_to_1"),0)) for k in CUTS}
    buckets = {k: Counter(dict.fromkeys(MISS_BUCKETS,0)) for k in CUTS}
    sizes = {k: {phase: [] for phase in ("clean","masked")} for k in CUTS}
    raw_trigger = {k: {p: Counter() for p in ("clean","masked")} for k in CUTS}
    raw_by_bucket = {k: {b: Counter() for b in BUCKETS} for k in CUTS}
    margin_rows_by_bucket = {b: [] for b in BUCKETS}
    scored = 0
    for idx,(ident, camera) in enumerate(zip(qi,qc)):
        if ident == -1:
            continue
        positive = gi == ident
        valid = (gi != -1) & ~(positive & (gc == camera))
        if not np.any(positive & valid):
            continue
        scored += 1
        negative = ~positive & valid
        pos = positive & valid
        has_neg = np.any(negative)
        def margin(row):
            return float(row[pos].max()-row[negative].max()) if has_neg else None
        orders = {}
        for phase, pair in (("clean",clean),("masked",masked)):
            orders[phase] = {key:_order(pair[key][idx]) for key in BRANCHES}
            for key in BRANCHES:
                branch_observations[phase][key].append(_outcome(orders[phase][key],positive,valid))
                branch_margins[phase][key].append(margin(pair[key][idx]))
        hits = {}
        for key, mat in factors.items():
            item = _outcome(_order(mat[idx]),positive,valid)
            hits[key] = bool(item[0])
            observations[key].append(item)
            factorial_margins[key].append(margin(mat[idx]))
        def best_rank(phase):
            return min(int(np.flatnonzero(positive[o[valid[o]]])[0])+1
                       for o in orders[phase].values())
        best_c, best_m = best_rank("clean"), best_rank("masked")
        new_miss = hits[CC] and not hits[MM]
        disagreement = {p: rows["backbone"][0] != rows["fusion"][0]
                        for p, rows in orders.items()}
        for k in CUTS:
            found = {}
            for phase, branch_orders in orders.items():
                union = _union(branch_orders["backbone"], branch_orders["fusion"], k)
                found[phase] = np.any(pos[union])
                sizes[k][phase].append((len(union), _counts(valid[union])))
                counts = raw_trigger[k][phase]
                counts["raw_top1_disagreements"] += int(disagreement[phase])
                counts["raw_top1_disagreements_with_valid_positive_in_union"] += int(disagreement[phase] and found[phase])
                correct = hits[CC if phase == "clean" else MM]
                counts["raw_top1_disagreements_equal_correct"] += int(disagreement[phase] and correct)
                counts["raw_top1_disagreements_equal_correct_with_valid_positive_in_union"] += int(
                    disagreement[phase] and correct and found[phase])
            c, m = found["clean"], found["masked"]
            transitions[k][f"{int(c)}_to_{int(m)}"] += 1
            if new_miss:
                filtered_bucket = BUCKETS[0 if best_m == 1 else 1 if best_m <= k else 2 if best_c <= k else 3]
                cross = raw_by_bucket[k][filtered_bucket]
                cross["queries"] += 1
                cross["clean_raw_union_has_valid_positive"] += int(c)
                cross["masked_raw_union_has_valid_positive"] += int(m)
                cross["masked_raw_top1_disagreement"] += int(disagreement["masked"])
                cross["masked_raw_top1_disagreement_and_positive_in_union"] += int(disagreement["masked"] and m)
                if k == CUTS[0]:
                    margin_rows_by_bucket[filtered_bucket].append(scored - 1)
                masked_branch_first = any(branch_observations["masked"][key][-1][0]
                                          for key in BRANCHES)
                # A valid positive can rank first after evaluation filtering but
                # fall outside both operational raw lists. The first bucket
                # deliberately takes precedence; candidate transition is still
                # reported independently on this same denominator.
                bucket = MISS_BUCKETS[0 if masked_branch_first else 1 if m else 2 if c else 3]
                buckets[k][bucket] += 1
                transitions[k][f"new_misses_{int(c)}_to_{int(m)}"] += 1
    if not scored:
        raise ValueError("No queries with a valid cross-camera positive")

    outcomes = {key:np.asarray(v) for key,v in observations.items()}
    newly_wrong = (outcomes[CC][:,0] == 1) & (outcomes[MM][:,0] == 0)
    n_new = _counts(newly_wrong)
    all_restore = {k:_contrasts(outcomes[MM],outcomes[k]) for k in (CM, MC, CC)}
    recovery = {k:_counts(newly_wrong & (outcomes[k][:,0] == 1)) for k in (CM,MC)}
    cm, mc = outcomes[CM][:,0] == 1, outcomes[MC][:,0] == 1
    recovered_partition = {
        "only_restore_clean_backbone":_counts(newly_wrong & cm & ~mc),
        "only_restore_clean_fusion":_counts(newly_wrong & ~cm & mc),
        "both_single_restorations_work":_counts(newly_wrong & cm & mc),
        "neither_single_restoration_works":_counts(newly_wrong & ~cm & ~mc),
    }
    interaction = (outcomes[CC] + outcomes[MM] - outcomes[CM] - outcomes[MC]).mean(0)
    branch_stats = {}
    for phase in ("clean","masked"):
        branch_stats[phase] = {key:_stats(np.asarray(arr)) for key,arr in branch_observations[phase].items()}
    def margins_summary(values):
        return dict(all=_score_summary([v for v in values if v is not None]),
                    new_equal_misses=_score_summary([v for v, flag in zip(values,newly_wrong)
                                                    if flag and v is not None]))
    branch_margin_stats = {phase:{key:margins_summary(v) for key,v in group.items()}
                           for phase,group in branch_margins.items()}
    margin_paired = {}
    for key in BRANCHES:
        margin_paired[key] = margins_summary([
            None if a is None or b is None else b-a
            for a,b in zip(branch_margins["clean"][key],branch_margins["masked"][key])])
    margin_paired["equal"] = margins_summary([
        None if a is None or b is None else b-a
        for a,b in zip(factorial_margins[CC],factorial_margins[MM])])

    def transform_margins(fn, selection=range(scored)):
        return _score_summary([float(fn(*(factorial_margins[key][i] for key in FACTORS)))
                               for i in selection
                               if all(factorial_margins[key][i] is not None for key in FACTORS)])

    margin_ops = {"restore_clean_backbone":lambda cc,cm,mc,mm: cm-mm,
                  "restore_clean_fusion":lambda cc,cm,mc,mm: mc-mm,
                  "restore_both":lambda cc,cm,mc,mm: cc-mm,
                  "descriptive_interaction":lambda cc,cm,mc,mm: cc-cm-mc+mm}
    partition_margin = {}
    for bucket, rows in margin_rows_by_bucket.items():
        partition_margin[bucket] = dict(
            count=len(rows), missing_valid_negative=len(rows)-len(
                [i for i in rows if factorial_margins[CC][i] is not None]),
            variants={key:_score_summary([v[i] for i in rows if v[i] is not None])
                      for key,v in factorial_margins.items()},
            effects={key:transform_margins(fn,rows) for key,fn in margin_ops.items()})

    shortlist = {}
    for k in CUTS:
        all_trans = {key:transitions[k][key] for key in ("0_to_0","0_to_1","1_to_0","1_to_1")}
        miss_trans = {key:transitions[k][f"new_misses_{key}"] for key in all_trans}
        if sum(all_trans.values()) != scored or sum(miss_trans.values()) != n_new or sum(buckets[k].values()) != n_new:
            raise RuntimeError("Candidate coverage counts don't reconcile")
        cross_keys = ("queries","clean_raw_union_has_valid_positive",
                      "masked_raw_union_has_valid_positive","masked_raw_top1_disagreement",
                      "masked_raw_top1_disagreement_and_positive_in_union")
        tied = {key:sum(counts[key] for counts in raw_by_bucket[k].values()) for key in cross_keys}
        trigger_keys = ("raw_top1_disagreements",
                        "raw_top1_disagreements_with_valid_positive_in_union",
                        "raw_top1_disagreements_equal_correct",
                        "raw_top1_disagreements_equal_correct_with_valid_positive_in_union")
        shortlist[str(k)] = dict(
            raw_top1_disagreement={phase:{name:counts[name] for name in trigger_keys}
                                   for phase,counts in raw_trigger[k].items()},
            new_equal_miss_raw_trigger=tied,
            new_equal_miss_cross_by_filtered_bucket={
                b:{name:v[name] for name in cross_keys} for b,v in raw_by_bucket[k].items()},
            clean_with_valid_positive=all_trans["1_to_0"] + all_trans["1_to_1"],
            masked_with_valid_positive=all_trans["0_to_1"] + all_trans["1_to_1"],
            valid_positive_transition=all_trans,
            new_equal_miss_valid_positive_transition=miss_trans,
            new_equal_miss_buckets=dict(buckets[k]),
            mean_raw_and_valid_union_size={phase:list(map(float,np.mean(x,axis=0)))
                                           for phase,x in sizes[k].items()})
    return dict(format="occlusion-branch-factorial-v1", condition=condition,
                primary=(condition=="heavy"), primary_k=5, valid_queries=scored,
                newly_wrong_equal=n_new, newly_wrong_fraction_of_valid=n_new/scored,
                missing_valid_negative_for_margin=sum(v is None for v in factorial_margins[CC]),
                native_mix_score_max_abs_diff=error,
                filtered_positive_rank={str(k):decompose(clean,masked,qi,qc,gi,gc,k=k)
                                        for k in CUTS},
                raw_union=shortlist, branch_metrics=branch_stats,
                branch_margin=branch_margin_stats, paired_masked_minus_clean_margin=margin_paired,
                factorial=dict(metrics={k:_stats(x) for k,x in outcomes.items()},
                               margin={k:margins_summary(x) for k,x in factorial_margins.items()},
                               margin_effects_over_all={k:transform_margins(fn) for k,fn in margin_ops.items()},
                               margin_groups_by_filtered_k5=partition_margin,
                               vs_masked_both=all_restore, restored_among_new_equal_misses=recovery,
                               restoration_partition_among_new_equal_misses=recovered_partition,
                               difference_in_differences=dict(rank1=float(interaction[0]),
                                                              mAP=float(interaction[1]))))
