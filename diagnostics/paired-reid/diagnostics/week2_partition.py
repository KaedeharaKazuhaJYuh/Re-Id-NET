"""Plan identity-disjoint Market train partitions from names only (no images read)."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import NormalDist, median

VERSION = "reid-week2-partition-v1"
MARKET = re.compile(r"^(-?[0-9]+)_c([1-9][0-9]*)(?:s[0-9]+)?_[A-Za-z0-9_-]+\.(?:jpg|jpeg|png)$", re.I)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _canon(obj):
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",",":"), allow_nan=False).encode()


def _hash(obj):
    return sha256(_canon(obj)).hexdigest()


def _key(seed, role, item):
    return _hash([seed, role, item])


def parse_filenames(names):
    """Return in-memory (filename, identity, camera) rows without opening pixels."""
    clean, skipped = [], Counter()
    encountered = set()
    for name in sorted(names):
        if Path(name).name != name or "\\" in name or "/" in name:
            raise ValueError("Expected training basenames, not paths")
        if name.casefold() in encountered:
            raise ValueError("Duplicate case-insensitive source basename")
        encountered.add(name.casefold())
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            skipped["non_image_entries"] += 1
            continue
        m = MARKET.fullmatch(name)
        if not m:
            raise ValueError("Unexpected training filename syntax")
        pid, camera = map(int, m.groups())
        if pid <= 0:
            skipped["reserved_nonpositive_identity_images"] += 1
            continue
        clean.append((name, pid, camera))
    if not clean:
        raise ValueError("No usable positive-identity training names")
    return clean, dict(skipped)


def training_names(train_dir):
    """Only accept a non-symlink standard training directory; no test paths opened."""
    p = Path(train_dir)
    if p.name != "bounding_box_train" or not p.is_dir() or p.is_symlink():
        raise ValueError("Use the regular Market-1501 bounding_box_train directory")
    names = []
    for x in p.iterdir():
        if x.suffix.lower() in IMAGE_SUFFIXES and (x.is_symlink() or not x.is_file()):
            raise ValueError("Training input contains a nonregular image entry")
        if x.is_file() and not x.is_symlink():
            names.append(x.name)
    return names


def _select(ids, camera_sets, n, seed, role):
    """Allocate exact counts by largest remainder on camera-presence strata."""
    if n < 0 or n > len(ids):
        raise ValueError(f"Cannot allocate {role}: insufficient eligible identities")
    if not n:
        return set()
    strata = defaultdict(list)
    for pid in ids:
        strata[camera_sets[pid]].append(pid)
    d = len(ids)
    quotas = {s: len(v)*n//d for s,v in strata.items()}
    remainder_order = sorted(strata, key=lambda s: (-(len(strata[s])*n % d), _key(seed, role+"-stratum", s)))
    remaining = n-sum(quotas.values())
    for s in remainder_order[:remaining]:
        quotas[s] += 1
    keep = set()
    for s, members in strata.items():
        ordered = sorted(members,key=lambda pid: _key(seed,role+"-id",pid))
        keep.update(ordered[:quotas[s]])
    if len(keep) != n:
        raise RuntimeError("Stratum quotas do not reconcile")
    return keep


def _qg(group, rows_by_id, seed, role):
    """Freeze one query per identity, leaving every other image in gallery."""
    q, g = [], []
    qcam_count = Counter()
    # Small camera supports allocated first, then hash ties. Within each support,
    # prefer currently least-loaded available camera, not any measured outcome.
    def order_id(pid):
        cams = {row[2] for row in rows_by_id[pid]}
        return len(cams), _key(seed,role+"-query-order",pid)
    for pid in sorted(group,key=order_id):
        by_cam = defaultdict(list)
        for row in rows_by_id[pid]:
            by_cam[row[2]].append(row)
        if len(by_cam) < 2:
            raise RuntimeError("Evaluation identity must have at least two cameras")
        cam = min(by_cam,key=lambda c:(qcam_count[c],_key(seed,role+"-qcam",[pid,c])))
        chosen = min(by_cam[cam],key=lambda r:_key(seed,role+"-query",r[0]))
        q.append(chosen)
        qcam_count[cam] += 1
        g.extend(row for row in rows_by_id[pid] if row != chosen)
    return sorted(q), sorted(g)


def _camera_counts(rows):
    counts = Counter(c for _,_,c in rows)
    ids = defaultdict(set)
    for _,p,c in rows:
        ids[c].add(p)
    return {"images":dict(sorted(counts.items())),
            "identities":{camera:len(ps) for camera,ps in sorted(ids.items())}}


def _audit_holdout(query, gallery):
    per_id = defaultdict(Counter)
    for _,p,c in gallery:
        per_id[p][c] += 1
    pos, cross_pairs = [], Counter()
    for _,p,c in query:
        relevant = {cam:n for cam,n in per_id[p].items() if cam!=c}
        if not relevant:
            raise RuntimeError("A query has no same-identity cross-camera gallery positive")
        pos.append(sum(relevant.values()))
        for cam,n in relevant.items():
            cross_pairs[f"{c}->{cam}"] += n
    if len({p for _,p,_ in query}) != len(query):
        raise RuntimeError("Expected exactly one query per held-out identity")
    return {"queries":len(query), "gallery":len(gallery),
            "queries_with_cross_camera_positive":len(pos),
            "cross_camera_positives_per_query_min":min(pos),
            "cross_camera_positives_per_query_median":float(median(pos)),
            "cross_camera_positives_per_query_max":max(pos),
            "query_camera_audit":_camera_counts(query),
            "gallery_camera_audit":_camera_counts(gallery),
            "same_identity_cross_camera_image_pairs":dict(sorted(cross_pairs.items()))}


def paired_power_guidance(n, *, discordance=.15, delta=.04, alpha=.05, target_power=.8):
    """Planning-only normal approximation on independent paired identity units."""
    if (isinstance(n,bool) or not isinstance(n,int) or n < 1 or
            not (0 < delta < discordance < 1 and 0 < alpha < 1 and .5 < target_power < 1)):
        raise ValueError("Invalid assumed paired power inputs")
    zsig = NormalDist().inv_cdf(1-alpha/2)
    zpow = NormalDist().inv_cdf(target_power)
    approx_n = math.ceil(((zsig*math.sqrt(discordance)+
                           zpow*math.sqrt(discordance-delta**2))/delta)**2)
    return {"independent_identity_units_assumed":n, "one_net_query_pp":100/n,
            "assumed_discordance_fraction":discordance, "assumed_rank1_delta_pp":delta*100,
            "two_sided_alpha":alpha, "target_power":target_power,
            "expected_discordant_identities":n*discordance,
            "approx_identity_units_needed":approx_n,
            "min_all_same_direction_discordances_exact_sign":math.ceil(math.log2(2/alpha)),
            "interpretation":"Descriptive until independent replication; hypothetical assumptions, not observed power. Do not multiply N by masks, views, or repeated seeds."}



def paired_sensitivity(n):
    """Scenario table, with approximate sensitivity to dependence, not study power."""
    scenarios = []
    for frac in (.05,.10,.20):
        row = paired_power_guidance(n, discordance=frac, delta=.02)
        row["design_effect_sensitivity"] = [
            {"design_effect":d, "effective_identity_units":n/d,
             "approx_raw_identity_units_needed":math.ceil(d*row["approx_identity_units_needed"])}
            for d in (1.,1.5,2.)]
        scenarios.append(row)
    return {"target_delta_pp":2., "scenarios":scenarios,
            "conditional_discordant_reference":
                "Net wins = rescues minus harms; cluster identities for uncertainty, and do not treat shared-gallery queries as proved independent."}

def build_manifest(names, *, calibration_identities, evaluation_identities, seed=20261122,
                   exposed_ids=None, min_train_identities=100):
    """Build local split plus aggregate view; prior exposure list is operator-supplied."""
    for value in (seed,calibration_identities,evaluation_identities,min_train_identities):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError("Seed and partition minima must be positive integers")
    if exposed_ids is not None and any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in exposed_ids):
        raise ValueError("Exposure exclusions must be positive integer identities")
    names = tuple(names)
    rows, skipped = parse_filenames(names)
    by_id = defaultdict(list)
    for r in rows:
        by_id[r[1]].append(r)
    all_ids = set(by_id)
    excluded = set(exposed_ids or ())
    if excluded-all_ids:
        raise ValueError("Exposure file mentions identities absent from training names")
    camera_sets = {p:tuple(sorted({row[2] for row in items})) for p,items in by_id.items()}
    eligible = {p for p in all_ids-excluded if len(camera_sets[p])>=2}
    if len(eligible) < calibration_identities + evaluation_identities:
        raise ValueError("Too few multi-camera identities after development-exposure exclusions")
    ev = _select(eligible,camera_sets,evaluation_identities,seed,"evaluation")
    ca = _select(eligible-ev,camera_sets,calibration_identities,seed,"calibration")
    tr = all_ids-ca-ev
    if len(tr) < min_train_identities:
        raise ValueError("Proposed split leaves too few model-training identities")
    if ev & ca or ev & tr or ca & tr or (ca | ev) & excluded:
        raise RuntimeError("Identity overlap despite deterministic assignment")
    tr_rows = [r for r in rows if r[1] in tr]
    eq,eg = _qg(ev,by_id,seed,"evaluation")
    cq,cg = _qg(ca,by_id,seed,"calibration")
    groups = {"model_train":tr_rows,"calibration_query":cq,"calibration_gallery":cg,
              "evaluation_query":eq,"evaluation_gallery":eg}
    all_used = [r[0] for bucket in groups.values() for r in bucket]
    if len(all_used)!=len(set(all_used)) or set(all_used)!={r[0] for r in rows}:
        raise RuntimeError("File allocation is not disjoint and exhaustive")

    source_sha = _hash(sorted(names))
    config = {"seed":seed,"calibration_identities":calibration_identities,
              "evaluation_identities":evaluation_identities,"min_train_identities":min_train_identities,
              "previous_development_ids_supplied":exposed_ids is not None}
    manifest = {"version":VERSION,"source_names_sha256":source_sha,
                "data_source":"bounding_box_train","config":config,
                "previous_development_id_exclusions":sorted(excluded),
                "identity_sets":{"train":sorted(tr),"calibration":sorted(ca),"evaluation":sorted(ev)},
                "records":{group:[{"name":r[0],"identity":r[1],"camera":r[2]} for r in rr]
                           for group,rr in groups.items()}}
    agg = {"version":VERSION,"source_names_sha256":source_sha,"partition_manifest_sha256":_hash(manifest),
           "hash_scope":"Canonical sorted-key compact JSON UTF-8 (not image bytes); detects changes to seen filenames or planned assignment",
           "config":config, "training_directory_only":True,
           "train_calibration_evaluation_disjoint":True,
           "prior_development_identity_exclusions":len(excluded),
           "prior_development_exposure_receipt_present":exposed_ids is not None,
           "model_ancestry_verified":False,"scientific_independence_verified":False,
           "required_before_training_or_evaluation":"Audit all previous development exposure; verify fresh ImageNet-only initialization, no Market checkpoint/optimizer reuse; verify executed sampler and loader against model_train-only manifest and record starting weight/config hashes.",
           "source_audit":{"usable_images":len(rows),"usable_identities":len(all_ids),
                           "eligible_holdout_identities":len(eligible),
                           "identity_camera_count_histogram":dict(sorted(Counter(map(len,camera_sets.values())).items())),
                           "cameras":_camera_counts(rows),"ignored":skipped},
           "model_train":{"identities":len(tr),"images":len(tr_rows),"camera_audit":_camera_counts(tr_rows)},
           "calibration":{"identities":len(ca),**_audit_holdout(cq,cg)},
           "evaluation":{"identities":len(ev),**_audit_holdout(eq,eg)},
           "paired_power_descriptive": paired_sensitivity(len(ev))}
    agg["local_role_view_sha256"] = {
        k:_hash(v) for k,v in role_views(manifest,agg["partition_manifest_sha256"]).items()}
    return manifest, agg



def role_views(manifest, partition_sha):
    """Separate deterministic input views so future runners use one authorized role."""
    scopes = {"train":("model_train",),
              "calibration":("calibration_query","calibration_gallery"),
              "evaluation":("evaluation_query","evaluation_gallery")}
    return {role: {"version":VERSION, "partition_manifest_sha256":partition_sha,
                   "role":role, "identity_ids":manifest["identity_sets"][role],
                   "records":{k:manifest["records"][k] for k in groups}}
            for role, groups in scopes.items()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path,help="New local private directory")
    p.add_argument("--seed",type=int,default=20261122)
    p.add_argument("--calibration-identities",required=True,type=int)
    p.add_argument("--evaluation-identities",required=True,type=int)
    p.add_argument("--min-train-identities",default=100,type=int)
    p.add_argument("--expect-prior-dev-count",type=int)
    p.add_argument("--prior-dev-ids",type=Path,help='Optional local JSON {"excluded_identity_ids": [...]}; aggregate then marks the receipt present, not verified.')
    args = p.parse_args()
    out = args.output_dir
    if out.exists() or out.is_symlink() or out.resolve().is_relative_to(args.train_dir.resolve().parent):
        p.error("Use a new output path outside the dataset root")
    exposed = None
    if args.prior_dev_ids is not None:
        receipt = json.loads(args.prior_dev_ids.read_text(encoding="utf-8"))
        if not isinstance(receipt,dict) or set(receipt)!={"excluded_identity_ids"} or not isinstance(receipt["excluded_identity_ids"],list):
            p.error("Expected local JSON object with excluded_identity_ids list")
        exposed = receipt["excluded_identity_ids"]
    if exposed is not None and any(type(x) is not int or x < 1 for x in exposed):
        p.error("Prior dev IDs must be positive integers")
    if args.expect_prior_dev_count is not None:
        if (args.expect_prior_dev_count < 0 or exposed is None or
                len(exposed) != args.expect_prior_dev_count or len(set(exposed)) != len(exposed)):
            p.error("Missing or mismatched number of distinct prior dev identities")
    local, summary = build_manifest(training_names(args.train_dir),
                                    calibration_identities=args.calibration_identities,
                                    evaluation_identities=args.evaluation_identities,
                                    seed=args.seed,exposed_ids=exposed,
                                    min_train_identities=args.min_train_identities)
    out.mkdir(mode=0o700,parents=False,exist_ok=False)
    with (out/"local_identity_manifest.json").open("xb") as f:
        f.write(_canon(local)+b"\n")
    for role, view in role_views(local,summary["partition_manifest_sha256"]).items():
        with (out/f"local_{role}_only.json").open("xb") as f:
            f.write(_canon(view)+b"\n")
    with (out/"aggregate_audit.json").open("xb") as f:
        f.write(_canon(summary)+b"\n")
    print(json.dumps(summary,ensure_ascii=True,sort_keys=True,allow_nan=False))


if __name__ == "__main__":
    main()
