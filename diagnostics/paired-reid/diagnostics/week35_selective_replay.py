"""Frozen calibration-only candidate replay. No dataset or filesystem I/O."""
from collections import Counter
import hashlib
import math
import numpy as np

VERSION = "week35-selective-calibration-v1"
CUTS = (5, 10, 20)
TOPK = 5
BUDGET = .20
GAP = .05
ADVANTAGE = .02
CLEAN_RANK1 = .70
CLEAN_MAP = .55
MIN_LEGAL_SHARED_WRONG = 3
MIN_ACTUAL_RESCUES = 3
MIN_CLEAN_DELTA = -.02
HASH_SEED = 20261122
KEYS = ("backbone","fusion","equal")


def _hash(salt, value):
    msg = f"{HASH_SEED}/{salt}/{int(value)}".encode()
    return hashlib.sha256(msg).digest()


def _order(a):
    return np.argsort(-a,kind="stable")


def _variant(scores, n, g):
    out = {k:np.asarray(scores[k],dtype=np.float64) for k in KEYS}
    if any(x.shape!=(n,g) or not np.isfinite(x).all() for x in out.values()):
        raise ValueError("Expected aligned finite clean and heavy scores")
    return out


def make_stripe_scorer(query_stripes, gallery_stripes):
    """Local score via two best same-position cosine pairs of four frozen stripes."""
    q, g = map(lambda x:np.asarray(x,dtype=np.float64),(query_stripes,gallery_stripes))
    if (q.ndim != 3 or g.ndim != 3 or q.shape[1]!=4 or q.shape[1:]!=g.shape[1:]
            or q.shape[-1]<1 or not len(q) or not len(g)):
        raise ValueError("Expected aligned query/gallery four-stripe embeddings")
    def norm(x):
        finite=np.isfinite(x).all(axis=-1)
        safe=np.where(finite[...,None],x,0)
        n=np.linalg.norm(safe,axis=-1)
        valid=finite & np.isfinite(n) & (n>1e-12)
        v=safe/np.where(valid,n,1)[...,None]
        return v,valid
    qn,qv=norm(q); gn,gv=norm(g)
    def scorer(i, candidates):
        ids=np.asarray(candidates,dtype=int)
        vals=np.einsum("sd,ksd->ks",qn[i],gn[ids])
        valid=qv[i][None,:] & gv[ids] & np.isfinite(vals)
        values=np.sort(np.where(valid,vals,-np.inf),axis=1)[:,-2:]
        out=np.full(len(ids),np.nan)
        available=valid.sum(axis=1)>=2
        out[available]=values[available].mean(axis=1)
        return out
    return scorer


def _evaluate(order,positive,valid):
    matches=positive[order[valid[order]]]
    ranks=np.flatnonzero(matches)+1
    return float(matches[0]), float(np.mean(np.arange(1,len(ranks)+1)/ranks))


def _promote(order,i):
    return np.r_[i,order[order!=i]] if i!=order[0] else order


def _observe_pair_scores(s, rows):
    observations=[]
    for i,r in enumerate(rows):
        b,f,e=[s[k][i] for k in KEYS]
        bo,fo,eo=map(_order,(b,f,e))
        p,v=r
        original=_evaluate(eo,p,v)
        incumbent=int(eo[0]); shared=(bo[0]==fo[0]); inc_legal=bool(v[incumbent])
        shared_this=bool(shared and bo[0]==incumbent)
        def raw_union(k):
            return np.union1d(np.union1d(bo[:k],fo[:k]),[incumbent])
        shortlist=raw_union(TOPK)
        allowed=shortlist[(e[incumbent]-e[shortlist])<=GAP]
        rr=np.empty_like(b); ss=np.empty_like(f)
        rr[bo]=1/(60+np.arange(1,len(b)+1)); ss[fo]=1/(60+np.arange(1,len(f)+1))
        rrf=rr+ss
        rival=int(shortlist[np.argmax(rrf[shortlist])])
        rank_promotion=rival if rrf[rival]>rrf[incumbent] else incumbent
        def gap(a,o):
            return float(a[o[0]]-a[o[1]])
        observations.append(dict(
            original=original, order=eo, incumbent=incumbent, union=shortlist, allowed=allowed,
            positive=p, valid=v, eligible_wrong_agreement=bool(
                shared_this and inc_legal and not p[incumbent] and not original[0]),
            eligible_right_agreement=bool(shared_this and inc_legal and p[incumbent] and original[0]),
            shared_raw_filtered=bool(shared_this and not inc_legal),
            raw_top_disagreement=not shared,
            raw_opportunity={str(k):bool(np.any((p&v)[raw_union(k)])) for k in CUTS},
            rank_only=_evaluate(_promote(eo,rank_promotion),p,v),
            rank_changed=rank_promotion!=incumbent,
            feature=(gap(e,eo),gap(b,bo),gap(f,fo),float(not shared)),
            full_rrf=_evaluate(_order(rrf),p,v)))
    return observations


def _local_operator(obs, scorer):
    metrics, scorable, head_changed, pair_evals=[],[],[],[]
    for i,row in enumerate(obs):
        inc, candidates=row["incumbent"],row["allowed"]
        if len(candidates)<=1:
            metrics.append(row["original"]);scorable.append(False);head_changed.append(False);pair_evals.append(0);continue
        vec=np.asarray(scorer(i,tuple(map(int,candidates))),dtype=np.float64)
        if vec.shape!=(len(candidates),) or np.any(np.isinf(vec)):
            raise ValueError("Local scorer must return aligned finite scores or NaN")
        if np.any(np.abs(vec[np.isfinite(vec)])>1+1e-7):
            raise ValueError("Local score must be the frozen cosine measure")
        i0=int(np.flatnonzero(candidates==inc)[0])
        has_inc=bool(np.isfinite(vec[i0]))
        eligible=(row["positive"]&row["valid"])[candidates]
        scorable.append(bool(has_inc and np.any(eligible & np.isfinite(vec))))   # evaluator only
        move=inc
        if has_inc:
            best=int(np.argmax(np.where(np.isfinite(vec),vec,-np.inf)))
            if vec[best]-vec[i0] > ADVANTAGE:   # inference: no expected identity/camera/mask
                move=int(candidates[best])
        metrics.append(_evaluate(_promote(row["order"],move),row["positive"],row["valid"]))
        head_changed.append(move!=inc)
        pair_evals.append(len(candidates))
    return np.asarray(metrics),np.asarray(scorable),np.asarray(head_changed),np.asarray(pair_evals)


def _outcomes(obs,field):
    return np.asarray([v[field] for v in obs],dtype=np.float64)


def _delta(base, alt):
    dif=alt-base
    return dict(rank1=float(alt[:,0].mean()), mAP=float(alt[:,1].mean()),
                delta_rank1_pp=float(100*dif[:,0].mean()), delta_mAP_pp=float(100*dif[:,1].mean()),
                rescues=int(np.count_nonzero(dif[:,0]>0)), harms=int(np.count_nonzero(dif[:,0]<0)))


def _positives(obs, heavy_budget):
    out={}
    for k in map(str,CUTS):
        eligible=sum(not r["original"][0] and r["raw_opportunity"][k] for r in obs)
        agreement=sum(r["eligible_wrong_agreement"] and r["raw_opportunity"][k] for r in obs)
        disagr=sum(r["raw_top_disagreement"] and not r["original"][0] and r["raw_opportunity"][k]
                   for r in obs)
        out[k]=dict(original_miss_with_legal_positive=eligible,shared_wrong_with_legal_positive=agreement,
                    raw_disagreement_miss_with_legal_positive=disagr,
                    optimistic_budget_rescue_upper_bound=min(eligible,heavy_budget))
    return out


def _selection(obs,budget,kind,demand=None):
    ids=range(len(obs))
    if kind=="disagreement_only":
        ids=[i for i in ids if obs[i]["raw_top_disagreement"]]
    if kind=="random":
        order=sorted(ids,key=lambda i:_hash("random",i))
    elif kind in ("disagreement_only","margin_only"):
        order=sorted(ids,key=lambda i:(obs[i]["feature"][0],_hash("tie",i)))
    elif kind=="expected_gain":
        order=sorted(ids,key=lambda i:(-float(demand[i]),_hash("tie",i)))
    else:
        raise ValueError("Unknown selection policy")
    return order[:budget]


def _budget_eval(obs,result,act,*,moves,local_pairs=None):
    orig=_outcomes(obs,"original")
    ix=np.asarray(act,dtype=int)
    out=orig.copy(); out[ix]=result[ix]
    metric=_delta(orig,out)
    shared_w=np.asarray([v["eligible_wrong_agreement"] for v in obs])
    shared_r=np.asarray([v["eligible_right_agreement"] for v in obs])
    delta=out[:,0]-orig[:,0]
    metric.update(activations=len(ix), head_changes=int(moves[ix].sum()),
                  shared_wrong_rescues=int(np.count_nonzero(shared_w & (delta>0))),
                  shared_right_harms=int(np.count_nonzero(shared_r & (delta<0))))
    if local_pairs is not None:
        metric["local_similarity_pairs"]=int(local_pairs[ix].sum())
    return metric


def _sigmoid(s):
    return 1/(1+np.exp(-np.clip(s,-40,40)))


def _fit_predict(x,y,eval_list):
    # Fixed logistic, no optimizer choice, hyperparameter sweep, or test refits.
    loc=x.mean(axis=0); scale=np.maximum(x.std(axis=0),1e-6)
    z=np.c_[np.ones(len(x)), np.clip((x-loc)/scale,-8,8)]
    co=np.zeros(z.shape[1]); p=float(y.mean())
    co[0]=np.log(p/(1-p))
    for _ in range(350):
        residual=_sigmoid(z@co)-y
        penalty=np.r_[0,co[1:]]*.1
        co-=.1*((z.T@residual)/len(x)+penalty)
    return tuple(_sigmoid(np.c_[np.ones(len(v)),np.clip((v-loc)/scale,-8,8)]@co)
                 for v in eval_list)


def _oof(cal_obs,clean_obs,ids,local):
    base=_outcomes(cal_obs,"original")
    change=local[:,0]-base[:,0]
    yR=(change>0).astype(float); yH=(change<0).astype(float)
    if yR.sum()<8 or yH.sum()<4:
        return None,None,{"fit":False,"reason":"requires at least 8 rescues and 4 harms before per-fold checks"}
    units=sorted(set(ids),key=lambda p:_hash("identity-fold",p))
    folds={p:i%5 for i,p in enumerate(units)}
    f=np.array([folds[p] for p in ids])
    x=np.asarray([r["feature"] for r in cal_obs]); y=np.asarray([r["feature"] for r in clean_obs])
    dh=np.empty(len(x)); dc=np.empty(len(y))
    for i in range(5):
        train=f!=i; test=~train
        if not test.any() or min(int(yR[train].sum()),int((1-yR[train]).sum()),
                                 int(yH[train].sum()),int((1-yH[train]).sum()))<3:
            return None,None,{"fit":False,"reason":"identity folds lack three examples per class in training"}
        rH,rC=_fit_predict(x[train],yR[train],(x[test],y[test]))
        hH,hC=_fit_predict(x[train],yH[train],(x[test],y[test]))
        dh[test],dc[test]=rH-hH,rC-hC
    return dh,dc,{"fit":True,"folds":5,"fit_scope":"calibration OOF identity only; no model or per-row score persisted"}


def replay(clean,heavy,query_ids,query_cams,gallery_ids,gallery_cams,
           *, clean_local=None, heavy_local=None, role="calibration"):
    """Calibration/synthetic only; raw score arrays/callbacks stay on local executor."""
    if role not in {"calibration","synthetic"}:
        raise ValueError("Protected/official/final evaluation is forbidden")
    qi,qc,gi,gc=map(np.asarray,(query_ids,query_cams,gallery_ids,gallery_cams))
    n,g=len(qi),len(gi)
    if (any(x.ndim!=1 for x in (qi,qc,gi,gc)) or len(qc)!=n or len(gc)!=g
            or n<1 or g<2 or (role=="calibration" and n!=120)
            or len(set(qi))!=n or any(v<=0 for v in qi)):
        raise ValueError("Unexpected fixed 120-calibration identity or aligned metadata contract")
    c,h=(_variant(x,n,g) for x in (clean,heavy))
    rows=[(gi==p,(gi!=-1)&~((gi==p)&(gc==cam))) for p,cam in zip(qi,qc)]
    if any(not np.any(p&v) for p,v in rows):
        raise ValueError("Every identity must have a valid cross-camera positive")
    co,ho=[_observe_pair_scores(z,rows) for z in (c,h)]
    cb,hb=[_outcomes(o,"original") for o in (co,ho)]
    budget=n//5
    oracle=_positives(ho,budget)
    noise={k:_delta(_outcomes(o,"original"),_outcomes(o,"full_rrf")) for k,o in (("clean",co),("heavy",ho))}
    quality=(cb[:,0].mean()>=CLEAN_RANK1 and cb[:,1].mean()>=CLEAN_MAP)
    cheap=bool(quality and oracle["5"]["shared_wrong_with_legal_positive"]>=MIN_LEGAL_SHARED_WRONG
               and oracle["5"]["optimistic_budget_rescue_upper_bound"]>=MIN_ACTUAL_RESCUES)
    out=dict(version=VERSION, role=role, evaluated_identities=n, activation_budget=budget,
             primary="heavy fixed raw Top5 each + incumbent; Top10/20 descriptive",
             clean_baseline=_delta(cb,cb), heavy_baseline=_delta(hb,hb), full_rrf=noise,
             oracle=oracle, lightweight_gate=dict(clean_quality_pass=bool(quality),
                  enough_raw_legal_shared_wrong=oracle["5"]["shared_wrong_with_legal_positive"]>=MIN_LEGAL_SHARED_WRONG,
                  pass_for_local_screen=cheap),
             legal_shared_wrong_total={k:sum(r["eligible_wrong_agreement"] for r in o) for k,o in (("clean",co),("heavy",ho))},
             filtered_shared_raw_incumbents={k:sum(r["shared_raw_filtered"] for r in o) for k,o in (("clean",co),("heavy",ho))},
             final_scope_opened=False)
    if not cheap:
        out["status"]="NO_GO_PHASE0_CHEAP_BOUND"
        return out
    if heavy_local is None or clean_local is None:
        out["status"]="AWAITING_FROZEN_LOCAL_EVIDENCE"
        return out
    hl,scorable,hm,hp=_local_operator(ho,heavy_local)
    cl,_,cm,cp=_local_operator(co,clean_local)
    scorable_n=sum(r["eligible_wrong_agreement"] and scorable[i] for i,r in enumerate(ho))
    hres=np.flatnonzero(hl[:,0]>hb[:,0])
    agree_res=sum(ho[i]["eligible_wrong_agreement"] for i in hres)
    out["local_operator_probe"]={
        "scorable_legal_shared_wrong":int(scorable_n),
        "actual_heavy_rescues_all_queries":len(hres),
        "actual_heavy_rescues_legal_shared_wrong":int(agree_res),
        "unbudgeted_operator_diagnostic":{"heavy":_delta(hb,hl), "clean":_delta(cb,cl)},
        "unbudgeted_diagnostic_not_deployment":True,
        "optimistic_budget_cap_of_observed_operator_rescues":min(budget,len(hres))}
    if scorable_n<MIN_LEGAL_SHARED_WRONG or len(hres)<MIN_ACTUAL_RESCUES or agree_res==0:
        out["status"]="NO_GO_PHASE0_LOCAL_HEADROOM"
        return out
    dh,dc,fit=_oof(ho,co,qi,hl)
    out["demand_model"]=fit
    selectors=("random","disagreement_only","margin_only")+(("expected_gain",) if fit["fit"] else ())
    # Strict disagreement is never padded with agreement queries. Match
    # realized activations per condition, and separately inspect the full 20%
    # cap for flexible policies (not marketed as cost-matched at lower support).
    common={k:min(budget,sum(r["raw_top_disagreement"] for r in obs))
            for k,obs in (("clean",co),("heavy",ho))}
    coverage={}
    max_budget_guard={}
    for selector in selectors:
        coverage[selector]={}
        max_budget_guard[selector]={}
        for condition,obs,local,moves,n_pairs,demand in (
                ("clean",co,cl,cm,cp,dc),("heavy",ho,hl,hm,hp,dh)):
            ix=_selection(obs,common[condition],selector,demand)
            coverage[selector][condition]={
                "local":_budget_eval(obs,local,ix,moves=moves,local_pairs=n_pairs),
                "rank_only":_budget_eval(obs,_outcomes(obs,"rank_only"),ix,
                                       moves=np.array([r["rank_changed"] for r in obs]))}
            nominal=_selection(obs,budget,selector,demand)
            max_budget_guard[selector][condition]=_budget_eval(
                obs,local,nominal,moves=moves,local_pairs=n_pairs)
        for table, nested in ((coverage,True),(max_budget_guard,False)):
            guard=table[selector]["clean"]
            if nested:
                guard=guard["local"]
            table[selector]["clean_harm_guard_pass"]=bool(
                guard["delta_rank1_pp"]>=100*MIN_CLEAN_DELTA and guard["delta_mAP_pp"]>=100*MIN_CLEAN_DELTA)
    out.update(status="CALIBRATION_ONLY_DO_NOT_OPEN_FINAL",matched_actual_activations=common,
               matched_replay=coverage, nominal_cap_replay_not_necessarily_cost_matched=max_budget_guard,
               independent_evaluation_authorized=False)
    return out
