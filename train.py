"""Train the independent paper reconstruction; defaults to synthetic CPU smoke."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import Market1501, PKBatchSampler, SyntheticReID, evaluate_reid
from losses import batch_hard_triplet, cross_covariance, objective, ranking_regularizer
from model import ReID


def market_transform(x, training):
    shape = x.shape[-2:]
    if training:
        images = []
        for v in x:
            if torch.rand(()) < .5:
                v = v.flip(-1)
            v = F.pad(v, (10, 10, 10, 10))
            iy, ix = torch.randint(21, (2,)).tolist()
            images.append(v[:, iy:iy+shape[0], ix:ix+shape[1]])
        x = torch.stack(images)
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    x = (x - mean) / std
    if training:
        for v in x:
            if torch.rand(()) < .5:
                h, w = shape
                for _ in range(10):
                    area = float(torch.empty(1).uniform_(.02, .40)) * h*w
                    ratio = math.exp(float(torch.empty(1).uniform_(math.log(.3), math.log(3.3))))
                    dh, dw = round(math.sqrt(area*ratio)), round(math.sqrt(area/ratio))
                    if 0 < dh < h and 0 < dw < w:
                        i, j = int(torch.randint(h-dh+1, ())), int(torch.randint(w-dw+1, ()))
                        v[:, i:i+dh, j:j+dw] = 0
                        break
    return x


@torch.inference_mode()
def collect(model, loader, device, market=False):
    vals, pids, cams = {}, [], []
    for images, ids, camera in loader:
        if market:
            images = market_transform(images, False)
        for key, v in model(images.to(device))["retrieval"].items():
            vals.setdefault(key, []).append(v.cpu())
        pids.append(ids)
        cams.append(camera)
    return {k: torch.cat(v) for k, v in vals.items()}, torch.cat(pids), torch.cat(cams)


def evaluate(model, query, gallery, device, market):
    model.eval()
    q, qp, qc = collect(model, query, device, market)
    g, gp, gc = collect(model, gallery, device, market)
    return {k: evaluate_reid(q[k], qp, qc, g[k], gp, gc) for k in q}


def run(args):
    torch.manual_seed(args.seed)
    if args.device == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch environment has no usable CUDA.")
    market = args.market_root is not None
    if args.backend == 'official-tiny' and device.type != 'cuda':
        raise ValueError("The official-tiny adapter uses the upstream CUDA selective scan.")
    size = (256, 128) if market or args.backend == 'official-tiny' else (64, 32)
    if market:
        tr = Market1501(args.market_root, 'train', size)
        qu, ga = (None, None) if args.skip_eval else [Market1501(args.market_root, s, size)
                                                    for s in ('query', 'gallery')]
    else:
        tr = SyntheticReID('train', 8, 4, size, args.seed)
        # Distinct procedural identity templates from training, shared across query/gallery.
        qu, ga = [SyntheticReID(s, 6, 4, size, args.seed+990001) for s in ('query','gallery')]
    p = args.p if args.p is not None else (16 if market else 4)
    sampler = PKBatchSampler(tr.pids, p, args.k, args.seed)
    training = DataLoader(tr, batch_sampler=sampler)
    query, gallery = [None if d is None else DataLoader(d, batch_size=args.eval_batch)
                      for d in (qu, ga)]
    hcen = args.structure != 'backbone'
    aux = args.structure in ('full','hcen-aul')
    sasf = args.structure == 'full'
    model = ReID(tr.classes, args.backend, args.imagenet_checkpoint, hcen=hcen,
                 sasf=sasf, aux=aux, fusion_dim=args.fusion_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    warmup = args.warmup if args.warmup is not None else (20 if market else 0)
    steps, epoch_log = 0, []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(args.epochs):
        multiplier = ((epoch+1)/warmup if epoch < warmup
                      else .5*(1+math.cos(math.pi*(epoch-warmup)/max(1,args.epochs-warmup))))
        for group in opt.param_groups:
            group['lr'] = args.lr*multiplier
        model.train()
        sampler.set_epoch(epoch)
        values = []
        for images, ids, _ in training:
            ids = ids.to(device)
            if market:
                images = market_transform(images, True)
            opt.zero_grad(set_to_none=True)
            out = model(images.to(device))
            if hcen:
                parts = objective(out, ids, variant=args.variant, tau=args.tau, margin=args.margin,
                                  cov_weight=args.cov_weight, consistency_weight=args.consistency_weight)
                loss = parts["loss"]
            else:
                z, logits = out['backbone']
                loss = F.cross_entropy(logits, ids, label_smoothing=.1) + batch_hard_triplet(z, ids, args.margin)
                parts = {"loss": loss}
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss after {steps} optimizer steps.")
            loss.backward()
            if steps == 0 and not all(p.grad is None or torch.isfinite(p.grad).all()
                                      for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradient on first step.")
            opt.step()
            if hcen and not market:
                with torch.no_grad():
                    emb_a, emb_b = out["backbone"][0], out["fusion"][0]
                    intra, inter = ranking_regularizer(emb_a, emb_b, ids, args.tau)
                    parts.update(signed_intra_diagnostic=intra, signed_inter_diagnostic=inter,
                                 cross_cov_diagnostic=cross_covariance(emb_a, emb_b))
            values.append({k: float(v.detach()) for k, v in parts.items()})
            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                break
        epoch_log.append({"epoch": epoch+1, "steps": len(values), "lr": args.lr*multiplier,
                          **{k: sum(d[k] for d in values)/len(values) for k in values[0]}})
        if args.max_steps is not None and steps >= args.max_steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter()-start
    stats = None if args.skip_eval else evaluate(model, query, gallery, device, market)
    if stats and "combined" in stats:
        stats["fusion_gain_over_best_branch"] = stats["combined"]["mAP"] - max(
            stats["backbone"]["mAP"], stats["fusion"]["mAP"])
    if args.checkpoint_out is not None:
        torch.save({"model": model.state_dict(), "config": vars(args)}, args.checkpoint_out)
    return {"dataset": "market1501" if market else "synthetic-smoke-only",
            "backend": args.backend, "torch": torch.__version__, "device": str(device),
            "structure": args.structure, "variant": args.variant, "seed": args.seed,
            "shape": size, "p": p, "k": args.k, "tau": args.tau, "margin": args.margin,
            "cov_weight": args.cov_weight, "consistency_weight": args.consistency_weight,
            "steps": steps, "epochs": epoch_log, "training_seconds": seconds,
            "imagenet_weights_supplied": args.imagenet_checkpoint is not None,
            "params_including_train_heads": sum(p.numel() for p in model.parameters()),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "evaluation": stats}


def args_from(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['reference','official-tiny'], default='reference')
    p.add_argument('--market-root', type=Path)
    p.add_argument('--imagenet-checkpoint', type=Path)
    p.add_argument('--checkpoint-out', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--structure', choices=['backbone','hcen','hcen-aul','full'], default='full')
    p.add_argument('--variant', choices=['none','ratr','cov','cov_consistency'], default='ratr')
    p.add_argument('--epochs', type=int)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--warmup', type=int)
    p.add_argument('--p', type=int)
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--fusion-dim', type=int)
    p.add_argument('--tau', type=float, default=.1)
    p.add_argument('--margin', type=float, default=.3)
    p.add_argument('--cov-weight', type=float, default=.1)
    p.add_argument('--consistency-weight', type=float, default=.1)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--lr', type=float, default=3.5e-4)
    p.add_argument('--eval-batch', type=int, default=32)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--skip-eval', action='store_true', help='Keep official test split untouched during preflight.')
    p.add_argument('--out', type=Path, default=Path('result.json'))
    a = p.parse_args(argv)
    if a.epochs is None:
        a.epochs = 160 if a.market_root else 1
    if min(a.epochs, a.k, a.eval_batch, a.cpu_threads) < 1 or (a.max_steps is not None and a.max_steps < 1):
        p.error("Epochs, K, evaluation batch, threads and explicit max steps must be positive.")
    if a.warmup is not None and a.warmup < 0:
        p.error("Warmup must be nonnegative.")
    if a.tau <= 0 or a.lr <= 0 or min(a.margin, a.cov_weight, a.consistency_weight) < 0:
        p.error("Invalid temperature, learning rate, or loss weight.")
    if a.fusion_dim is not None and a.fusion_dim < 2:
        p.error("Fusion width must be at least 2.")
    if a.market_root and a.backend != 'official-tiny':
        p.error("Market mode requires the official-tiny adapter; smoke reference uses synthetic data only.")
    if a.market_root and not a.imagenet_checkpoint:
        p.error("Market mode requires an explicit local ImageNet Tiny checkpoint.")
    if a.structure != 'full' and a.variant != 'none':
        p.error("Select --variant none to reproduce the structural ablations.")
    return a


if __name__ == '__main__':
    args = args_from()
    result = run(args)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('dataset','variant','steps','evaluation')}, ensure_ascii=False))
