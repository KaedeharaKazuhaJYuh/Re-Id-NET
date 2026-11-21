"""Train the independent paper reconstruction; defaults to synthetic CPU smoke."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import Market1501, PKBatchSampler, SyntheticReID, evaluate_reid, market_train_holdout
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


def resume_config(args, tr, device, protocol=None):
    # Save only safe-deserializable data. Market identity is based on the ordered
    # training filenames, not guessed from the total sample count.
    opts = ('backend', 'market_root', 'imagenet_checkpoint', 'structure', 'variant',
            'epochs', 'warmup', 'p', 'k', 'fusion_dim', 'tau', 'margin',
            'cov_weight', 'consistency_weight', 'seed', 'lr', 'cpu_threads',
            'progress_every')
    config = {k: str(v.resolve()) if isinstance(v, Path) else v
              for k in opts for v in (getattr(args, k),)}
    config.update(device=str(device), torch=str(torch.__version__))
    if args.market_root:
        index = '\n'.join(p.name for p, _, _ in tr.items).encode('utf-8')
        config['train_index_sha256'] = hashlib.sha256(index).hexdigest()
    if protocol is not None:
        config["validation_protocol"] = protocol
    return config


def atomic_checkpoint(path, payload):
    # Replace only after a successful write; a hard stop leaves the prior
    # checkpoint usable, even if a temporary file is orphaned.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


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
    protocol = None
    if market and args.validation_identities:
        tr, qu, ga, protocol = market_train_holdout(
            args.market_root, args.validation_identities, args.validation_split_seed, size,
            min_train_ids=args.p if args.p is not None else 16)
        if args.skip_eval:
            qu = ga = None
    elif market:
        tr = Market1501(args.market_root, 'train', size)
        qu, ga = (None, None) if args.skip_eval else [Market1501(args.market_root, s, size)
                                                    for s in ('query', 'gallery')]
    else:
        tr = SyntheticReID('train', 8, 4, size, args.seed)
        # Distinct procedural identity templates from training, shared across query/gallery.
        qu, ga = [SyntheticReID(s, 6, 4, size, args.seed+990001) for s in ('query','gallery')]
    p = args.p if args.p is not None else (16 if market else 4)
    sampler = PKBatchSampler(tr.pids, p, args.k, args.seed)
    # DataLoader creates a base seed even with zero workers. Keep it out of
    # the RNG stream used by model stochasticity and training transforms.
    training = DataLoader(tr, batch_sampler=sampler, generator=torch.Generator().manual_seed(args.seed))
    query, gallery = [None if d is None else DataLoader(d, batch_size=args.eval_batch)
                      for d in (qu, ga)]
    hcen = args.structure != 'backbone'
    aux = args.structure in ('full','hcen-aul')
    sasf = args.structure == 'full'
    state_path = args.resume_state or args.resume_from
    config = resume_config(args, tr, device, protocol) if state_path else None
    progress = dict(steps=0, epoch=0, offset=0, totals={}, epoch_log=[],
                    window_steps=0, window_loss=0.)
    state = None
    if args.resume_from:
        state = torch.load(args.resume_from, map_location='cpu', weights_only=True)
        if state.get('version') != 1 or state.get('config') != config:
            raise ValueError("Resume checkpoint version or training configuration differs.")
        progress = state['progress']
        if (not 0 <= progress['epoch'] <= args.epochs
                or not 0 <= progress['offset'] < sampler.steps
                or progress['steps'] != progress['epoch'] * sampler.steps + progress['offset']
                or len(progress['epoch_log']) != progress['epoch']
                or (progress['epoch'] == args.epochs and progress['offset'])):
            raise ValueError("Invalid checkpoint training cursor.")
        if args.max_steps is not None and args.max_steps < progress['steps']:
            raise ValueError("--max-steps is cumulative and cannot be smaller than the checkpoint step.")
    model = ReID(tr.classes, args.backend, None if state else args.imagenet_checkpoint,
                 hcen=hcen, sasf=sasf, aux=aux, fusion_dim=args.fusion_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    if state:
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['rng_cuda'], device)
        del state

    def save_state():
        if not state_path:
            return
        atomic_checkpoint(state_path, {
            'version': 1, 'config': config, 'progress': progress,
            'model': model.state_dict(), 'optimizer': opt.state_dict(),
            'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        })

    warmup = args.warmup if args.warmup is not None else (20 if market else 0)
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(progress['epoch'], args.epochs):
        if args.max_steps is not None and progress['steps'] >= args.max_steps:
            break
        multiplier = ((epoch+1)/warmup if epoch < warmup
                      else .5*(1+math.cos(math.pi*(epoch-warmup)/max(1,args.epochs-warmup))))
        for group in opt.param_groups:
            group['lr'] = args.lr*multiplier
        model.train()
        sampler.set_epoch(epoch, progress['offset'])
        for images, ids, _ in training:
            steps = progress['steps']
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
            for k, value in parts.items():
                progress['totals'][k] = progress['totals'].get(k, 0.) + float(value.detach())
            steps += 1
            progress['steps'] = steps
            progress['offset'] += 1
            if args.progress_every:
                progress['window_steps'] += 1
                progress['window_loss'] += float(loss.detach())
            epoch_end = progress['offset'] == sampler.steps
            at_limit = args.max_steps is not None and steps >= args.max_steps
            if epoch_end:
                progress['epoch_log'].append({
                    "epoch": epoch+1, "steps": progress['offset'], "lr": args.lr*multiplier,
                    **{k: v/progress['offset'] for k, v in progress['totals'].items()},
                })
                progress.update(epoch=epoch+1, offset=0, totals={})
            report = None
            if args.progress_every and steps % args.progress_every == 0:
                report = {"event": "train_progress", "step": steps, "epoch": epoch+1,
                          "window_steps": progress['window_steps'],
                          "mean_loss": progress['window_loss']/progress['window_steps'],
                          "lr": args.lr*multiplier,
                          "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                                        if device.type == "cuda" else None)}
                progress.update(window_steps=0, window_loss=0., latest_report=report)
            if state_path and (epoch_end or at_limit or steps % args.save_every_steps == 0):
                save_state()
            if report is not None:
                print(json.dumps(report, allow_nan=False), flush=True)
            if at_limit:
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter()-start
    epoch_log = list(progress['epoch_log'])
    if progress['offset']:
        epoch_log.append({"epoch": progress['epoch']+1, "steps": progress['offset'],
                          "lr": opt.param_groups[0]['lr'], "partial": True,
                          **{k: v/progress['offset'] for k, v in progress['totals'].items()}})
    complete = progress['epoch'] == args.epochs
    validation_at_limit = bool(args.eval_validation_at_limit and not complete
                               and progress['steps'] == args.max_steps)
    if validation_at_limit:
        # A cap reached by this loop saved state above. If resuming exactly at
        # the cap into a new output path, materialize that copy before eval.
        if not state_path.is_file():
            save_state()
    stats = (None if args.skip_eval or (state_path and not complete and not validation_at_limit)
             else evaluate(model, query, gallery, device, market))
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
            "steps": progress['steps'], "epochs": epoch_log, "training_seconds": seconds,
            "timing_and_memory_scope": "current_process",
            "imagenet_weights_supplied": args.imagenet_checkpoint is not None,
            "params_including_train_heads": sum(p.numel() for p in model.parameters()),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "evaluation": stats,
            **({"validation_protocol": protocol,
                "evaluation_split": "market1501_train_identity_holdout",
                "validation_evaluated_at_step_limit": validation_at_limit} if protocol else {})}


def args_from(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['reference','official-tiny'], default='reference')
    p.add_argument('--market-root', type=Path)
    p.add_argument('--validation-identities', type=int, default=0,
                   help='Hold out this many cross-camera IDs from Market bounding_box_train only; 0 keeps default.')
    p.add_argument('--eval-validation-at-limit', action='store_true',
                   help='Evaluate the train-only holdout after a saved cumulative --max-steps cap.')
    p.add_argument('--validation-split-seed', type=int, default=20261121,
                   help='Seed for validation identity and query selection, separate from the training seed.')
    p.add_argument('--imagenet-checkpoint', type=Path)
    p.add_argument('--checkpoint-out', type=Path, help='Final model export (not resume state).')
    p.add_argument('--resume-state', type=Path, help='Write atomic optimizer/training state here.')
    p.add_argument('--resume-from', type=Path, help='Resume a local checkpoint with the same training configuration.')
    p.add_argument('--save-every-steps', type=int, default=10)
    p.add_argument('--progress-every', type=int, default=0, help='Emit a flushed JSON summary every N steps; 0 disables.')
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
    if a.save_every_steps < 1 or a.progress_every < 0:
        p.error("Save interval must be positive and progress interval nonnegative.")
    state_paths = {path.resolve() for path in (a.resume_state, a.resume_from) if path}
    if any(path and path.resolve() in state_paths
           for path in (a.out, a.checkpoint_out, a.imagenet_checkpoint)):
        p.error("Resume state must not share a path with output JSON, final model or pretrained weights.")
    if (a.resume_state and a.resume_state.exists()
            and (not a.resume_from or a.resume_state.resolve() != a.resume_from.resolve())):
        p.error("Refusing to overwrite an existing checkpoint that was not selected with --resume-from.")
    if a.warmup is not None and a.warmup < 0:
        p.error("Warmup must be nonnegative.")
    if a.tau <= 0 or a.lr <= 0 or min(a.margin, a.cov_weight, a.consistency_weight) < 0:
        p.error("Invalid temperature, learning rate, or loss weight.")
    if a.fusion_dim is not None and a.fusion_dim < 2:
        p.error("Fusion width must be at least 2.")
    if a.eval_validation_at_limit and (
            a.validation_identities <= 0 or not a.market_root or a.skip_eval
            or a.max_steps is None or not state_paths):
        p.error("--eval-validation-at-limit requires --validation-identities > 0, --market-root, "
                "--max-steps and resume state; it cannot be used with --skip-eval.")
    if a.validation_identities < 0 or (a.validation_identities and not a.market_root):
        p.error("--validation-identities must be nonnegative and requires --market-root.")
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
