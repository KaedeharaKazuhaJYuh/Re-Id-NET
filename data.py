"""Datasets, deterministic identity batches, and standard single-gallery ReID metrics."""
from collections import defaultdict
import math
from pathlib import Path
import random
import re

from PIL import Image
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset, Sampler


class SyntheticReID(Dataset):
    """Procedural identity textures, not real people or a scientific benchmark."""

    def __init__(self, split="train", identities=8, instances=4, size=(64, 32), seed=0):
        if split not in {"train", "query", "gallery"}:
            raise ValueError(f"Unknown split: {split}")
        if min(identities, instances, *size) <= 0:
            raise ValueError("identities, instances, and image dimensions must be positive")
        self.split, self.size, self.seed = split, tuple(size), seed
        self.classes = identities
        self.pids = [pid for pid in range(identities)
                     for _ in range(instances if split != "query" else 1)]
        self.camids = ([1] * len(self.pids) if split == "query" else
                       [2 + i % 2 for i in range(len(self.pids))])

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid = self.pids[index]
        g = torch.Generator().manual_seed(self.seed + 1009 * pid)
        grid = torch.rand((1, 3, 4, 2), generator=g)
        x = torch.nn.functional.interpolate(grid, self.size, mode="bilinear",
                                             align_corners=False)[0]
        split_id = {"train": 0, "query": 1, "gallery": 2}[self.split]
        g.manual_seed(self.seed + 314159 + 1000003 * split_id + index)
        x = (x + 0.05 * torch.randn(x.shape, generator=g)).clamp(0, 1)
        return x, pid, self.camids[index]


class Market1501(Dataset):
    """Read only an existing Market-1501 installation; no downloading."""

    folders = {"train": "bounding_box_train", "query": "query", "gallery": "bounding_box_test"}
    pattern = re.compile(r"^(-?\d+)_c(\d+)")

    def __init__(self, root, split, size=(256, 128)):
        if split not in self.folders:
            raise ValueError(f"Unknown split: {split}")
        folder = Path(root).expanduser() / self.folders[split]
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected an existing Market-1501 split: {folder}")
        records = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            match = self.pattern.match(path.name)
            if match is None:
                raise ValueError(f"Cannot parse Market-1501 identity/camera from {path.name}")
            pid, camera = map(int, match.groups())
            if pid == -1:  # Official junk labels are not evaluated.
                continue
            records.append((path, pid, camera))
        if not records:
            raise ValueError(f"No usable Market-1501 images found in {folder}")
        relabel = {pid: i for i, pid in enumerate(sorted({x[1] for x in records}))}
        self.items = [(p, relabel[pid] if split == "train" else pid, camera)
                      for p, pid, camera in records]
        self.pids = [item[1] for item in self.items]
        self.camids = [item[2] for item in self.items]
        self.classes = len(relabel)
        self.size = tuple(size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, pid, camera = self.items[index]
        with Image.open(path) as im:
            image = im.convert("RGB").resize(self.size[::-1], Image.Resampling.BILINEAR)
            x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        x = x.reshape(*self.size, 3).permute(2, 0, 1).float().div_(255)
        return x, pid, camera


class PKBatchSampler(Sampler):
    """P identities × K observations, with replacement for underfull identity groups."""

    def __init__(self, labels, p, k, seed=0):
        self.p, self.k, self.seed, self.epoch = p, k, seed, 0
        self.start_batch = 0
        if p < 2 or k < 2:
            raise ValueError("Triplet batches require P >= 2 and K >= 2")
        self.groups = defaultdict(list)
        for i, label in enumerate(labels):
            self.groups[int(label)].append(i)
        if len(self.groups) < p:
            raise ValueError(f"Need at least P={p} distinct identities")
        self.steps = max(1, math.ceil(sum(map(len, self.groups.values())) / (p * k)))

    def set_epoch(self, epoch, start_batch=0):
        if not 0 <= start_batch <= self.steps:
            raise ValueError("Invalid sampler offset")
        self.epoch = int(epoch)
        self.start_batch = int(start_batch)

    def __len__(self):
        return self.steps - self.start_batch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        ids = sorted(self.groups)
        # Favor identities containing more images while covering each identity once at minimum
        # across the candidate pool; actual sampling is seeded and may repeat across batches.
        bags = {}
        for pid in ids:
            bags[pid] = self.groups[pid].copy()
            rng.shuffle(bags[pid])
        for i in range(self.steps):
            # Prioritize unused observations, and allow refills after a bag is exhausted.
            active = [pid for pid in ids if bags[pid]]
            selected = rng.sample(active, min(self.p, len(active)))
            if len(selected) < self.p:
                selected.extend(rng.sample([pid for pid in ids if pid not in selected],
                                           self.p - len(selected)))
            batch = []
            for pid in selected:
                bag = bags[pid]
                take = min(self.k, len(bag))
                chosen = [bag.pop() for _ in range(take)]
                pool = [idx for idx in self.groups[pid] if idx not in chosen]
                unique = min(self.k - len(chosen), len(pool))
                chosen.extend(rng.sample(pool, unique))
                chosen.extend(rng.choices(self.groups[pid], k=self.k - len(chosen)))
                batch.extend(chosen)
            if i >= self.start_batch:
                yield batch


def evaluate_reid(query_features, query_pids, query_camids,
                  gallery_features, gallery_pids, gallery_camids):
    """Cosine matching; remove same-ID/same-camera and junk gallery; valid-query average.

    mAP and rank1 are fractions, not percentages. No reranking is performed.
    """
    qf = torch.as_tensor(query_features).detach().float().cpu()
    gf = torch.as_tensor(gallery_features).detach().float().cpu()
    qp = torch.as_tensor(query_pids).cpu().reshape(-1)
    qc = torch.as_tensor(query_camids).cpu().reshape(-1)
    gp = torch.as_tensor(gallery_pids).cpu().reshape(-1)
    gc = torch.as_tensor(gallery_camids).cpu().reshape(-1)
    if qf.ndim != 2 or gf.ndim != 2 or qf.shape[1] != gf.shape[1]:
        raise ValueError("Features must be matrices with equal embedding dimension")
    if len(qf) != len(qp) or len(qf) != len(qc) or len(gf) != len(gp) or len(gf) != len(gc):
        raise ValueError("Feature, identity and camera counts must match")
    if not torch.isfinite(qf).all() or not torch.isfinite(gf).all():
        raise ValueError("Non-finite features cannot be evaluated")
    qf, gf = normalize(qf, dim=1), normalize(gf, dim=1)
    avg_precision, rank1 = [], []
    for start in range(0, len(qf), 128):
        similarities = qf[start:start + 128] @ gf.T
        orders = similarities.argsort(dim=1, descending=True, stable=True)
        for offset, order in enumerate(orders):
            i = start + offset
            if qp[i] == -1:
                continue
            keep = (gp[order] != -1) & ~((gp[order] == qp[i]) & (gc[order] == qc[i]))
            hits = gp[order[keep]] == qp[i]
            ranks = hits.nonzero().flatten() + 1
            if not ranks.numel():
                continue
            precision = torch.arange(1, len(ranks) + 1).float() / ranks
            avg_precision.append(float(precision.mean()))
            rank1.append(float(hits[0]))
    if not avg_precision:
        raise ValueError("No query identity has a valid cross-camera gallery match")
    return {"mAP": sum(avg_precision) / len(avg_precision),
            "rank1": sum(rank1) / len(rank1), "valid_queries": len(rank1)}
