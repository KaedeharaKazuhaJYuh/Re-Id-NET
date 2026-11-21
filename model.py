"""Independent reconstruction of Zhou et al., not authors' code."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveMixer(nn.Module):
    """Portable selective-scan reference for synthetic smoke tests."""
    def __init__(self, dim, state=8):
        super().__init__()
        half = dim // 2
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.conv_x = nn.Conv1d(half, half, 3, padding=1, groups=half)
        self.conv_z = nn.Conv1d(half, half, 3, padding=1, groups=half)
        n_dt = max(1, math.ceil(dim/16))
        self.params = nn.Linear(half, n_dt + 2*state, bias=False)
        self.delta = nn.Linear(n_dt, half)
        self.delta.bias.data.fill_(-3)
        self.log_a = nn.Parameter(torch.arange(1, state+1).float().log().repeat(half, 1))
        self.skip = nn.Parameter(torch.ones(half))
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, seq):
        x, z = self.in_proj(seq).transpose(1, 2).chunk(2, dim=1)
        x = F.silu(self.conv_x(x)).transpose(1, 2)
        z = F.silu(self.conv_z(z)).transpose(1, 2)
        dt, b, c = self.params(x).split((self.delta.in_features, self.log_a.shape[1],
                                         self.log_a.shape[1]), dim=-1)
        dt = F.softplus(self.delta(dt))
        a = -self.log_a.exp()
        hidden = x.new_zeros(x.shape[0], x.shape[-1], a.shape[-1])
        ys = []
        for t in range(x.shape[1]):
            step = dt[:, t, :, None]
            hidden = (step*a).exp()*hidden + step*x[:, t, :, None]*b[:, t, None, :]
            ys.append((hidden*c[:, t, None, :]).sum(-1))
        return self.out(torch.cat((torch.stack(ys, dim=1) + x*self.skip, z), dim=-1))


class TokenBlock(nn.Module):
    def __init__(self, channels, mixer):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels*4), nn.GELU(),
                                 nn.Linear(channels*4, channels))

    def forward(self, x):
        y = self.norm1(x)
        if isinstance(self.mixer, nn.MultiheadAttention):
            y = self.mixer(y, y, y, need_weights=False)[0]
        else:
            y = self.mixer(y)
        x = x + y
        return x + self.mlp(self.norm2(x))


def conv_bn(a, b, ks=3, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, ks, stride=stride, padding=ks//2, bias=False),
                         nn.BatchNorm2d(b))


class LocalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(conv_bn(dim, dim), nn.GELU(), conv_bn(dim, dim))

    def forward(self, x):
        return x + self.body(x)


class ReferenceBackbone(nn.Module):
    channels = (16, 24, 32, 48)
    def __init__(self):
        super().__init__()
        a, b, c, d = self.channels
        self.stem = nn.Sequential(conv_bn(3, 8, stride=2), nn.GELU(),
                                  conv_bn(8, a, stride=2), nn.GELU())
        self.f1 = LocalBlock(a)
        self.f2 = nn.Sequential(conv_bn(a, b, stride=2), LocalBlock(b))
        self.down = conv_bn(b, c, stride=2)
        self.seq3 = self.blocks(c)
        self.proj = nn.Conv2d(c, d, 1, bias=False)
        self.seq4 = self.blocks(d)

    @staticmethod
    def blocks(dim):
        return nn.Sequential(TokenBlock(dim, SelectiveMixer(dim)),
                             TokenBlock(dim, nn.MultiheadAttention(dim, 4, batch_first=True)))

    @staticmethod
    def apply_tokens(fn, x):
        return fn(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(x)

    def forward(self, x):
        f1 = self.f1(self.stem(x))
        f2 = self.f2(f1)
        f3 = self.apply_tokens(self.seq3, self.down(f2))
        f4 = self.apply_tokens(self.seq4, self.proj(f3))
        return f1, f2, f3, f4


class OfficialTinyAdapter(nn.Module):
    channels = (80, 160, 320, 512)
    def __init__(self, checkpoint=None):
        super().__init__()
        try:
            from mambavision.models.mamba_vision import mamba_vision_T, MambaVisionLayer
        except ImportError as exc:
            raise RuntimeError("Install NVlabs MambaVision and its CUDA selective-scan dependencies.") from exc
        orig = mamba_vision_T(pretrained=False)
        if checkpoint:
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for key in ("state_dict", "model"):
                if key in state:
                    state = state[key]
                    break
            state = {k.removeprefix("module."): v for k, v in state.items()}
            expected = orig.state_dict()
            matching = {k: v for k, v in state.items()
                        if k in expected and expected[k].shape == v.shape}
            missing = set(expected) - set(matching)
            need = ("patch_embed.", "levels.0.", "levels.1.", "levels.2.blocks.")
            if any(k.startswith(need) and not k.endswith("num_batches_tracked") for k in missing):
                raise ValueError("Incompatible ImageNet checkpoint; required Tiny tensors are absent.")
            orig.load_state_dict(matching, strict=False)
        self.stem = orig.patch_embed
        self.stages = nn.ModuleList(list(orig.levels[:3]))
        self.downs = nn.ModuleList([self.stages[0].downsample, self.stages[1].downsample])
        for stage in self.stages:
            stage.downsample = None
        self.proj = nn.Conv2d(320, 512, 1, bias=False)
        self.stage4 = MambaVisionLayer(dim=512, depth=4, num_heads=16, window_size=7,
                                       downsample=False, transformer_blocks=[2, 3])
        self.norm = nn.BatchNorm2d(512)

    def forward(self, x):
        f1 = self.stages[0](self.stem(x))
        f2 = self.stages[1](self.downs[0](f1))
        f3 = self.stages[2](self.downs[1](f2))
        f4 = self.norm(self.stage4(self.proj(f3)))
        return f1, f2, f3, f4


def depth_sep(channels, k, dilation=1):
    return nn.Sequential(nn.Conv2d(channels, channels, k, padding=dilation*(k//2),
                                   groups=channels, dilation=dilation, bias=False),
                         nn.Conv2d(channels, channels, 1, bias=False))


class SASF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.branches = nn.ModuleList([depth_sep(c, 3, 2), depth_sep(c, 5, 2)])
        self.weights = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        weight = self.weights.softmax(0)
        return x + sum(w*f(x) for w,f in zip(weight, self.branches))


class MFM(nn.Module):
    def __init__(self, cin_left, cin_right, c, depth, shallow, sasf, mixer_factory):
        super().__init__()
        self.left = nn.Conv2d(cin_left, c, 1)
        self.right = nn.Conv2d(cin_right, c, 1)
        self.left_clean = depth_sep(c, 3) if shallow else None
        self.right_clean = depth_sep(c, 3) if shallow else None
        self.interact = nn.Sequential(*(mixer_factory(2*c) for _ in range(depth)))
        self.delta_left = nn.Conv2d(2*c, c, 1)
        self.delta_right = nn.Conv2d(2*c, c, 1)
        self.spatial = SASF(c) if sasf else nn.Identity()

    def forward(self, a, b, size):
        a = self.left(F.adaptive_avg_pool2d(a, size))
        b = self.right(F.adaptive_avg_pool2d(b, size))
        if self.left_clean is not None:
            a = a + self.left_clean(a)
            b = b + self.right_clean(b)
        cat = torch.cat((a, b), 1)
        z = self.interact(cat.flatten(2).transpose(1,2)).transpose(1,2).reshape_as(cat)
        return self.spatial((a+self.delta_left(z)) + (b+self.delta_right(z)))


class Head(nn.Module):
    def __init__(self, c, classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)
        self.bn.bias.requires_grad_(False)
        self.fc = nn.Linear(c, classes, bias=False)

    def forward(self, feature):
        v = feature.mean((2,3))
        b = self.bn(v)
        return (v, self.fc(b)), b


class ReID(nn.Module):
    def __init__(self, classes, backend="reference", checkpoint=None, hcen=True,
                 sasf=True, aux=True, fusion_dim=None, depths=(1,2,3)):
        super().__init__()
        if backend == "official-tiny":
            self.backbone = OfficialTinyAdapter(checkpoint)
            from mambavision.models.mamba_vision import MambaVisionMixer
            mix = lambda d: MambaVisionMixer(d_model=d, d_state=8, d_conv=3, expand=1)
        elif backend == "reference":
            if checkpoint is not None:
                raise ValueError("The synthetic reference cannot use ImageNet checkpoints.")
            self.backbone = ReferenceBackbone()
            mix = SelectiveMixer
        else:
            raise ValueError(f"Unknown backend: {backend}")
        c = fusion_dim or (512 if backend == "official-tiny" else 16)
        self.backbone_head = Head(self.backbone.channels[-1], classes)
        self.hcen = hcen
        self.mfm = nn.ModuleList()
        self.aux_heads = nn.ModuleList()
        if hcen:
            ch = self.backbone.channels
            self.mfm.extend(MFM(a,b,c,n,i==0,sasf,mix)
                            for i,(a,b,n) in enumerate(zip((ch[0],c,c),ch[1:],depths)))
            self.fusion_head = Head(c, classes)
            if aux:
                self.aux_heads.extend(Head(c, classes) for _ in range(2))

    def forward(self, x):
        feats = self.backbone(x)
        main, eb = self.backbone_head(feats[-1])
        result = {"backbone": main, "fusion": None, "aux": (), "retrieval": {"backbone": eb}}
        if self.hcen:
            g = feats[0]
            inter = []
            for layer, f in zip(self.mfm, feats[1:]):
                g = layer(g, f, feats[-1].shape[-2:])
                inter.append(g)
            fusion, ef = self.fusion_head(g)
            result.update(fusion=fusion, aux=tuple(h(v)[0] for h,v in zip(self.aux_heads, inter)))
            result["retrieval"].update(fusion=ef,
                combined=torch.cat((F.normalize(eb, dim=1), F.normalize(ef, dim=1)), 1))
        return result
