"""
Based on lucidrains implementation: https://github.com/lucidrains/perceiver-pytorch
"""

from math import pi, log
from functools import wraps
from typing import *

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class SELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.selu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            SELU(),
            # GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        # add leaky relu
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.LeakyReLU(negative_slope=1e-2)
        )

        # self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class HealNet(nn.Module):
    def __init__(
        self,
        *,
        modalities: int,
        num_freq_bands: int,
        depth: int,
        max_freq: float,
        input_channels: List,
        input_axes: List,
        num_latents: int = 512,
        latent_dim: int = 512,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        num_classes: int = 1000,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = False,
        fourier_encode_data: bool = True,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True
    ):
        super().__init__()
        assert len(input_channels) == len(input_axes), 'input channels and input axis must be of the same length'
        assert len(input_axes) == modalities, 'input axis must be of the same length as the number of modalities'

        self.input_axes = input_axes
        self.input_channels=input_channels
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.modalities = modalities
        self.self_per_cross_attn = self_per_cross_attn

        self.fourier_encode_data = fourier_encode_data

        # get fourier channels and input dims for each modality
        fourier_channels = []
        input_dims = []
        for axis in input_axes:
            fourier_channels.append((axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0)
        for f_channels, i_channels in zip(fourier_channels, input_channels):
            input_dims.append(f_channels + i_channels)


        # initialise shared latent bottleneck
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # modality-specific attention layers
        funcs = []
        for m in range(modalities):
            funcs.append(lambda m=m: PreNorm(latent_dim, Attention(latent_dim, input_dims[m], heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dims[m]))
        cross_attn_funcs = tuple(map(cache_fn, tuple(funcs)))

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])


        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(get_latent_attn(**cache_args, key = block_ind))
                self_attns.append(get_latent_ff(**cache_args, key = block_ind))


            cross_attn_layers = []
            for j in range(modalities):
                cross_attn_layers.append(cross_attn_funcs[j](**cache_args))
                cross_attn_layers.append(get_cross_ff(**cache_args))

            # print(f"Layer {i+1} module list: ")
            # print(nn.ModuleList([*cross_attn_layers, self_attns]))

            self.layers.append(nn.ModuleList(
                [*cross_attn_layers, self_attns])
            )

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()


    def forward(self,
                tensors: List[torch.Tensor],
                mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False
                ):

        for i in range(self.modalities):
            data = tensors[i]
            # sanity checks
            b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
            assert len(axis) == self.input_axes[i], (f'input data for modality {i+1} must hav'
                                                          f' the same number of axis as the input axis parameter')

            # fourier encode for each modality
            if self.fourier_encode_data:
                pos = torch.linspace(0, 1, axis[0], device = device, dtype = dtype)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, 'n d -> () n d')
                enc_pos = repeat(enc_pos, '() n d -> b n d', b = b)
                data = torch.cat((data, enc_pos), dim = -1)

            # concat and flatten axis for each modality
            data = rearrange(data, 'b ... d -> b (...) d')
            tensors[i] = data


        x = repeat(self.latents, 'n d -> b n d', b = b) # note: batch dim should be identical across modalities

        for layer in self.layers:
            for i in range(self.modalities):
                cross_attn= layer[i*2]
                cross_ff = layer[(i*2)+1]
                x = cross_attn(x, context = tensors[i], mask = mask) + x
                x =  cross_ff(x) + x

            if self.self_per_cross_attn > 0:
                self_attn, self_ff = layer[-1]

                x = self_attn(x) + x
                x = self_ff(x) + x

        if return_embeddings:
            return x

        return self.to_logits(x)



















