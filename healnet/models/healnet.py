from math import pi, log
from functools import wraps
from typing import *

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce



class HealNet(nn.Module):
    def __init__(
        self,
        *,
        n_modalities: int,
        channel_dims: List,
        num_spatial_axes: List, 
        out_dims: int,
        depth: int = 3,
        num_freq_bands: int = 2,
        max_freq: float=10.,
        l_c: int = 128,
        l_d: int = 128,
        x_heads: int = 8,
        l_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = False,
        fourier_encode_data: bool = True,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        snn: bool = True,
    ):
        """
        Network architecture for easy-to-use multimodal fusion for any number and type of modalities.
        
        The input for each modality should be of shape ``(b, (*spatial_dims) c)``, where ``c`` corresponds to the dimensions 
        where positional encoding does not matter (e.g., color channels, set-based features, or tabular features). 

        Args:
            n_modalities (int): Maximum number of modalities for forward pass. Note that fewer modalities can be passed
                if modalities for individual samples are missing (see ``.forward()``)
            channel_dims (List[int]): Number of channels or tokens for each modality. Length must match ``n_modalities``. 
                The channel_dims are non-spatial dimensions where positional encoding is not required. 
            num_spatial_axes (List[int]): Spatial axes for each modality.The each spatial axis will be assigned positional 
                encodings, so that ``num_spatial_axis`` is 2 for 2D images, 3 for Video/3D images. 
            out_dims (int): Output shape of task-specific head. Forward pass returns logits of this shape. 
            num_freq_bands (int, optional): Number of frequency bands for positional encodings. Defaults to 2.
            max_freq (float, optional): Maximum frequency for positional encoding. Defaults to 10.
            l_c (int, optional): Number of channels for latent bottleneck array (akin to a "learned query array"). Defaults to 128.
            l_d (int, optional): Dimensions for latent bottleneck. Defaults to 128.
            x_heads (int, optional): Number of heads for cross attention. Defaults to 8.
            l_heads (int, optional): Number of heads for latent attention. Defaults to 8.
            cross_dim_head (int, optional): Dimension of each cross attention head. Defaults to 64.
            latent_dim_head (int, optional): Dimension of each latent attention head. Defaults to 64.
            attn_dropout (float, optional): Dropout rate for attention layers. Defaults to 0.
            ff_dropout (float, optional): Dropout rate for feed-forward layers. Defaults to 0.
            weight_tie_layers (bool, optional): False for weight sharing between fusion layers, True for specific 
                weights for each layer. Note that the number of parameters will multiply by ``depth`` if True. 
                Defaults to False.
            fourier_encode_data (bool, optional): Whether to use positional encoding. Recommended if meaningful spatial 
                spatial structure should be preserved. Defaults to True.
            self_per_cross_attn (int, optional): Number of self-attention layers per cross-attention layer. Defaults to 1.
            final_classifier_head (bool, optional): Whether to include a final classifier head. Defaults to True.
            snn (bool, optional): Whether to use a self-normalizing network. Defaults to True.

        Example: 
        
            ```python
            from healnet import HealNet
            from healnet.etl import MMDataset
            import torch
            import einops

            # synthetic data example
            n = 100 # number of samples
            b = 4 # batch size
            img_c = 3 # image channels
            tab_c = 1 # tabular channels
            tab_d = 2000 # tabular features
            # 2D dims
            h = 224 # image height
            w = 224 # image width
            # 3d dim
            d = 12

            tab_tensor = torch.rand(size=(n, tab_c, tab_d)) 
            img_tensor_2d = torch.rand(size=(n, h, w, img_c)) # h w c
            img_tensor_3d = torch.rand(size=(n, d, h, w, img_c)) # d h w c
            dataset = MMDataset([tab_tensor, img_tensor_2d, img_tensor_3d])

            [tab_sample, img_sample_2d, img_sample_3d] = dataset[0]

            # batch dim for illustration purposes
            tab_sample = einops.repeat(tab_sample, 'c d -> b c d', b=1) # spatial axis: None (pass as 1)
            img_sample_2d = einops.repeat(img_sample_2d, 'h w c -> b h w c', b=1) # spatial axes: h w
            img_sample_3d = einops.repeat(img_sample_3d, 'd h w c -> b d h w c', b=1) # spatial axes: d h w

            tensors = [tab_sample, img_sample_2d, img_sample_3d]


            model = HealNet(
                        n_modalities=3, 
                        channel_dims=[2000, 3, 3], # (2000, 3, 3) number of channels/tokens per modality
                        num_spatial_axes=[1, 2, 3], # (1, 2, 3) number of spatial axes (will be positionally encoded to preserve spatial information)
                        out_dims = 4
                    )

            # example forward pass
            logits = model(tensors)
            ```
        """
        
        
        super().__init__()
        assert len(channel_dims) == len(num_spatial_axes), 'input channels and input axis must be of the same length'
        assert len(num_spatial_axes) == n_modalities, 'input axis must be of the same length as the number of modalities'

        self.input_axes = num_spatial_axes
        self.input_channels=channel_dims
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.modalities = n_modalities
        self.self_per_cross_attn = self_per_cross_attn

        self.fourier_encode_data = fourier_encode_data

        # get fourier channels and input dims for each modality
        fourier_channels = []
        input_dims = []
        for axis in num_spatial_axes:
            fourier_channels.append((axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0)
        for f_channels, i_channels in zip(fourier_channels, channel_dims):
            input_dims.append(f_channels + i_channels)


        # initialise shared latent bottleneck
        self.latents = nn.Parameter(torch.randn(l_c, l_d))

        # modality-specific attention layers
        funcs = []
        for m in range(n_modalities):
            funcs.append(lambda m=m: PreNorm(l_d, Attention(l_d, input_dims[m], heads = x_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dims[m]))
        cross_attn_funcs = tuple(map(cache_fn, tuple(funcs)))

        get_latent_attn = lambda: PreNorm(l_d, Attention(l_d, heads = l_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_cross_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))
        get_latent_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))

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
            for j in range(n_modalities):
                cross_attn_layers.append(cross_attn_funcs[j](**cache_args))
                cross_attn_layers.append(get_cross_ff(**cache_args))


            self.layers.append(nn.ModuleList(
                [*cross_attn_layers, self_attns])
            )

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(l_d),
            nn.Linear(l_d, out_dims)
        ) if final_classifier_head else nn.Identity()




    def forward(self,
                tensors: List[Union[torch.Tensor, None]],
                mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False, 
                verbose: bool = False
                ):

        missing_idx = [i for i, t in enumerate(tensors) if t is None]
        if verbose: 
            print(f"Missing modalities indices: {missing_idx}")
        for i in range(len(tensors)):
            if i in missing_idx: 
                continue
            else: 
                data = tensors[i]
                # sanity checks
                b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
                assert len(axis) == self.input_axes[i], (f'input data for modality {i+1} must hav'
                                                            f' the same number of axis as the input axis parameter')

            # fourier encode for each modality
            if self.fourier_encode_data:
                axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
                pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
                enc_pos = repeat(enc_pos, '... -> b ...', b = b)
                data = torch.cat((data, enc_pos), dim = -1)
                

            # concat and flatten axis for each modality
            data = rearrange(data, 'b ... d -> b (...) d')
            tensors[i] = data


        x = repeat(self.latents, 'n d -> b n d', b = b) # note: batch dim should be identical across modalities

        for layer_idx, layer in enumerate(self.layers):
            for i in range(self.modalities):
                if i in missing_idx: 
                    if verbose: 
                        print(f"Skipping update in fusion layer {layer_idx + 1} for missing modality {i+1}")
                        continue
                cross_attn=layer[i*2]
                cross_ff = layer[(i*2)+1]
                try:
                    x = cross_attn(x, context = tensors[i], mask = mask) + x
                    x =  cross_ff(x) + x
                except:
                    pass

                if self.self_per_cross_attn > 0:
                    self_attn, self_ff = layer[-1]

                    x = self_attn(x) + x
                    x = self_ff(x) + x

        if return_embeddings:
            return x

        return self.to_logits(x)

    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Helper function which returns all attention weights for all attention layers in the model
        Returns:
            all_attn_weights: list of attention weights for each attention layer
        """
        all_attn_weights = []
        for module in self.modules():
            if isinstance(module, Attention):
                all_attn_weights.append(module.attn_weights)
        return all_attn_weights



# HELPERS/UTILS
"""
Helper class implementations based on: https://github.com/lucidrains/perceiver-pytorch
"""


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

class GELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class SELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.selu(gates)

class RELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.relu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., snn: bool = False):
        super().__init__()
        activation = SELU() if snn else GELU()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            activation,
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def temperature_softmax(logits, temperature=1.0, dim=-1):
    """
    Temperature scaled softmax
    Args:
        logits:
        temperature:
        dim:

    Returns:
    """
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=dim)



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

        self.attn_weights = None
        # self._init_weights()

    def _init_weights(self):
    # Use He initialization for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # Initialize bias to zero if there's any
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        # attn = sim.softmax(dim = -1)
        attn = temperature_softmax(sim, temperature=0.5, dim=-1)
        self.attn_weights = attn
        attn = self.dropout(attn)


        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)