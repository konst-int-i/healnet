import pytest
import einops
from healnet.models import *
import torch


@pytest.fixture(autouse=True)
def shape_test_vars():
    b = 10
    # tabular
    # Note - dimensions always denoted as
    t_c = 1 # number of channels (1 for tabular) ; note that channels correspond to modality input/features
    t_d = 2189 # dimensions of each channel
    i_c = 100 # number of patches
    i_d = 1024 # dimensions per patch
    l_c = 256 # number of latent channels (num_latents)
    l_d = 32 # latent dims
    # latent_dim
    query = torch.randn(b, t_c, t_d)
    latent = torch.randn(b, l_c, l_d)
    return b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent


def test_latent_cross_attention(shape_test_vars):

    b, t_c, t_d, _, _, l_c, l_d, query, latent = shape_test_vars
    print("\nquery", query.shape)
    print("context", latent.shape)


    # latent cross attention
    lc_attention = LatentCrossAttention(query_dim=t_c, latent_dim=l_d)
    lc_context = lc_attention(query, latent)
    # expect updated context to be of original context shape
    assert lc_context.shape == (b, l_c, l_d)
    # attn weights: (b, x_d) (feature-dim)


def test_attention_update(shape_test_vars):

    b, t_c, t_d, _, _, l_c, l_d, query, latent = shape_test_vars
    attention_update = AttentionUpdate(c_n=t_d, l_d=l_d)
    updated_context = attention_update(x=query, context=latent)
    assert updated_context.shape == (b, l_c, l_d)
    # attn weights: (b, x_d)

def test_attention(shape_test_vars):
    b, t_c, t_d, _, _, l_c, l_d, query, latent = shape_test_vars
    # attention
    attention = Attention(query_dim=l_d, context_dim=t_d)
    # NOTE - traditional attention expectes the latent as the query and returns the latent
    # Problem is that this also means that the attention-matrix is at the latent level
    updated_latent = attention(x=latent, context=query)

    assert updated_latent.shape == (b, l_c, l_d)
    # attn weights: (b, l_c)


# def test_healnet(shape_test_vars):
#     b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent = shape_test_vars
#
#     tabular_data = torch.randn(b,t_d, t_c) # expects (b dims channels)
#     image_data = torch.randn(b, i_c, i_d)
#     # print(t_n)
#
#     healnet = HealNet(modalities=1,
#                       input_channels=[t_d],
#                       input_axes=[2], # second axis
#                       max_freq=8.,
#                       depth=1,
#                       num_classes=4,
#                       )
#
#     logits = healnet([tabular_data])






