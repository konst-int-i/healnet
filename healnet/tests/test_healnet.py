import pytest
import einops
from healnet.models import *
import torch


@pytest.fixture(autouse=True)
def shape_test_vars():
    b = 10
    # tabular
    # Note - dimensions always denoted as
    c_n = 1 # number of channels (1 for tabular) ; note that channels correspond to modality input/features
    c_d = 2189 # dimensions of each channel
    l_n = 256 # number of latents
    l_d = 32 # latent dims
    # latent_dim
    query = torch.randn(b, c_n, c_d)
    latent = torch.randn(b, l_n, l_d)
    return b, c_n, c_d, l_n, l_d, query, latent


def test_latent_cross_attention(shape_test_vars):

    b, c_n, c_d, l_n, l_d, query, latent = shape_test_vars
    print("\nquery", query.shape)
    print("context", latent.shape)


    # latent cross attention
    lc_attention = LatentCrossAttention(query_dim=c_n, latent_dim=l_d)
    lc_context = lc_attention(query, latent)
    # expect updated context to be of original context shape
    assert lc_context.shape == (b, l_n, l_d)


def test_attention_update(shape_test_vars):

    b, c_n, c_d, l_n, l_d, query, latent = shape_test_vars
    # query = einops.rearrange(query, "b c_n c_d -> b c_d c_n")
    # attention update
    attention_update = AttentionUpdate(c_n=c_d, l_d=l_d)
    updated_context = attention_update(x=query, context=latent)
    assert updated_context.shape == (b, l_n, l_d)

def test_attention(shape_test_vars):
    b, c_n, c_d, l_n, l_d, query, latent = shape_test_vars
    # attention
    attention = Attention(query_dim=l_d, context_dim=c_d)
    # NOTE - traditional attention expectes the latent as the query and returns the latent
    # Problem is that this also means that the attention-matrix is at the latent level
    updated_latent = attention(x=latent, context=query)

    assert updated_latent.shape == (b, l_n, l_d)


