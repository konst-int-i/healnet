from healnet.models import *
import torch

def test_latent_cross_attention():
    b = 10
    # tabular
    # Note - dimensions always denoted as
    c_n = 1 # number of channels (1 for tabular)
    c_d = 2189 # dimensions of each channel
    l_n = 256 # number of latents
    l_d = 32 # latent dims
    # latent_dim
    query = torch.randn(b, c_n, c_d)
    context = torch.randn(b, l_n, l_d)

    attention_module = LatentCrossAttention(query_dim=c_n, latent_dim=l_d)
    context_prime = attention_module(query, context)

    assert context_prime.shape == (b, l_n, l_d)