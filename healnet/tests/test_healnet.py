import pytest
import einops
from healnet.models import *
import torch


@pytest.fixture(autouse=True, scope="module")
def vars():
    b = 10
    # tabular
    # Note - dimensions always denoted as
    t_c = 1 # number of channels (1 for tabular) ; note that channels correspond to modality input/features
    t_d = 2189 # tabular dimensions
    i_c = 100 # number of images
    i_w = 224 # image dims
    i_h = 224    
    # i_d = 1024 # dimensions per patch
    l_c = 256 # number of latent channels (num_latents)
    l_d = 32 # latent dims
    # latent_dim
    query = torch.randn(b, t_c, t_d)
    latent = torch.randn(b, l_c, l_d)
    return b, t_c, t_d, i_c, i_h, i_w, l_c, l_d, query, latent


def test_attention(vars):
    b, t_c, t_d, i_c, i_h, i_w, l_c, l_d, query, latent = vars
    # attention
    attention = Attention(query_dim=l_d, context_dim=t_d)
    updated_latent = attention(x=latent, context=query)

    assert updated_latent.shape == (b, l_c, l_d)



def test_healnet(vars):
    b, t_c, t_d, i_c, i_h, i_w, l_c, l_d, query, latent = vars

    tabular_data = torch.randn(b, t_c, t_d)  # tabular data uses each feature as a token
    image_data = torch.randn(b, i_h, i_w, i_c) # each channel is a token, rest positional encoded features

    # unimodal case smoke test
    m1 = HealNet(n_modalities=1,
                 channel_dims=[t_d],
                 num_spatial_axes=[1],  # 
                 out_dims=5
                 )
    logits1 = m1([tabular_data])
    assert logits1.shape == (b, 5)

    # bi-modal case
    m2 = HealNet(n_modalities=2,
                 channel_dims=[t_d, i_c],  # correct deimansion
                 num_spatial_axes=[1, 2],
                 out_dims=4
                 )
    logits2 = m2([tabular_data, image_data])
    assert logits2.shape == (b, 4) # default num_classes

    print(m2)

    # check misaligned args (1 mod but list of tensors)
    with pytest.raises(AssertionError):
        m3 = HealNet(n_modalities=1,
                     channel_dims=[t_d, i_c],  # level of attention
                     num_spatial_axes=[1, 1],
                     out_dims=4,
                     )