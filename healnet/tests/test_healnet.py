import pytest
import einops
from healnet.models import *
import torch


@pytest.fixture(autouse=True)
def vars():
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


def test_attention(vars):
    b, t_c, t_d, _, _, l_c, l_d, query, latent = vars
    # attention
    attention = Attention(query_dim=l_d, context_dim=t_d)
    # NOTE - traditional attention expectes the latent as the query and returns the latent
    # Problem is that this also means that the attention-matrix is at the latent level
    updated_latent = attention(x=latent, context=query)

    assert updated_latent.shape == (b, l_c, l_d)



def test_healnet(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent = vars

    tabular_data = torch.randn(b, t_d, t_c)  # expects (b dims channels)
    image_data = torch.randn(b, i_c, i_d)

    # unimodal case smoke test
    m1 = HealNet(modalities=1,
                 input_channels=[t_d],
                 input_axes=[1],  # second axis
                 num_classes=5
                 )
    logits1 = m1([tabular_data])
    assert logits1.shape == (b, 5)

    # bi-modal case
    m2 = HealNet(modalities=2,
                 input_channels=[t_c, i_c],  # level of attention
                 input_axes=[1, 1],
                 num_classes=4
                 )
    logits2 = m2([tabular_data, image_data])
    assert logits2.shape == (b, 4) # default num_classes

    # check misaligned args (1 mod but list of tensors)
    with pytest.raises(AssertionError):
        m3 = HealNet(modalities=1,
                     input_channels=[t_c, i_c],  # level of attention
                     input_axes=[1, 1],
                     )





