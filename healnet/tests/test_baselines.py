import pytest
import torch
import einops
import torch.nn as nn
from healnet.baselines.multimodn import ResNet, MLPEncoder, ClassDecoder, PatchEncoder, MultiModN
from healnet.baselines.multimodn.better_multimodn import MultiModNModule
from typing import *


@pytest.fixture(autouse=True, scope="module")
def vars():
    b = 1
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

    tab_tensor = torch.randn(b, t_d, t_c)  # expects (b dims channels)
    img_tensor = torch.randn(b, i_c, i_d)
    return b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, tab_tensor, img_tensor


def test_multmodn_encoders(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, tab_tensor, img_tensor = vars

    # init state
    # only allows 1D stat
    state = nn.Parameter(torch.randn(b, l_d), requires_grad=True) #  doesn't allow channels :(

    # expects raw image
    img_tensor = torch.randn(b, 3, 255, 255) # standard resnet18 dims

    # image encoder pass
    img_enc = ResNet(
        state_size=l_d,
        )
    upd_state = img_enc(state=state, x=img_tensor)

    assert upd_state.shape == (b, l_d)

    # # patched image
    patch_tensor = torch.randn(b, i_c, i_d)
    # for patched images, it makes sense to use their RNN encoder
    patch_enc = PatchEncoder(
        state_size=l_d,
        hidden_layers=[128, 64],
        n_features=i_d,
    )

    upd_state = patch_enc(state=state, x=patch_tensor)

    assert upd_state.shape == (b, l_d)


    # omic
    omic_tensor = torch.randn(b, t_d)
    omic_enc = MLPEncoder(
        state_size=l_d,
        hidden_layers=[128, 64],
        n_features=t_d,
    )
    upd_state = omic_enc(state=state, x=omic_tensor)
    assert upd_state.shape == (b, l_d)

#
def test_multimodn_decoders(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, tab_tensor, img_tensor = vars

    # latent

    latent = nn.Parameter(torch.randn(b, l_d))
    # the "decoders" are actually task-specific FF classifier heads
    head = ClassDecoder(state_size=l_d,
                        activation=torch.sigmoid,
                        n_classes=4
                        )

    logits = head(latent)
    assert logits.shape == (b, 4)

def test_multimodn_task(vars):
    b, t_c, t_d, i_c, i_d, l_c, l_d, query, latent, tab_tensor, img_tensor = vars

    # ModN model spec
    tab_tensor = torch.randn(b, t_d)
    encoders = [
        MLPEncoder(state_size=l_d, hidden_layers=[128, 64], n_features=t_d),
        PatchEncoder(state_size=l_d, hidden_layers=[128, 64], n_features=i_d)
    ]
    decoders = [ClassDecoder(state_size=l_d, activation=torch.sigmoid, n_classes=4)]

    target = torch.rand(4)

    model = MultiModNModule(
        state_size=l_d,
        encoders=encoders,
        decoders=decoders,
    )

    loss, logits = model(x=[tab_tensor, img_tensor], target=target)

    assert logits.shape == (b, 4)
