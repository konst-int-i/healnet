import pandas as pd
import torch as nn


def write_model(model, path):
    nn.save(model.state_dict(), path)

def read_model(path):
    model = nn.load(path)
    return model
