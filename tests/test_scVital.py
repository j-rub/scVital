import pytest
import numpy as np
import pandas as pd
import anndata as an
from scipy.sparse import csr_matrix

import torch
import warnings

import scVital as scVt

@pytest.fixture
def setup_data():
    counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)

    adata = an.AnnData(counts)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    bt = np.random.choice(["b1", "b2"], size=(adata.n_obs,))
    adata.obs["batch"] = pd.Categorical(bt)  # Categoricals are preferred for efficiency

    ct = np.random.choice(["m", "h"], size=(adata.n_obs,))
    adata.obs["species"] = pd.Categorical(ct)  # Categoricals are preferred for efficiency

    return {
        'adata': adata,
        'batchLabel': 'batch',
        'miniBatchSize': 512,
        'numEpoch': 64,
        'learningRate': 3e-1,
        'hid1': 1024,
        'hid2': 128,
        'latentSize': 12,
        'discHid': 6,
        'reconCoef': 2e0,
        'klCoef': 1e-1,
        'discCoef': 1e0,
        'discIter': 5,
        'earlyStop': 1e-2,
        'train': False,
        'seed': 18,
        'verbose': True
    }

def test_invalid_adata(setup_data):
    with pytest.raises(ValueError):
        scVt.makeScVital(None, setup_data['batchLabel'])

def test_invalid_batchLabel(setup_data):
    with pytest.raises(ValueError):
        scVt.makeScVital(setup_data['adata'], 123)

def test_invalid_miniBatchSize(setup_data):
    with pytest.raises(ValueError):
        scVt.makeScVital(setup_data['adata'], setup_data['batchLabel'], miniBatchSize=-1)

def test_high_learningRate_warning(setup_data):
    with pytest.warns(UserWarning):
        scVt.makeScVital(setup_data['adata'], setup_data['batchLabel'], learningRate=2e0)

def test_valid_makeScVital(setup_data):
    model = scVt.makeScVital(**setup_data)
    assert isinstance(model, scVt.scVitalModel)

def test_scVital_initialization(setup_data):
    model = scVt.scVitalModel(
        setup_data['adata'], setup_data['batchLabel'], setup_data['miniBatchSize'], setup_data['numEpoch'], setup_data['learningRate'],
        setup_data['hid1'], setup_data['hid2'], setup_data['latentSize'], setup_data['discHid'], 
        setup_data['reconCoef'], setup_data['klCoef'], setup_data['discCoef'], setup_data['discIter'], 
        setup_data['earlyStop'], setup_data['seed'], setup_data['verbose']
    )
    assert csr_matrix.sum(model.adata.X) == csr_matrix.sum(setup_data['adata'].X)
    assert model.batchLabel == setup_data['batchLabel']
    assert model.miniBatchSize == setup_data['miniBatchSize']
    assert model.numEpoch == setup_data['numEpoch']
    assert model.learningRate == setup_data['learningRate']
    assert model.hid1 == setup_data['hid1']
    assert model.hid2 == setup_data['hid2']
    assert model.latentSize == setup_data['latentSize']
    assert model.discHid == setup_data['discHid']
    assert model.reconCoef == (len(model.adata)**0.5)*setup_data['reconCoef']
    assert model.klCoef == setup_data['klCoef']
    assert model.discCoef == setup_data['discCoef']
    assert model.discIter == setup_data['discIter']
    assert model.earlyStop == setup_data['earlyStop']
    assert model.seed == setup_data['seed']
    assert model.verbose == setup_data['verbose']
