#made with the help of copilot

import pytest
import numpy as np
import pandas as pd
import anndata as an
from scipy.sparse import csr_matrix

import torch
import warnings

import scVital as scVt

@pytest.fixture
def setupData():
    counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)

    adata = an.AnnData(counts)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    bt = np.random.choice(["b1", "b2"], size=(adata.n_obs,))
    adata.obs["batch"] = pd.Categorical(bt)  # Categoricals are preferred for efficiency

    ct = np.random.choice(["ct1", "ct2"], size=(adata.n_obs,))
    adata.obs["cellType"] = pd.Categorical(ct)  # Categoricals are preferred for efficiency

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

def test_invalid_adata(setupData):
    with pytest.raises(ValueError):
        scVt.makeScVital(None, setupData['batchLabel'])

def test_invalid_batchLabel(setupData):
    with pytest.raises(ValueError):
        scVt.makeScVital(setupData['adata'], 123)

def test_invalid_miniBatchSize(setupData):
    with pytest.raises(ValueError):
        scVt.makeScVital(setupData['adata'], setupData['batchLabel'], miniBatchSize=-1)

def test_high_learningRate_warning(setupData):
    with pytest.warns(UserWarning):
        scVt.makeScVital(setupData['adata'], setupData['batchLabel'], learningRate=2e0)

def test_valid_makeScVital(setupData):
    model = scVt.makeScVital(**setupData)
    assert isinstance(model, scVt.scVitalModel)

def test_scVital_initialization(setupData):
    model = scVt.scVitalModel(
        setupData['adata'], setupData['batchLabel'], setupData['miniBatchSize'], setupData['numEpoch'], setupData['learningRate'],
        setupData['hid1'], setupData['hid2'], setupData['latentSize'], setupData['discHid'], 
        setupData['reconCoef'], setupData['klCoef'], setupData['discCoef'], setupData['discIter'], 
        setupData['earlyStop'], setupData['seed'], setupData['verbose']
    )
    assert csr_matrix.sum(model.getAdata().X) == csr_matrix.sum(setupData['adata'].X)
    assert model.getBatchLabel() == setupData['batchLabel']
    assert model.getMiniBatchSize() == setupData['miniBatchSize']
    assert model.getNumEpoch() == setupData['numEpoch']
    assert model.getLearningRate() == setupData['learningRate']
    assert model.getLayerDims() == [len(setupData['adata'].var_names), setupData['hid1'], setupData['hid2'], setupData['latentSize']]
    assert model.getLatentSize() == setupData['latentSize']
    assert model.getDiscDims() == [setupData['latentSize'], setupData['discHid'], 2] 
    assert model.getReconCoef() == (len(setupData['adata'])**0.5)*setupData['reconCoef']
    assert model.getKlCoef() == setupData['klCoef']
    assert model.getDiscCoef() == setupData['discCoef']
    assert model.getDiscIter() == setupData['discIter']
    assert model.getEarlyStop() == setupData['earlyStop']
    #assert model.seed == setupData['seed']
    #assert model.verbose == setupData['verbose']
