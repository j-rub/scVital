import pytest
import anndata as an
import torch
import warnings

import scVital as scVt

@pytest.fixture
def setup_data():
    adata = an.AnnData(X=[[1, 2], [3, 4]], obs={'batch': ['A', 'B']})
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
        'reconCoef': 1,
        'klCoef': 1e-1,
        'discCoef': 1,
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
        scVt.makeScVital(setup_data['adata'], setup_data['batchLabel'], learningRate=2)

def test_valid_makeScVital(setup_data):
    model = scVt.makeScVital(**setup_data)
    assert isinstance(model, scVital)

def test_scVital_initialization(setup_data):
    model = scVt.scVitalModel(
        setup_data['adata'], setup_data['batchLabel'], setup_data['miniBatchSize'], setup_data['numEpoch'], setup_data['learningRate'],
        setup_data['hid1'], setup_data['hid2'], setup_data['latentSize'], setup_data['discHid'], setup_data['reconCoef'], setup_data['klCoef'],
        setup_data['discCoef'], setup_data['discIter'], setup_data['earlyStop'], setup_data['seed'], setup_data['verbose']
    )
    assert model.adata == setup_data['adata']
    assert model.batchLabel == setup_data['batchLabel']
    assert model.miniBatchSize == setup_data['miniBatchSize']
    assert model.numEpoch == setup_data['numEpoch']
    assert model.learningRate == setup_data['learningRate']
    assert model.hid1 == setup_data['hid1']
    assert model.hid2 == setup_data['hid2']
    assert model.latentSize == setup_data['latentSize']
    assert model.discHid == setup_data['discHid']
    assert model.reconCoef == setup_data['reconCoef']
    assert model.klCoef == setup_data['klCoef']
    assert model.discCoef == setup_data['discCoef']
    assert model.discIter == setup_data['discIter']
    assert model.earlyStop == setup_data['earlyStop']
    assert model.seed == setup_data['seed']
    assert model.verbose == setup_data['verbose']
