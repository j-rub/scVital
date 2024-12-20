import pytest
import warnings

import numpy as np
import pandas as pd
import anndata as an
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance

# Assuming the functions are imported from lss.py
from scVital.lss import calcPairsLSS, calcLSS, calcAUC, calcClustDist, calcTotalDist

# Mock data for testing
@pytest.fixture
def mock_data():
	np.random.seed(1)

	counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
	latent = 'X_pca'
	batchName = 'batch'
	cellTypeLabel = 'cellType'

	adata = an.AnnData(counts)
	adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
	adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]

	bt = np.random.choice(["m", "h"], size=(adata.n_obs,))
	adata.obs[batchName] = pd.Categorical(bt)  # Categoricals are preferred for efficiency

	ct = np.random.choice(["ct1", "ct2", "ct3"], size=(adata.n_obs,))
	adata.obs[cellTypeLabel] = pd.Categorical(ct)  # Categoricals are preferred for efficiency

	latentSpace = np.random.normal(loc=0.0, scale=1.0, size=(len(adata),10))
	adata.obsm[latent] = latentSpace
	return adata, latent, batchName, cellTypeLabel

def test_calcPairsLSS(mock_data):
	adata, latent, batchName, cellTypeLabel = mock_data
	clustDist, lssAUC, totalDist, allCellTypes, ctPairs = calcPairsLSS(adata, latent, batchName, cellTypeLabel)
	assert np.sum(clustDist.values) == 31.8386
	assert np.round(lssAUC, decimals=4) == 0.5593
	assert totalDist == 2.3621
	assert allCellTypes == ['h~ct1', 'h~ct2', 'h~ct3', 'm~ct1', 'm~ct2', 'm~ct3']
