import pytest
import pandas as pd
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
import scanpy as sc

from scVital.merging import mergeAdatas

@pytest.fixture
def adata_human():
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(10)])
    var = pd.DataFrame(index=[f"GENE_{i}" for i in range(1000)])
    X = np.random.rand(10, 1000)
    return AnnData(X=X, obs=obs, var=var)

@pytest.fixture
def adata_mouse():
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(10)])
    var = pd.DataFrame(index=[f"Gene_{i}" for i in range(1000)])
    X = np.random.rand(10, 1000)
    return AnnData(X=X, obs=obs, var=var)

@pytest.fixture
def homology():
    data = {
        'human': [f"GENE_{i}" for i in range(1000)],
        'mouse': [f"Gene_{i}" for i in range(1000)]
    }
    return pd.DataFrame(data)


def test_merge_adatas(adata_human, adata_mouse, homology):
    adatas = [adata_human, adata_mouse]
    species = ["human", "mouse"]
    names = ["inHuman", "inMouse"]
    #data = {
    #    'human': [f"GENE_{i}" for i in range(1000)],
    #    'mouse': [f"Gene_{i}" for i in range(1000)]
    #}
    #homology = pd.DataFrame(data)

    merged_adata = mergeAdatas(adatas, homology=homology, species=species, names=names, label="dataset")
    
    assert merged_adata is not None
    assert merged_adata.obs_names.equals(pd.Index([f"cell_{i}-inHuman" for i in range(10)] + [f"cell_{i}-inMouse" for i in range(10)]))
    assert "species" in merged_adata.obs.columns
    assert merged_adata.obs["species"].unique().tolist() == ["human", "mouse"]
    assert "dataset" in merged_adata.obs.columns
    assert merged_adata.obs["dataset"].unique().tolist() == ["inHuman", "inMouse"]

def testMergeAdatasSameSpec(adata_human):
    adatas = [adata_human, adata_human]

    with pytest.warns(UserWarning, match="No species given, inferring them"):
        merged_adata = mergeAdatas(adatas)
    
    assert merged_adata is not None
    assert "dataset" in merged_adata.obs.columns
    assert merged_adata.obs["dataset"].unique().tolist() == ["adata0human", "adata1human"]
    assert "species" in merged_adata.obs.columns
    assert merged_adata.obs["species"].unique().tolist() == ["human"]

def test_merge_adatas_no_species(adata_human, adata_mouse, homology):
    adatas = [adata_human, adata_mouse]
    #data = {
    #    'human': [f"GENE_{i}" for i in range(1000)],
    #    'mouse': [f"Gene_{i}" for i in range(1000)]
    #}
    #homology = pd.DataFrame(data)

    with pytest.warns(UserWarning, match="No species given, inferring them"):
        merged_adata = mergeAdatas(adatas, homology=homology)
    
    assert merged_adata is not None
    assert "species" in merged_adata.obs.columns

#def test_merge_adatas_no_homology(adata_human, adata_mouse):
#    adatas = [adata_human, adata_mouse]
#    species = ["m1", "m2"]
#    
#    with pytest.warns(UserWarning, match="No homology given, inferring them"):
#        merged_adata = mergeAdatas(adatas, species=species)
#    
#    assert merged_adata is not None
#    assert "species" in merged_adata.obs.columns
