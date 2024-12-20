#code to merge adata boject for ease of use to run in scVital

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import warnings

from scipy.sparse import csr_matrix


def mergeAdatas(adatas, homology=None, species=None, names=None, label="dataset"):
    """
    Merge multiple AnnData objects, inferring species and homology if not provided.

    Parameters
    ----------
    adatas : list of AnnData
        List of AnnData objects to be merged.
    homology : pandas.DataFrame, optional
        DataFrame containing homology information between species.
    species : list of str, optional
        List of species corresponding to each AnnData object.
    names : list of str, optional
        List of names for each AnnData object.
    label : str, optional
        Label for the concatenated AnnData object. Default is "dataset".

    Returns
    -------
    AnnData
        A concatenated AnnData object containing all input AnnData objects.
    """
    if species is None:
        warnings.warn("No species given, inferring them")
        species = []
        for adata in adatas:
            if "species" in adata.obs.columns:
                species.append(adata.obs["species"].unique()[0])
            else:
                if sum([gene.upper() == gene for gene in adata.var_names.values]) > 0.8 * len(adata.var_names):
                    species.append("human")
                    adata.obs["species"] = pd.Categorical(["human"] * len(adata.obs_names))
                if sum([gene[0].upper() + gene[1:].lower() == gene for gene in adata.var_names.values]) > 0.8 * len(adata.var_names):
                    species.append("mouse")
                    adata.obs["species"] = pd.Categorical(["mouse"] * len(adata.obs_names))
        warnings.warn(f"Species inferred: {species}")
    allGenes = set()
    for adata in adatas:
        if "species" not in adata.obs.columns:
            if sum([gene.upper() == gene for gene in adata.var_names.values]) > 0.8 * len(adata.var_names):
                adata.obs["species"] = pd.Categorical(["human"] * len(adata.obs_names))
            elif sum([gene[0].upper() + gene[1:].lower() == gene for gene in adata.var_names.values]) > 0.8 * len(adata.var_names):
                adata.obs["species"] = pd.Categorical(["mouse"] * len(adata.obs_names))
            else:
                warnings.warn("Not mouse or human and not given")
                adata.obs["species"] = pd.Categorical(["unk"] * len(adata.obs_names))
        allGenes.update(list(adata.var_names.values))

    if homology is None and len(set(species)) > 1:
        warnings.warn("No homology given, inferring them")
        if species is None:
            raise ValueError("Species information is required to infer homology")
        species = [sp.lower() for sp in species]
        if "mouse" in species and "human" in species:
            homology = pd.read_csv("../data/homology/MouseHumanHomology.csv")
        else:
            raise ValueError("Only mouse and human homology is known by default")
    elif len(set(species)) == 1:
        homology = pd.DataFrame({species[0]:list(allGenes)})
    geneSpecDict = []
    for i, adata in enumerate(adatas):
        try:
            if csr_matrix.max(adata.X) - csr_matrix.min(adata.X) > 20:
                warnings.warn("Data not log normalized, calculating now")
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                #warnings.warn(f"Post-normalization range: {csr_matrix.max(adata.X) - csr_matrix.min(adata.X)}")
        except:
            if np.max(adata.X) - np.min(adata.X) > 20:
                warnings.warn("Data not log normalized, calculating now")
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                #warnings.warn(f"Post-normalization range: {np.max(adata.X) - np.min(adata.X)}")
        geneSpecDict.append({"genes": np.array(adata.var_names.copy()), "species": species[i]})

    geneSpecDict = getOverlapGenesMulti(homology, geneSpecDict)
    for i, genesDict in enumerate(geneSpecDict):
        adatas[i].var_names = genesDict["genes"]

    if names is None:
        warnings.warn("No names given, inferring them")
        names = [f"adata{i}{specie}" for i, specie in enumerate(species)]
        warnings.warn(f"Names inferred: {names}")

    mAdata = ad.concat(adatas, join="outer", label=label, keys=[name for name in names], fill_value=0, index_unique="-")
    return mAdata

def getOverlapGenesMulti(homology, geneSpecDict, sep="/"):
    """
    Get overlapping genes across multiple species using homology information.

    Parameters
    ----------
    homology : pandas.DataFrame
        DataFrame containing homology information between species.
    geneSpecDict : list of dict
        List of dictionaries containing genes and species information.
    sep : str, optional
        Separator used to join gene names. Default is "/".

    Returns
    -------
    list of dict
        Updated list of dictionaries with overlapping genes.
    """
    for data in geneSpecDict:
        genes = data["genes"]
        species = data["species"]
        gene_121 = np.array(homology.loc[:, species.lower()])
        for i, gene in enumerate(genes):
            allGenes = gene
            if gene in gene_121:
                allGenes = sep.join(homology.loc[np.where(gene_121 == gene)[0][0], :])
            genes[i] = allGenes
    return geneSpecDict
