# scVital

[![Latest PyPI Version][pb]][pypi] [![PyPI Downloads][db]][pypi] [![tests][gb]][yml] 

[gb]: https://github.com/j-rub/scVital/actions/workflows/publish.yml/badge.svg
[yml]: https://github.com/j-rub/scVital/actions/workflows/publish.yml
[pb]: https://img.shields.io/pypi/v/scVital.svg
[pypi]: https://pypi.org/project/scVital/

[db]: https://img.shields.io/pypi/dm/scVital?label=pypi%20downloads


![scVial workflow](https://github.com/j-rub/scVital/blob/main/images/scVitalWorkflow.png)


Read the pre-print! [A deep-learning tool for species-agnostic integration of cancer cell states
](https://www.biorxiv.org/content/10.1101/2024.12.20.629285v1)


# scVital

scVital is a powerful tool designed for the integration and analysis of single-cell RNA sequencing (scRNA-seq) data from multiple species. It leverages deep learning techniques to integrate datasets, enabling comprehensive comparative analyses and insights into conserved and species-specific cell states.


## Features

- **Cross-Species Data Integration**: Seamlessly integrate scRNA-seq data from different species.
- **Deep Learning Models**: Utilizes autoencoders and discriminators for effective data integration.
- **Comprehensive Evaluation Metrics**: Evaluate model performance using latent space similarity, UMAP visualization, cluster identification, and more.
- **User-Friendly Interface**: Easy to use with detailed documentation and examples.

## Installation

### From PyPI

To install scVital from PyPI, use the following command:

   ```bash
   pip install scVital
   ```

### From GitHub Release

To download and install scVital from a GitHub release, follow these steps:

1. **Navigate to the GitHub Repository**:
   - Go to the scVital GitHub repository.

2. **Go to the Releases Section**:
   - Click on the "Releases" tab, which is usually found on the right side of the repository's main page.

3. **Find the Desired Release**:
   - Browse through the list of releases and find the one you want to download. Releases are typically tagged with version numbers.

4. **Download the Release**:
   - Under the desired release, you will find assets such as `.zip` or `.tar.gz` files. Click on the appropriate file to download it.

5. **Install the Package**:
   - Once downloaded, you can install the package using `pip`. Navigate to the directory where the downloaded file is located and run the following command in your terminal:

     ```bash
     pip install path/to/downloaded/file.zip
     ```

     Replace `path/to/downloaded/file.zip` with the actual path to the downloaded file.

### Example

Let's say you want to download and install a package from a GitHub release:

1. **Navigate to the repository**: `https://github.com/j-rub/scVital`
2. **Go to the Releases section**: `https://github.com/j-rub/scVital/releases`
3. **Download the release**: Click on `scVital-1.0.0.zip` to download it.
4. **Install the package**:

   ```bash
   pip install ~/Downloads/scVital-1.0.0.zip
   ```

## Usage Example

```python

import numpy as np
import pandas as pd
import scanpy as sc
import scVital as sv

writeDir = "../data"
tissue = "muscle"

species1 = "human"
adataFile1 = f'{writeDir}/{tissue}Human_DeM_QC.h5ad'

species2 = "mouse"
adataFile2 = f'{writeDir}/{tissue}Mouse_DeM_QC.h5ad'

adata1 = sc.read_h5ad(adataFile1)
adata2 = sc.read_h5ad(adataFile2)

adata = sv.mg.mergeAdatas([adata1, adata2]) 

batchKey = "species"
sc.pp.highly_variable_genes(adata, batch_key=batchKey, n_top_genes=2000)
adata = adata[:, np.logical_and(adata.var.highly_variable, np.logical_not(adata.var.mt))]


setupData = {
    'adata': adata,
    'batchLabel': 'species',
    'miniBatchSize': 1024,
    'numEpoch': 50,
    'learningRate': 1e-3,
    'hid1': 1024,
    'hid2': 128,
    'latentSize': 12,
    'discHid': 6,
    'reconCoef': 2e0,
    'klCoef': 5e-2,
    'discCoef': 1e0,
    'discIter': 5,
    'earlyStop': 1e-2,
    'train': False,
    'seed': 18,
    'verbose': True
}

scVitalModel = sv.makeScVital(setupData['adata'], setupData['batchLabel'], setupData['miniBatchSize'], setupData['numEpoch'], setupData['learningRate'],
        setupData['hid1'], setupData['hid2'], setupData['latentSize'], setupData['discHid'], 
        setupData['reconCoef'], setupData['klCoef'], setupData['discCoef'], setupData['discIter'], 
        setupData['earlyStop'], setupData['train'], seed=setupData['seed'], verbose=setupData['verbose']
    )

scVitalModel.runTrainScVital()

adata = scVitalModel.getAdata()

umapKey = "scVitalModel"
neighborsKey = "scVitalModel"
pcaRep = "X_scVital"

sc.pp.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_pcs=scVitalModel.getLatentSize(), use_rep=pcaRep, key_added=umapKey)
sc.tl.umap(adata, neighbors_key = umapKey)
sc.tl.leiden(adata, resolution=0.1, key_added = neighborsKey, neighbors_key = umapKey)#, flavor="igraph", n_iterations=2,  directed=False)

sc.pl.umap(adata, color = ["species",neighborsKey,"cell_annotation"], ncols = 2)

```

## Future Improvements for scVital

1. **Additional Discriminator**:
   - Address inter-patient heterogeneity to reduce batch effects when integrating human data.
     - One discriminator to remove **patient** batch effect.
     - One discriminator to remove **species** batch effect.
   - Enhance integration when multiple human patients are involved.

2. **Use Reconstructed Output Data as Imputed Data**:
   - Utilize imputed gene expression data for further downstream analysis, including differential gene expression.
      - Overcome gene dropout in scRNA-seq data.

3. **Expand to Perform Cell Clustering**:
   - Add an output softmax clustering layer to the latent space.
       - Identify cell states without needing Leiden clustering after training.

4. **Utilize GPUs to Speed Up the Process**:
   - Implement GPU acceleration to enhance processing speed and efficiency.
       - Possibly use NVIDIA rapids 


README made with the help of copilot.
