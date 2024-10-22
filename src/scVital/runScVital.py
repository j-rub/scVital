#!/usr/bin/env python
#evaluate the DL model on simulated data

import sys
import os
import time
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import isspmatrix

import matplotlib.pyplot as plt
import matplotlib.image as img

import pdb
import torch

import FuncVizBench as vb
import AutoEncoder as ae
import Discriminator as dis 

inAdataFile = snakemake.input["inAdata"]
inVAEFile = snakemake.input["outVAE"]
inLossDict = snakemake.input["lossDict"]

outAdataFile = snakemake.output["outAdata"]

#print(snakemake.params)

paramDict = snakemake.params["paramVals"]

batchLabel = paramDict["batchName"]

scVitalLatentOutFile= snakemake.output["scVitalLatent"]
scVitalClusterOutFile=snakemake.output["scVitalCluster"]

try:
	cellLabel = paramDict["cellName"]
except:
	cellLabel = ""

#numOutLayer = paramDict["lastLayer"]

try:
	res = float(paramDict["res"])
except:
	res= 0.3
#nNeighborsUMAP = paramDict["nNeighborsUMAP"]
#inters = paramDict["inters"]
#cellsPerIter = paramDict["cellsPerIter"]
#nNeighborsKbet = paramDict["nNeighborsKbet"]

allVars = f"inAdataFile: {inAdataFile} \n\
	outAdataFile: {outAdataFile} \n\
	batchLabel: {batchLabel} \n\
	cellLabel: {cellLabel}\n\
	is NA: {pd.isna(cellLabel)}\n\
	res: {res}"
	#numOutLayer: {numOutLayer} nNeighborsUMAP: {nNeighborsUMAP} inters: {inters} cellsPerIter: {cellsPerIter}, nNeighborsKbet {nNeighborsKbet} \n\
#print(allVars)

outClustStatDir = os.sep.join(scVitalClusterOutFile.split("/")[:-2]+["figures"])+os.sep
#print(outClustStatDir)
sc.settings.figdir = outClustStatDir
if not os.path.exists(outClustStatDir):
	os.mkdir(outClustStatDir)

clustKey = "scVital"
latentRep = "X_pcAE"

adata = sc.read(inAdataFile)

numOutLayer = adata.obsm[latentRep].shape[1] #encoder.mu[0].out_features

#adata.obsm[latentRep] = bLatent

vb.testClustAndStats(adata, umapKey = "ae", neighborsKey=clustKey, pcaRep=latentRep, 
					cellLabel=cellLabel, batchLabel=batchLabel, res=res,
					numOutLayer=numOutLayer, outClustStatDir=outClustStatDir)
#					nNeighborsKbet=nNeighborsKbet, inters=inters, cellsPerIter=cellsPerIter, outUMAPFile=outUMAPFile, 


adata.write(outAdataFile)
scVitalLatent= adata.obsm[latentRep].copy()

pd.DataFrame(scVitalLatent).to_csv(scVitalLatentOutFile,header=False, index=False)
pd.DataFrame(adata.obs[clustKey]).to_csv(scVitalClusterOutFile)




