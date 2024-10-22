import sys
import os
import random
import time
import scanpy as sc
import pandas as pd
import numpy as np
import pdb
import pickle
from scib_metrics.benchmark import Benchmarker

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import FuncVizBench as vb
import FuncAUC as auc

#print(snakemake.input)
#print(snakemake.params)

inAdataFile = snakemake.input["inAdata"]
name = inAdataFile.split("~")[1].split("/")[0]

dataName = snakemake.params[0]["filename"] #inAdataFile.split("~")[1].split("/")[0]
batchName = snakemake.params[0]["batchName"] #inAdataFile.split("/")[4].split("_")[9].split("~")[1]
labelName = snakemake.params[0]["cellName"] #inAdataFile.split("/")[4].split("_")[10].split("~")[1]
dataDir = ("/").join(inAdataFile.split("/")[:-1])

#outBenchFile = snakemake.output["outBench"]
outAdataFile = snakemake.output["outAdata"]

latNameSet = set()
clutNameSet = set()

if(labelName != ""):
	adata = sc.read(inAdataFile)    #set adata  
	
	for infoDir in os.listdir(os.path.join(dataDir)):##in directory with latents and clusters
	    if(os.path.isdir(os.path.join(dataDir,infoDir))): 
	        for file in os.listdir(os.path.join(dataDir,"latents")):
	            if(".csv" in file):
	                name = file.split(".")[0]
	                adata.obsm[f"X_{name}"] = pd.read_table(os.path.join(dataDir,"latents",file), sep=",",header=None).to_numpy()
	                latNameSet.add(f"X_{name}")
	
	        for file in os.listdir(os.path.join(dataDir,"clusters")):
	            if(".csv" in file):
	                name = file.split(".")[0]
	                adata.obs[name] = pd.read_table(os.path.join(dataDir,"clusters",file), sep=",",index_col=0,dtype=str).iloc[:,0].to_numpy()
	                clutNameSet.add(f"{name}")
	
	latents = list(latNameSet)
	adata = auc.checkPairsAddSimple(adata, batchName, labelName)
	
	#pdb.set_trace()
	batches = adata.obs[batchName].cat.categories.values

	
	#adata.uns["cellTypes1"] = np.unique(adata[adata.obs.species==batches[0]].obs[labelName])
	#adata.uns["cellTypes2"] = np.unique(adata[adata.obs.species==batches[1]].obs[labelName])
	aucScores = auc.plotGetAuc(adata, dataName, latents, cellTypeKey=labelName, 
							  humanCellTypes=adata.uns["cellTypes1"], mouseCellTypes=adata.uns["cellTypes2"], pairs=adata.uns["pairs"], 
							  batchKey=batchName, plot=False, save=dataDir)
	
	aucDf = pd.DataFrame(aucScores,columns=[dataName],index=latents).T
	aucDf.to_csv(os.path.join(dataDir,"figures","aucScores.csv"))

	sep="~"
	allCt = []
	for batch in batches:
		cellTypes = np.unique(adata.obs[adata.obs[batchName]==batch][labelName])
		batchCt = [f"{batch}{sep}{ct1}" for ct1 in cellTypes]
		allCt = allCt + batchCt

	pairs = {(ctP[0], ctP[1]) for ctP in adata.uns["pairs"]}
	
	ctPairs = []
	for p1 in allCt:
		for p2 in allCt:
			name1, ct1 = p1.split(sep)
			name2, ct2 = p2.split(sep)
			if(((ct1==ct2) or ((ct1,ct2) in pairs or (ct2,ct1) in pairs)) and (name1 != name2)):
				ctPairs = ctPairs + [[p1, p2]]
	print(ctPairs)
	#ctPairs = [[f"{batches[0]}{sep}{pair[0]}",f"{batches[1]}{sep}{pair[1]}"] for pair in pairs]+[[f"{batches[1]}{sep}{pair[1]}",f"{batches[0]}{sep}{pair[0]}"] for pair in pairs]
	clustDists, totalDists, retRocAucs = auc.plotGetAucAllCt(adata, latents, labelName, batchName, 
	                                                     allCt, ctPairs=ctPairs, save=dataDir, plot=False)
	labelName = "overlapLabel"
	
	aucDf = pd.DataFrame(retRocAucs, columns=[dataName],index=latents).T
	aucDf.to_csv(os.path.join(dataDir,"figures","fullAucScores.csv"))

	for i,clustDist in enumerate(clustDists):
		batchToColorDict = dict(zip(batches,adata.uns[f"{batchName}_colors"]))
		annoToColorDict = dict(zip(np.unique(adata.obs[labelName]),adata.uns[f"{labelName}_colors"]))

		auc.makeGraphLSS(clustDist, batchToColorDict, annoToColorDict, np.sort(np.unique(adata.obs[labelName])), adata.uns["pairs"],
						 name=latents[i], save=os.path.join(dataDir,"figures"))
	

	bm = Benchmarker(adata,
					 batch_key = batchName, 
					 label_key = labelName,
					 embedding_obsm_keys = latents,
					 n_jobs=len(latents))
	bm.benchmark()
	benchDF = bm.get_results(min_max_scale=False)
	benchDF.to_csv(os.path.join(dataDir,"figures","bench.csv"))
	bm.plot_results_table(min_max_scale=False,show=False, save_dir=os.path.join(dataDir,"figures"))

	
	adata.write(outAdataFile)