import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns

from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
import networkx as nx

from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import RocCurveDisplay, roc_curve
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster import hierarchy 
from scipy.stats import power_divergence
from scipy.sparse import isspmatrix
from scipy.spatial import distance



def calcLSS(adata, latent, batchLabel, trueLabel, newLabel):
	pass 

def graphVizLSS():
	pass 

def heatVizLSS():
	pass 


def calcClustDist(adata, latent, allCellTypes, batchLabel):
	clustDist = pd.DataFrame(np.zeros((len(allCellTypes),len(allCellTypes))), columns=allCellTypes, index=allCellTypes)
	for hcs in clustDist.index:
		hs, hc = hcs.split("~")
		hMean = np.mean(adata[np.logical_and(adata.obs[cellTypeKey]==hc,adata.obs[batchLabel]==hs)].obsm[latent], axis=0)
		for mcs in clustDist.columns:
			ms, mc = mcs.split("~")
			if(hc == mc):
				clustDist.loc[hcs,mcs] = 0
			else:
				mMean = np.mean(adata[np.logical_and(adata.obs[cellTypeKey]==mc,adata.obs[batchLabel]==ms)].obsm[latent], axis=0)
				clustDist.loc[hcs,mcs] = np.round(distance.cosine(hMean,mMean),decimals=4)
				#clustDist.loc[hcs,mcs] = np.round(mean_squared_error(hMean,mMean),decimals=4)
				#r, _ = scipy.stats.pearsonr(hMean,mMean)
				#clustDist.loc[hcs,mcs] = np.abs(np.round(r,decimals=4)-1)
	return(clustDist)


def plotHeatLSS(adata, latent, cellTypeKey, allCellTypes, ctPairs, batchLabel, save=False, plot=False):
	condClustDist = squareform(clustDist)
	Z = linkage(condClustDist, 'complete')
	dn = hierarchy.dendrogram(Z, no_plot=True)
	
	mask = np.zeros_like(clustDist)
	mask[np.tril_indices_from(mask)] = True
	mask = mask[:,np.argsort(dn["leaves"])]
	mask = mask[np.argsort(dn["leaves"]),:]
	
	cdMap = sns.clustermap(clustDist, mask=mask, row_linkage=Z, col_linkage=Z, cmap="RdBu",vmin=0,vmax=2)
	cdMap.ax_col_dendrogram.set_visible(False)
	cdMap.ax_heatmap.set_title(latent)
	realPairs, flatRealPair = pairToIndex(np.array(allCellTypes), np.array(allCellTypes), np.array(ctPairs))
	totalDist = 0
	for (i, j) in np.array(realPairs):
		clustered_row_i, clustered_col_j = cdMap.dendrogram_row.reordered_ind.index(i), cdMap.dendrogram_col.reordered_ind.index(j)
		# Draw a rectangle on the clustered heatmap
		if(clustered_row_i < clustered_col_j):
			rect = patches.Rectangle((clustered_col_j , clustered_row_i), 1, 1, linewidth=4, edgecolor='black', facecolor='none')
			cdMap.ax_heatmap.add_patch(rect)
			totalDist += clustDist.iloc[i,j]

	if(not plot):
		plt.close(cdMap.fig)
	if(save): #save
		cdMap.savefig(f"{save}/Clmap_{latent}.svg")

def plotGraphLSS(clustDist, batchDict, annoToColorDict, overlap, pairs, name="", prog="neato", wLab=False, qCut = 0.28, plot=False, save=False):
	batchToColorDict = {lab:batchDict[lab][1] for lab in batchDict.keys()}
	batchToShapeDict = {lab:batchDict[lab][2] for lab in batchDict.keys()}
	
	fig, ax = plt.subplots()
	G = nx.Graph()
	for i in clustDist.columns:
		G.add_node(i)

	allDists = clustDist.to_numpy().flatten()
	cutoff = np.quantile(allDists[allDists>0], qCut)
	
	for i in range(len(clustDist.columns)):
		for j in range(i,len(clustDist.columns)):
			if((i != j) and clustDist.iloc[i,j] < cutoff):
				G.add_edge(clustDist.columns[i], clustDist.index[j], weight=clustDist.iloc[i,j])
	pos = graphviz_layout(G, prog=prog)#, seed=42)
	nx.draw_networkx_edges(G, pos, ax=ax)
	lw=1
	for j,bat in enumerate(list(set([cl.split("~")[0] for cl in clustDist.columns]))):
		nodes = clustDist.columns[[bat==cl.split("~")[0] for cl in clustDist.columns]]		
		ctColors = getOverColors([label.split("~")[1] for label in nodes], overlap, pairs, annoToColorDict)
		nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=ctColors, node_size=150, edgecolors = batchToColorDict[bat],  node_shape=batchToShapeDict[bat], linewidths=lw, ax=ax)#alpha=0.9
	
	fig.suptitle(f"LSS Cut:{qCut}, Prog:{prog}")
	ax.axis("off")
	plt.tight_layout()
	if(not plot):
		plt.close(fig)
	if(save):
		#clustDist.to_csv(f"{save}/multiLSS_{name}scVital.csv")
		#nx.write_adjlist(G, f"{save}/multiLSS_{name}scVital.adjlist")
		#nx.write_weighted_edgelist(G, f"{save}/multiLSS_{name}scVital.weighted.edgelist")
		fig.savefig(f"{save}/graphLSS_{name}.svg", format="svg")

def pairToIndex(cellTypes1, cellTypes2, ctPairs):
	overlapCT = []
	flatOver = []
	for ct1, ct2 in ctPairs:
		if(type(cellTypes1)==np.ndarray):
			try:
				ct1I = np.where(cellTypes1 == ct1)[0][0]
				ct2I = np.where(cellTypes2 == ct2)[0][0]
			except:
				#print("not np array")
				continue
		else:
			ct1I = cellTypes1.index(ct1)
			ct2I = cellTypes2.index(ct2)
		try:
			overlapCT.append([ct1I,ct2I])
			flatOver.append((ct1I*len(cellTypes2)+ct2I))
		except:
			continue

	return(overlapCT, flatOver)


def calcLSS(adata, latents, cellTypeKey, batchLabel, allCellTypes, ctPairs, save=False, plot=False):
	clustDist = calcClustDist(adata, latent, allCellTypes, batchLabel)


	realPairs, flatRealPair = pairToIndex(np.array(allCellTypes), np.array(allCellTypes), np.array(ctPairs))
	totalDist = 0
	for (i, j) in np.array(realPairs):
		clustered_row_i, clustered_col_j = cdMap.dendrogram_row.reordered_ind.index(i), cdMap.dendrogram_col.reordered_ind.index(j)
		# Draw a rectangle on the clustered heatmap
		if(clustered_row_i < clustered_col_j):
			rect = patches.Rectangle((clustered_col_j , clustered_row_i), 1, 1, linewidth=4, edgecolor='black', facecolor='none')
			cdMap.ax_heatmap.add_patch(rect)
			totalDist += clustDist.iloc[i,j]
			
	clustDists, flatOver, totalDists = plotClmapAllCt(adata, latents, cellTypeKey, allCellTypes, ctPairs, batchLabel, save=save, plot=plot)
	retRocAucs = getAUCs(latents, clustDists, flatOver, ignFirst=True, save=save, plot=plot)
	return(clustDists, totalDists, retRocAucs)

def getAUCs(predName, clustDists, flatOver, ignFirst=False, save=False, plot=False):
	retRocAucs = []
	fig, ax = plt.subplots(figsize=(6, 6))

	for name, clutDista in zip(predName,clustDists):
		labelPred = getPred(clutDista, flatOver,ignFirst)
		labelTrue = [1]*sum(labelPred)+[0]*(len(labelPred)-sum(labelPred))
		roc = RocCurveDisplay.from_predictions(labelTrue, labelPred, name=f"ROC for {name[2:]}", ax=ax)
		retRocAucs.append(roc.roc_auc)

	_ = ax.set(
		xlabel="False Positive Rate",
		ylabel="True Positive Rate",
		title="ROC Smallest Cluster Distance",
	)
	if(save):
		fig.savefig(f"{save}/rocAUC.svg")
	if(not plot):
		plt.close(fig)
	return retRocAucs

def checkPairsAddSimple(adata, batchName, labelName):
	batches = adata.obs[batchName].cat.categories.values
	try:
		adata.uns["pairs"]
	except:
		print("finding pairs")
		adata.uns["pairs"] = simpleCasePairsAdata(adata, labelName, batchLabel=batchName)

	adata.obs["overlapLabel"] = findOverlapLabels(adata.uns["pairs"], adata.obs[labelName])
	return(adata)

def getUniCtPairs(adata, batchName, labelName):
	batches = adata.obs[batchName].cat.categories.values

	sep="~"
	allCt = []
	for batch in batches:
		cellTypes = np.unique(adata.obs[adata.obs[batchName]==batch][labelName])
		batchCt = [f"{batch}{sep}{ct}" for ct in cellTypes]
		allCt = allCt + batchCt

	pairs = {(ctP[0], ctP[1]) for ctP in adata.uns["pairs"]}

	ctPairs = []
	for p1 in allCt:
		for p2 in allCt:
			name1, ct1 = p1.split(sep)
			name2, ct2 = p2.split(sep)
			if(((ct1==ct2) or ((ct1,ct2) in pairs or (ct2,ct1) in pairs)) and (name1 != name2)):
				ctPairs = ctPairs + [[p1, p2]]

	return(allCt, ctPairs)


def simpleCasePairsAdata(adata, cellTypeLabel, batchLabel="species"):
	batches = adata.obs[batchName].cat.categories.values
	#for batch in batches:
	#	adata.uns[f"cellTypes{batch}"] = np.unique(adata[adata.obs[batchName]==batch].obs[labelName])

	cellTypesBatch =[set(adata[adata.obs[batchName]==batch].obs[labelName]) for batch in batches]

	pairs = []

	for ctB1 in cellTypesBatch:
		for ctB2 in cellTypesBatch:
			for ct in ctB1.intersection(ctB2):
				pairs.append([ct,ct])
			#for ctb1 in ctB1:
			#	for ctb2 in ctB2:
			#		if ctb1 == ctb2:
			#			pairs.append([ctb1,ctb2])
	return(pairs)


#commented with copilot
def findOverlapLabels(pairs, ogLabels):
	# Initialize an empty list to store overlapping label sets
	simple = []
	# Iterate through each pair of labels
	for pair in pairs:
		setPair = set(pair)
		addPair = True
		# Check if the current pair intersects with any existing label set
		for i, setLab in enumerate(simple):
			if setLab.intersection(setPair):
				# If there's an intersection, merge the sets
				simple[i] = setLab.union(setPair)
				addPair = False
		# If no intersection, add the pair as a new label set
		if addPair:
			simple.append(setPair)
	# Create a dictionary to map labels to simplified labels
	simple = np.unique(simple)
	label2simp = dict()
	for i, setLabels in enumerate(simple):
		label2simp.update(dict(zip(list(setLabels), [f"{i}"] * len(setLabels))))
	# Assign unique simplified labels to any remaining original labels
	totalLabels = len(simple)
	for anno in np.unique(ogLabels):
		if(anno not in label2simp.keys()):
			label2simp.update({anno: f"{totalLabels}"})
			totalLabels += 1
	# Return a list of simplified labels corresponding to the original labels
	simpLabs = np.full(totalLabels, fill_value="" ,dtype=object)
	cellLabels = list(label2simp.keys())
	overlapLabelNum = list(label2simp.values())
	for i in range(totalLabels):
		simpLabs[i] = cellLabels[overlapLabelNum.index(str(i))]
	retSimpleLabels = [simpLabs[int(label2simp[lab])] for lab in ogLabels]
	return(retSimpleLabels)










