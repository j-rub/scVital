import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns

from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
import networkx as nx

import pdb

from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import RocCurveDisplay, roc_curve
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster import hierarchy 
from scipy.stats import power_divergence
from scipy.sparse import isspmatrix
from scipy.spatial import distance

import textwrap

###		   ### 
# AUC Functions #
###		   ###

def simpleCasePairsAdata(adata, cellTypeLabel, batchLabel="species", batches = ["human","mouse"]):
	hlab = np.unique(adata[adata.obs[batchLabel]==batches[0]].obs[cellTypeLabel])
	mlab = np.unique(adata[adata.obs[batchLabel]==batches[1]].obs[cellTypeLabel])

	pairs = []
	for h in hlab:
		for m in mlab:
			if h == m:
				pairs.append([h,m])
	return([hlab, mlab, pairs])

	#copilot
	#def find_common_items(*grocery_lists):
		# Convert the lists to sets
		#sets = [set(grocery_list) for grocery_list in grocery_lists]
		# Find the intersection of all sets
		#common_items = set.intersection(*sets)
		# Convert the common items to pairs
		#common_items_pairs = [(item, item) for item in common_items]
		#return common_items_pairs

  
def simpleCasePairs(adataFileDir,cellTypeLabel,batchLabel="species", batches = ["human","mouse"]):
	adata =  sc.read(os.path.join(adataFileDir,"vaeOut_EvalBench.h5ad"))
	hlab, mlab, pairs = simpleCasePairsAdata(adata,cellTypeLabel,batchLabel="species", batches = batches)
	return([adataFileDir,cellTypeLabel, hlab, mlab, pairs])

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

def checkPairsAddSimple(adata, batchName, labelName):
	batches = adata.obs[batchName].cat.categories.values
	try:
		adata.uns["cellTypes1"] = np.unique(adata[adata.obs[batchName]==batches[0]].obs[labelName])
		adata.uns["cellTypes2"] = np.unique(adata[adata.obs[batchName]==batches[1]].obs[labelName])
		print(adata.uns["pairs"])
	except:
		#print("pairs not found")
		hlab, mlab, pairs = simpleCasePairsAdata(adata,labelName,batchLabel=batchName,batches=batches)
		adata.uns["cellTypes1"] = hlab
		adata.uns["cellTypes2"] = mlab
		adata.uns["pairs"] = pairs

	adata.obs["overlapLabel"] = findOverlapLabels(adata.uns["pairs"], adata.obs[labelName])
	return(adata)


def pairToIndex(humanCellTypes, mouseCellTypes,ctPairs):
	overlapCT = []
	flatOver = []
	for hct, mct in ctPairs:
		if(type(humanCellTypes)==np.ndarray):
			try:
				hctI = np.where(humanCellTypes == hct)[0][0]
				mctI = np.where(mouseCellTypes == mct)[0][0]
			except:
				#pdb.set_trace()
				#print("not np array")
				continue
		else:
			hctI = humanCellTypes.index(hct)
			mctI = mouseCellTypes.index(mct)
		#pdb.set_trace()

		try:
			overlapCT.append([hctI,mctI])
			flatOver.append((hctI*len(mouseCellTypes)+mctI))
		except:
			continue

	return(overlapCT, flatOver)

def getClustDist(adata, cellLabel, latentLabel, humanCellTypes, mouseCellTypes,ctPairs, batchKey = "species", batches=["human","mouse"]):
	batches = adata.obs[batchKey].cat.categories.values
	clustDist = pd.DataFrame(np.zeros((len(humanCellTypes),len(mouseCellTypes))), columns=mouseCellTypes, index=humanCellTypes)
	for hc in clustDist.index:
		hMean = np.mean(adata[np.logical_and(adata.obs[cellLabel]==hc,adata.obs[batchKey]==batches[0])].obsm[latentLabel], axis=0)
		for mc in clustDist.columns:
			mMean = np.mean(adata[np.logical_and(adata.obs[cellLabel]==mc,adata.obs[batchKey]==batches[1])].obsm[latentLabel], axis=0)
			clustDist.loc[hc,mc] = np.round(distance.cosine(hMean,mMean),decimals=4)
	return clustDist

def compLatClust(adata, cellLabel, latentLabel, humanCellTypes, mouseCellTypes,ctPairs, batchKey = "species", cmap="Reds", clustMet ="euclidean", rowCol=None, colCol=None, plot=False):
	clustDist = getClustDist(adata, cellLabel, latentLabel, humanCellTypes, mouseCellTypes,ctPairs, batchKey = batchKey)
	if plot:
		g = sns.clustermap(clustDist, cmap=cmap, metric=clustMet, row_colors=rowCol, col_colors=colCol)

	overlapCT, _ = pairToIndex(humanCellTypes, mouseCellTypes,ctPairs)
	totalDist = 0
	for (i, j) in overlapCT:
		if plot:
			clustered_row_i, clustered_col_j = g.dendrogram_row.reordered_ind.index(i), g.dendrogram_col.reordered_ind.index(j)
			# Draw a rectangle on the clustered heatmap
			rect = patches.Rectangle((clustered_col_j , clustered_row_i), 1, 1, linewidth=4, edgecolor='black', facecolor='none')
			g.ax_heatmap.add_patch(rect)

		totalDist +=  clustDist.iloc[i,j]
	if plot:
		plt.show()
	exclu = sum(sum(clustDist.values))-totalDist
	return totalDist,exclu,clustDist
	
def getPred(clustDist, flatOver, ignFirst=False):
	flatDist = clustDist.values.flatten()
	distSorted = np.argsort(flatDist).astype(int)
	if(ignFirst):
		distSorted = distSorted[len(clustDist)::2]
	return(np.array([1 if x in flatOver else 0 for x in distSorted]))

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

def wraptext(ax, width = 10, break_long_words=False):
	labs = []
	for lab in ax.get_xticklabels():
		text = lab.get_text()
		labs.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
	ax.set_xticklabels(labs, rotation=90)
	labs = []
	for lab in ax.get_yticklabels():
		text = lab.get_text()
		labs.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
	ax.set_yticklabels(labs, rotation=0)

def plotAllClustDist(adata, name, latents, clustDists=None, overlapCT=None, cellTypeKey=None,  humanCellTypes=None, mouseCellTypes=None, pairs=None, save=False):
	# Create a new figure and axes
	fig, axes = plt.subplots(figsize=(30, 20))
	if len(latents) > 1:
		fig, axes = plt.subplots((len(latents)>2)+1, (len(latents)//2)+1, figsize=(30, 20))
	
	# Reposition each clustermap into the subplots
	for i,latentLabel in enumerate(latents):
		if len(latents) > 1:
			axs = axes[i%2,i//2]
		else:
			axs = axes
		
		if(clustDists == None):
			clustDist = getClustDist(adata, cellTypeKey, latentLabel, humanCellTypes, mouseCellTypes, pairs, batchKey = "species")
			overlapCT, _ = pairToIndex(humanCellTypes, mouseCellTypes, pairs)
		else:
			clustDist = clustDists[i]
		#pdb.set_trace()
		#print(clustDist)
		clmap = sns.clustermap(clustDist)#, vmin=0, vmax=2)
		htmap = sns.heatmap(clmap.data2d, ax=axs, cmap="Reds", cbar=True)#, vmin=0, vmax=2)
		axs.set_title(latentLabel)
		
		for (i, j) in overlapCT:
			clustered_row_i, clustered_col_j = clmap.dendrogram_row.reordered_ind.index(i), clmap.dendrogram_col.reordered_ind.index(j)
			rect = patches.Rectangle((clustered_col_j , clustered_row_i), 1, 1, linewidth=4, edgecolor='black', facecolor='none')
			axs.add_patch(rect)

		wraptext(axs)
		plt.close(clmap.fig)

	fig.suptitle(name)
	# Adjust layout
	plt.tight_layout()

	if(save):
		fig.savefig(f'{save}/{name}_latentDist.png')
	else:
		plt.show()
	plt.close(fig)
		
def plotClmapAllCt(adata, latents, cellTypeKey, allCellTypes, ctPairs, batchKey, save=False, plot=False):
	clustDists = []
	totalDists = []
	#pdb.set_trace()
	for x_lat in latents:
		clustDist = pd.DataFrame(np.zeros((len(allCellTypes),len(allCellTypes))), columns=allCellTypes, index=allCellTypes)
		for hcs in clustDist.index:
			hs, hc = hcs.split("~")
			hMean = np.mean(adata[np.logical_and(adata.obs[cellTypeKey]==hc,adata.obs[batchKey]==hs)].obsm[x_lat], axis=0)
			for mcs in clustDist.columns:
				ms, mc = mcs.split("~")
				mMean = np.mean(adata[np.logical_and(adata.obs[cellTypeKey]==mc,adata.obs[batchKey]==ms)].obsm[x_lat], axis=0)
				clustDist.loc[hcs,mcs] = np.round(distance.cosine(hMean,mMean),decimals=4)
		#print(clustDist)
		#pdb.set_trace()
		
		condClustDist = squareform(clustDist)
		Z = linkage(condClustDist, 'complete')
		dn = hierarchy.dendrogram(Z, no_plot=True)
		
		mask = np.zeros_like(clustDist)
		mask[np.tril_indices_from(mask)] = True
		mask = mask[:,np.argsort(dn["leaves"])]
		mask = mask[np.argsort(dn["leaves"]),:]
		
		cdMap = sns.clustermap(clustDist, mask=mask, row_linkage=Z, col_linkage=Z, cmap="RdBu")#,vmin=0,vmax=2)
		cdMap.ax_col_dendrogram.set_visible(False)
		cdMap.ax_heatmap.set_title(x_lat)
		realPairs, flatRealPair = pairToIndex(np.array(allCellTypes), np.array(allCellTypes), np.array(ctPairs))
		totalDist = 0
		for (i, j) in np.array(realPairs):
			clustered_row_i, clustered_col_j = cdMap.dendrogram_row.reordered_ind.index(i), cdMap.dendrogram_col.reordered_ind.index(j)
			# Draw a rectangle on the clustered heatmap
			if(clustered_row_i < clustered_col_j):
				rect = patches.Rectangle((clustered_col_j , clustered_row_i), 1, 1, linewidth=4, edgecolor='black', facecolor='none')
				cdMap.ax_heatmap.add_patch(rect)
				totalDist +=  clustDist.iloc[i,j]
			
		clustDists.append(clustDist)
		totalDists.append(totalDist)
		if(not plot):
			plt.close(cdMap.fig)
		if(save):
			cdMap.savefig(f"{save}/{x_lat[2:]}_Clmap.svg")
	return(clustDists, flatRealPair, totalDists)
	
def plotClutLatent(adata, name, latents, cellLabel, humanCellTypes, mouseCellTypes,ctPairs, batchKey = "species", batches=["human","mouse"], save=False):
	clustDists = []
	for lat in latents:
		clustDist = getClustDist(adata, cellLabel, lat, humanCellTypes, mouseCellTypes,ctPairs, batchKey = batchKey, batches=batches)
		clustDists.append(clustDist)
		overlapCT, flatOver = pairToIndex(humanCellTypes, mouseCellTypes, ctPairs)
	#pdb.set_trace()
	plotAllClustDist(adata, name, latents, clustDists=clustDists, overlapCT=overlapCT, cellTypeKey=None,  humanCellTypes=None, mouseCellTypes=None, pairs=None, save=save)
	return(clustDists, flatOver)

def plotGetAuc(adata, name, latents,cellTypeKey, humanCellTypes, mouseCellTypes, pairs, batchKey, batches=["human","mouse"], plot=False, save=False):
	clustDists, flatOver = plotClutLatent(adata, name, latents, cellTypeKey, humanCellTypes, mouseCellTypes, pairs, batchKey = batchKey, batches=batches,save=save)
	return(np.round(getAUCs(latents, clustDists, flatOver, plot=plot), decimals=3))

def plotGetAucAllCt(adata, latents, cellTypeKey, batchKey, allCellTypes, ctPairs, save=False, plot=False):
	clustDists, flatOver, totalDists = plotClmapAllCt(adata, latents, cellTypeKey, allCellTypes, ctPairs, batchKey, save=save, plot=plot)
	retRocAucs = getAUCs(latents, clustDists, flatOver, ignFirst=True, save=save, plot=plot)
	return(clustDists, totalDists, retRocAucs)

def getAllAucs(dataInfoDict, latents, batchKey = "species", adataFileSuffix="vaeOut_EvalBench.h5ad", save=False):
	aucs = pd.DataFrame(np.zeros((len(dataInfoDict.keys()),len(latents))), columns=latents, index=dataInfoDict.keys())
	for i,tiss in enumerate(dataInfoDict.keys()):
		adata = sc.read(os.path.join(dataInfoDict[tiss]["dataDir"],adataFileSuffix))
		aucs.iloc[i] = plotGetAuc(adata,tiss,latents,dataInfoDict[tiss]["cellTypeKey"], 
								  dataInfoDict[tiss]["humanCellTypes"], dataInfoDict[tiss]["mouseCellTypes"],dataInfoDict[tiss]["pairs"], batchKey,plot=False)
	return(aucs)

def findPair(pairs, ctf, colorDict):
	check1 = False
	check2 = False
	for ct1,ct2 in pairs:
		if (ctf==ct1):
			if(ct2 in colorDict.keys()):
				return ct2
			else:
				check2 = ct2
		elif (ctf==ct2):
			if(ct1 in colorDict.keys()):
				return ct1
			else:
				check1 = ct1
	if(check1):
		return findPair(pairs, check1, colorDict)
	if(check2):
		return findPair(pairs, check2, colorDict)

def getOverColors(ogLabel,overlabel,pairs,colorDict):
	colorOut = ogLabel.copy()
	for i,ctf in enumerate(ogLabel):
		if (ctf not in overlabel):
			ctf = findPair(pairs, ctf, colorDict)
		colorOut[i] = colorDict[ctf]
	return colorOut


def makeGraphLSS(clustDist, batchToColorDict, annoToColorDict, overlap, pairs, name="", wLab=False, qCut = 0.28, save=False):
	fig, ax = plt.subplots()
	G = nx.Graph()
	for i in clustDist.columns:
		G.add_node(i)
	batchColors = [batchToColorDict[label.split("~")[0]] for label in clustDist.columns]
	cellTColors = getOverColors([label.split("~")[1] for label in clustDist.columns], overlap, pairs, annoToColorDict)
								 #adata.obs.overlapLabel.cat.categories.values, adata.uns["pairs"]
	cutoff = np.quantile(clustDist.to_numpy().flatten(), qCut)
	for i in range(len(clustDist.columns)):
		for j in range(i,len(clustDist.columns)):
			if((i != j) and clustDist.iloc[i,j] < cutoff):
				G.add_edge(clustDist.columns[i], clustDist.index[j], weight=clustDist.iloc[i,j])

	mouseNodes = clustDist.columns[["human" not in l for l in clustDist.columns]]
	mCtColors = getOverColors([label.split("~")[1] for label in mouseNodes], overlap, pairs, annoToColorDict)
	
	humanNodes = clustDist.columns[["human" in l for l in clustDist.columns]]
	hCtColors = getOverColors([label.split("~")[1] for label in humanNodes], overlap, pairs, annoToColorDict)
	lw=3
	pos = nx.spring_layout(G, seed=42)
	
	nx.draw_networkx_nodes(G,pos,nodelist=mouseNodes, edgecolors = "black", node_color=mCtColors, node_shape='s',linewidths=lw, ax=ax)
	nx.draw_networkx_nodes(G,pos,nodelist=humanNodes, edgecolors = "gray",  node_color=hCtColors, node_shape='o',linewidths=lw, ax=ax)
	nx.draw_networkx_edges(G,pos, ax=ax)
	
	ax.axis("off")

	plt.tight_layout()
	#plt.show()
	if(save):
		fig.savefig(f"{save}/LSSgraph_{name}.svg", format="svg")
		nx.write_adjlist(G, f"{save}/LSSgraph_{name}.adjlist")



def makeGraphLSSMulti(clustDist, batchToColorDict, annoToColorDict, overlap, pairs, name="", wLab=False, qCut = 0.28, save=False):
	fig, ax = plt.subplots()
	G = nx.Graph()
	for i in clustDist.columns:
		G.add_node(i)
	cutoff = np.quantile(clustDist.to_numpy().flatten(), qCut)
	for i in range(len(clustDist.columns)):
		for j in range(i,len(clustDist.columns)):
			if((i != j) and clustDist.iloc[i,j] < cutoff):
				G.add_edge(clustDist.columns[i], clustDist.index[j], weight=clustDist.iloc[i,j])
	pos = nx.spring_layout(G, seed=42)
	nx.draw_networkx_edges(G,pos, ax=ax)
	lw=3
	for bat in list(set([cl.split("~")[0] for cl in clustDist.columns])):
		nodes = clustDist.columns[[bat==cl.split("~")[0] for cl in clustDist.columns]]
		ctColors = getOverColors([label.split("~")[1] for label in nodes], overlap, pairs, annoToColorDict)
		nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=ctColors,edgecolors = batchToColorDict[bat]["color"], node_shape=batchToColorDict[bat]["shape"],linewidths=lw, ax=ax)

	#mouseNodes = clustDist.columns[["human" not in l for l in clustDist.columns]]
	#mCtColors = getOverColors([label.split("~")[1] for label in mouseNodes], overlap, pairs, annoToColorDict)
	#nx.draw_networkx_nodes(G,pos,nodelist=mouseNodes, edgecolors = "black", node_color=mCtColors, node_shape='s',linewidths=lw, ax=ax)

	#humanNodes = clustDist.columns[["human" in l for l in clustDist.columns]]
	#hCtColors = getOverColors([label.split("~")[1] for label in humanNodes], overlap, pairs, annoToColorDict)
	#nx.draw_networkx_nodes(G,pos,nodelist=humanNodes, edgecolors = "gray",  node_color=hCtColors, node_shape='o',linewidths=lw, ax=ax)
	
	ax.axis("off")

	plt.tight_layout()
	#plt.show()
	if(save):
		fig.savefig(f"{save}/LSSgraph_{name}.svg", format="svg")
		nx.write_adjlist(G, f"{save}/LSSgraph_{name}.adjlist")









