import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import anndata as an

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D      
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import matplotlib.image as img
import matplotlib.patches as patches
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster import hierarchy 
from scipy.spatial import distance

from collections import defaultdict

def calcPairsLSS(adata, latent, batchName, cellTypeLabel):
    # Check and add simple pairs to the data
    adata = checkPairsAddSimple(adata, batchName, cellTypeLabel)
    # Get unique cell types and their pairs
    allCellTypes, ctPairs = getUniCtPairs(adata, batchName, cellTypeLabel)
    # Calculate LSS metrics
    clustDist, lssAUC, totalDist = calcLSS(adata, latent, cellTypeLabel, batchName, allCellTypes, ctPairs)
    # Return calculated metrics and cell type information
    return clustDist, lssAUC, totalDist, allCellTypes, ctPairs

def calcLSS(adata, latent, cellTypeLabel, batchName, allCellTypes, ctPairs):
    # Calculate cluster distances
    clustDist = calcClustDist(adata, latent, allCellTypes, batchName, cellTypeLabel)
    # Convert cell type pairs to indices
    realPairs = __pairToIndex(np.array(allCellTypes), np.array(allCellTypes), np.array(ctPairs))
    if len(realPairs) == 0:
        print("No pairs or simple pairs") 
        return clustDist, 0, 0
    # Calculate total distance
    totalDist = calcTotalDist(clustDist, realPairs)
    # Calculate AUC for LSS
    lssAUC = calcAUC(clustDist, realPairs)
    # Return calculated distances and AUC
    return clustDist, lssAUC, totalDist


def calcAUC(clustDist, realPairs): 
    # Simplify distance matrix for AUC calculation
    simpDist = np.triu(-1 * clustDist.values) + np.tril(np.full(clustDist.shape[0], fill_value=-2))
    # Initialize true labels matrix
    labelTrue = np.zeros(clustDist.shape)
    # Mark true pairs in the label matrix
    for (i, j) in np.array(realPairs):
        labelTrue[i, j] = 1
    # Flatten the matrices for AUC calculation
    labelPred = simpDist.flatten()
    labelTrueF = np.triu(labelTrue).flatten()
    # Calculate and return AUC of F1 score
    precision, recall, thresholds = precision_recall_curve(labelTrueF, labelPred)
    return(auc(recall, precision))    #auc=roc_auc_score(labelTrue, labelPred)

def calcClustDist(adata, latent, allCellTypes, batchName, cellTypeLabel):
    # Initialize cluster distance matrix
    clustDist = pd.DataFrame(np.zeros((len(allCellTypes), len(allCellTypes))), columns=allCellTypes, index=allCellTypes)
    # Calculate mean latent space values for each cell type and batch
    for hcs in clustDist.index:
        hs, hc = hcs.split("~")
        hMean = np.mean(adata[np.logical_and(adata.obs[cellTypeLabel] == hc, adata.obs[batchName] == hs)].obsm[latent], axis=0)
        for mcs in clustDist.columns:
            ms, mc = mcs.split("~")
            if hcs == mcs:
                clustDist.loc[hcs, mcs] = 0
            else:
                mMean = np.mean(adata[np.logical_and(adata.obs[cellTypeLabel] == mc, adata.obs[batchName] == ms)].obsm[latent], axis=0)
                # Calculate cosine distance between means
                clustDist.loc[hcs, mcs] = np.round(distance.cosine(hMean, mMean), decimals=4)
                # Alternative distance metrics (commented out)
                # clustDist.loc[hcs, mcs] = np.round(mean_squared_error(hMean, mMean), decimals=4)
                # r, _ = scipy.stats.pearsonr(hMean, mMean)
                # clustDist.loc[hcs, mcs] = np.abs(np.round(r, decimals=4) - 1)
    # Return the cluster distance matrix
    return clustDist


def calcTotalDist(clustDist, realPairs): 
    # Initialize total distance
    totalDist = 0
    # Sum distances for all real pairs
    for (i, j) in np.array(realPairs):
        totalDist += clustDist.iloc[i, j]
    # Adjust total distance
    totalDist *= 0.5
    # Return total distance
    return totalDist


def checkPairsAddSimple(adata, batchName, labelName):
    batches = adata.obs[batchName].cat.categories.values
    try:
        adata.uns["pairs"]
    except:
        print("finding pairs")
        adata.uns["pairs"] = simpleCasePairsAdata(adata, batchName, labelName)

    adata.obs["overlapLabel"] = pd.Categorical(__findOverlapLabels(adata.uns["pairs"], adata.obs[labelName]))
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


def simpleCasePairsAdata(adata, batchName, cellTypeLabel):
    batches = adata.obs[batchName].cat.categories.values
    #for batch in batches:
    #    adata.uns[f"cellTypes{batch}"] = np.unique(adata[adata.obs[batchName]==batch].obs[labelName])

    cellTypesBatch =[set(adata[adata.obs[batchName]==batch].obs[cellTypeLabel]) for batch in batches]

    pairs = []

    for ctB1 in cellTypesBatch:
        for ctB2 in cellTypesBatch:
            for ct in ctB1.intersection(ctB2):
                pairs.append([ct,ct])
            #for ctb1 in ctB1:
            #    for ctb2 in ctB2:
            #        if ctb1 == ctb2:
            #            pairs.append([ctb1,ctb2])
    return(pairs)


#commented with copilot
def __findOverlapLabels(pairs, ogLabels):
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

def __pairToIndex(cellTypes1, cellTypes2, ctPairs):
    overlapCT = []
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
        except:
            continue

    return(overlapCT)

def plotHeatLSS(adata, clustDist, latent, allCellTypes, ctPairs, plot=True, save=False):
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
    realPairs = __pairToIndex(np.array(allCellTypes), np.array(allCellTypes), np.array(ctPairs))
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

def plotGraphLSS(adata, cellTypeLabel, batchName, clustDist, name="", ctColors=plt.get_cmap('tab10').colors, btColors=None, shapes="ospx^><.....", prog="neato", wLab=False, qCut = 0.28, plot=True, save=False):
    batchDict, annoToColorDict = __getBatchCellDicts(adata, cellTypeLabel, batchName, ctColors, btColors, shapes)

    overlap = np.unique(adata.obs[cellTypeLabel])
    pairs = adata.uns["pairs"]
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
        ctColors = __getOverColors([label.split("~")[1] for label in nodes], overlap, pairs, annoToColorDict)
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
    return batchDict, annoToColorDict

def __group_pairs(pairs):
    # Dictionary to hold the groups
    groups = defaultdict(set)

    # Iterate through each pair
    for name1, name2 in pairs:
        # Find the groups that name1 and name2 belong to
        group1 = next((group for group in groups.values() if name1 in group), None)
        group2 = next((group for group in groups.values() if name2 in group), None)

        if group1 and group2:
            if group1 != group2:
                # Merge the two groups if they are different
                group1.update(group2)
                for name in group2:
                    groups[name] = group1
        elif group1:
            group1.add(name2)
            groups[name2] = group1
        elif group2:
            group2.add(name1)
            groups[name1] = group2
        else:
            # Create a new group if neither name is in any group
            new_group = {name1, name2}
            groups[name1] = new_group
            groups[name2] = new_group

    # Extract unique groups
    unique_groups = set(frozenset(group) for group in groups.values())

    # Convert each group to a list
    return [list(group) for group in unique_groups]

def __findPair(pairs, ctf, visited=None): #modified with copilot
    if visited is None:
        visited = set()
    
    if ctf in visited:
        return None  # or handle the circular reference case appropriately
    
    visited.add(ctf)
    
    check1 = False
    check2 = False
    for ct1,ct2 in pairs:
        if (ctf==ct1):
            if(ct2 in annoToColorDict.keys()):
                return ct2
            else:
                check2 = ct2
        elif (ctf==ct2):
            if(ct1 in annoToColorDict.keys()):
                return ct1
            else:
                check1 = ct1
    if(check1):
        return __findPair(pairs, check1)
    if(check2):
        return __findPair(pairs, check2)

def __getOverColors(ogLabel, overlabel, pairs, colorDict):
    colorOut = ogLabel.copy()
    for i,ctf in enumerate(ogLabel):
        if (ctf not in overlabel):
            ctf = __findPair(pairs, ctf)
        colorOut[i] = colorDict[ctf]
    return colorOut

def __getBatchCellDicts(adata, cellTypeLabel, batchName, ctColors=plt.get_cmap('tab10').colors, btColors=None, shapes="ospx^><....."):
    numBatch=len(adata.obs[batchName].cat.categories)
    if len(shapes) < numBatch:
        shapes = shapes + "."*numBatch

    if len(ctColors) < max(len(adata.obs[cellTypeLabel].cat.categories), len(adata.obs["overlapLabel"].cat.categories)):
        ctColors = list(ctColors) + [ctColors[0]]*(max(len(adata.obs[cellTypeLabel].cat.categories), len(adata.obs["overlapLabel"].cat.categories)) - len(ctColors))

    pairs = adata.uns["pairs"]
    cellTypeDict = dict(zip(adata.obs[cellTypeLabel].cat.categories, list(zip(adata.obs[cellTypeLabel].cat.categories, 
                                                                              ctColors[:len(adata.obs[cellTypeLabel].cat.categories)]))))
    overlapDict = dict(zip(adata.obs["overlapLabel"].cat.categories, list(zip(adata.obs["overlapLabel"].cat.categories, 
                                                                              ctColors[:len(adata.obs["overlapLabel"].cat.categories)]))))
    for overKey in overlapDict.keys():
        cellTypeDict[overKey] = overlapDict[overKey]
    
    gpairs = __group_pairs(pairs)
    for gpair in gpairs:
        overColor = (0,0,0)
        for csc in gpair:
            if csc in overlapDict.keys():
                overColor = overlapDict[csc][1]
                break
        for csc in gpair:
            cellTypeDict[csc] = (csc, overColor)
    
    annoToColorDict = {lab:cellTypeDict[lab][1] for lab in cellTypeDict.keys()}
    
    if(btColors is None):
        btColors = [(i * 0.9/(numBatch-1),i * 0.9/(numBatch-1),i * 0.9/(numBatch-1)) for i in range(numBatch)]
    batchDict = dict(zip(adata.obs[batchName].cat.categories, 
                list(zip(adata.obs[batchName].cat.categories, btColors[:numBatch],shapes[:numBatch]))))
    
    makeLegend(adata.obs[cellTypeLabel], adata.obs[batchName], cellTypeDict, batchDict, name="", save=False)

    return(batchDict, annoToColorDict)


def makeLegend(ctVals, btVals, cellTypeColorDict, batchColorDict, name="", save=False):
    ctColLab = cellTypeColorDict.values()
    btColLab = batchColorDict.values()

    ctLegendEle = [Line2D([0], [0], color=ctc, marker="o", lw=0, label=ctLabel) for ctLabel,ctc in ctColLab]
    spaceLegEle = [Line2D([0], [0], marker='o', lw=0, color='white', markeredgecolor='white', label="")]
    btLegendEle = [Line2D([0], [0], color="white", marker=btShape, lw=0, markeredgecolor=btc, label=btLabel) for i,(btLabel,btc,btShape) in enumerate(btColLab)]

    ctColors = np.array([cellTypeColorDict[ct][1] for ct in ctVals])
    btColors = np.array([batchColorDict[bt][1] for bt in btVals])

    legendEle = btLegendEle + spaceLegEle + ctLegendEle

    fig, ax = plt.subplots()
    plt.legend(handles=legendEle)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    if(save):
        fig.savefig(f"{save}/legend_{name}.svg", format="svg")
    
    #return(legendEle, ctColors, btColors)
