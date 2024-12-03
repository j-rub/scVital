#NOT IMPEMENTED

import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns

from matplotlib import cm
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches

from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import RocCurveDisplay, roc_curve
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster import hierarchy 
from scipy.stats import power_divergence
from scipy.sparse import isspmatrix
from scipy.spatial import distance

#n_pcs -> last layer dims
def kBet(adata,  batchID= "batch", numNeighbors = 50, interations = 20, cellsPerIter = 50, rep="X_pcAE",n_pcs=8):
	uniqueBatch = np.array(adata.obs[batchID].cat.categories)
	numBatch = len(uniqueBatch)
	batchDict = dict(zip(uniqueBatch,range(0,numBatch)))
	batch = np.array([batchDict[x] for x in adata.obs[batchID]])
	uniqueBatch = np.unique(batch)

	totalPropCells = [np.mean(batch == uniqueBatch[i]) for i in range(numBatch)]
	totalProp = np.multiply([totalPropCells]*cellsPerIter,numNeighbors)#np.round()
	totalPvalue = np.zeros(interations*cellsPerIter)
	
	if any([prop > 0.90 for prop in totalPropCells]):
		return(-0.01)

	#TODO change neighbor matrix to latent name and neighbor matrix
	
	sc.pp.neighbors(adata, n_neighbors=numNeighbors+1, n_pcs=n_pcs, use_rep=rep, key_added=f"leiden{numNeighbors+1}")
	neighborMatrix = adata.obsp[f"leiden{numNeighbors+1}_distances"] 
	#neighborMatrix = adata.obsp[f"{rep.split("_")[1]}_distances"] 

	for i in range(interations):
		indices = np.random.choice(len(adata), size=cellsPerIter)
		propCell = [np.bincount(batch[neighborMatrix[indices].nonzero()[1][neighborMatrix[indices].nonzero()[0] == j]], minlength=numBatch) for j in range(cellsPerIter)]
		chiSqOutPvalue = 0
		try:
			chiSqOut = power_divergence(propCell, f_exp=totalProp, axis=1, lambda_=1)
			chiSqOutPvalue = chiSqOut.pvalue
		except:
			chiSqOutPvalue = -0.03
		totalPvalue[i*cellsPerIter:i*cellsPerIter+cellsPerIter] = chiSqOutPvalue
	return(np.median(totalPvalue))

def inverSampson(countMatrix):
	n = sum(countMatrix)
	typeSum = 0
	for i in countMatrix:
		if i > 0:
			typeSum = typeSum + (i*(i-1))
	return((n*(n-1))/typeSum)

#n_pcs -> out layer dims
def getClustMetrics(adata, clusterID, batchID= "batch", numNeighbors = 50, interations = 20, cellsPerIter = 40, rep="X_pcAE",n_pcs=8):
	clusters = list(adata.obs[clusterID].cat.categories)
	metricNames = ["kBet"] #,"iLSI"
	stats = pd.DataFrame(np.zeros((len(clusters),len(metricNames))), index=clusters, columns=metricNames)
	for i, ctype in enumerate(clusters):
		adataClust = adata[[cellType == ctype for cellType in adata.obs[clusterID]],:]
		#pdb.set_trace()
		stats.loc[ctype,"kBet"] = kBet(adataClust, batchID=batchID, numNeighbors=numNeighbors, 
			interations=interations, cellsPerIter=cellsPerIter, rep=rep, n_pcs=n_pcs)
		#stats.loc[ctype,"iLSI"] = inverSampson(adataClust.obs[batchID].value_counts())

	stats["SortCluster"] = stats.index.astype(int)
	stats = stats.sort_values("SortCluster")
	return(stats)
		#print(f"kBet p-value: {np.around(kBetScore, decimals=2)}")
		#print(f"invSamp index: {np.around(isIndex, decimals=2)}")


def testClustAndStats(adata, umapKey, neighborsKey, pcaRep, 
					  cellLabel, batchLabel, res, numOutLayer, outClustStatDir,
					  nNeighborsUMAP=25, nNeighborsKbet=45, inters=25, cellsPerIter=30,
					  save=True):
	sc.set_figure_params(scanpy=True, dpi_save=150, fontsize=24, format='svg', frameon=False,transparent=True)
	
	batchUmapFilename = f"_{batchLabel}_{neighborsKey}"
	neighUmapFilename = f"_{neighborsKey}"
	trueUmapFilename = f"_{cellLabel}_{neighborsKey}"
	if(not save):
		batchUmapFilename, neighUmapFilename, trueUmapFilename = False, False, False

	#adata.uns[f'{batchLabel}_colors'] = ['#FF7F50','#76EEC6']
	if (neighborsKey=="BBKNN"):
		import bbknn
		startTrain = time.process_time() 
		sc.external.pp.bbknn(adata, batch_key=batchLabel, use_rep="X_pca")#, neighbors_within_batch=4, n_pcs=numOutLayer)#, key_added=umapKey)
		endTrain = time.process_time()
		scale = pd.DataFrame(np.array([[adata.X.size, endTrain-startTrain]]), columns=["Size", "BBKNN"])
		scale.to_csv(f"{outClustStatDir}scale_BBKNN.csv")
	else:
		sc.pp.neighbors(adata, n_neighbors=nNeighborsUMAP, n_pcs=numOutLayer, use_rep=pcaRep, key_added=umapKey)

	sc.pp.pca(adata, svd_solver="arpack")
	sc.tl.umap(adata, neighbors_key = umapKey)

	if(cellLabel == "None"):
		cellLabel = np.nan
		trueUmapFilename = False

	#res = 0.5
	#maxAri = 0
	
	if(not pd.isna(cellLabel)):
		#adata = auc.checkPairsAddSimple(adata, batchLabel, cellLabel)
		cellLabel = "overlapLabel"
		#print(" \t YES TRUE CLUSTERING")
		maxARI = 0
		maxRes = 0
		#sc.pp.neighbors(adata, n_neighbors=nNeighborsUMAP, n_pcs=numOutLayer, use_rep=pcaRep, key_added=umapKey)
		
		for i, res in enumerate(np.arange(0.05, 0.8, 0.05)):
			if (neighborsKey=="BBKNN"):
				res=res*0.1
			
			try:
				sc.tl.leiden(adata, resolution=res, key_added = neighborsKey, neighbors_key = umapKey)#, flavor="igraph", n_iterations=2,  directed=False)
			except:
				sc.pp.neighbors(adata, n_neighbors=nNeighborsUMAP, n_pcs=numOutLayer, use_rep=pcaRep, key_added=umapKey)
				sc.tl.leiden(adata, resolution=res, key_added = neighborsKey, neighbors_key = umapKey)#, flavor="igraph", n_iterations=2,  directed=False)

			newARI = metrics.adjusted_rand_score(adata.obs[cellLabel], adata.obs[neighborsKey])

			if (newARI > maxARI):
				maxARI = newARI
				maxRes = res
		
			if(newARI==1 or maxARI > (newARI+0.1)):
				break
		
		sc.pp.neighbors(adata, n_neighbors=nNeighborsUMAP, n_pcs=numOutLayer, use_rep=pcaRep, key_added=umapKey)
		sc.tl.leiden(adata, resolution=maxRes, key_added = neighborsKey, neighbors_key = umapKey)
		
		#sc.pl.umap(adata, color = [cellLabel], ncols = 1, show=False, save=trueUmapFilename, legend_fontsize="xx-small")#palette="tab10",
		metricDF = getClusterMetricDF(labels_true = adata.obs[cellLabel], labels = adata.obs[neighborsKey], neighborsKey=neighborsKey)
		metricDF.to_csv(f"{outClustStatDir}metrics_{neighborsKey}.csv")
	else:
		sc.tl.leiden(adata, resolution = res, key_added = neighborsKey, neighbors_key = umapKey)
		
	#pdb.set_trace()
	#sc.pl.umap(adata, color = [batchLabel], ncols = 1, show=False, save=batchUmapFilename, legend_fontsize="xx-small") #, palette=colorDict,
	#sc.pl.umap(adata, color = [neighborsKey], ncols = 1, show=False, save=neighUmapFilename, legend_fontsize="xx-small")#palette="Set2",

	sc.pl.umap(adata, color = [neighborsKey,batchLabel,cellLabel], ncols = 3, show=False, save=neighUmapFilename, legend_fontsize="xx-small")#palette="Set2",
	sc.pl.pca(adata, color = [neighborsKey,batchLabel,cellLabel], ncols = 3, show=False, save=neighUmapFilename, legend_fontsize="xx-small")
	
	#batchColorDict = dict(zip(adata.obs[batchLabel].unique(), adata.uns[f'{batchLabel}_colors']))
	#cellTColorDict = dict(zip(adata.obs[cellLabel].unique(), adata.uns[f'{cellLabel}_colors']))
	#bCmap = [batchColorDict[x] for x in adata.obs[batchLabel]]
	#cCmap = [cellTColorDict[x] for x in adata.obs[cellLabel]]
	#latentViz = sns.clustermap(adata.obsm[pcaRep][:,0:numOutLayer], cmap="bwr",row_colors=[bCmap,cCmap],col_cluster=False).figure
	#latentViz.savefig(f"{outClustStatDir}latentHeatmap_{neighborsKey}.png")

	aeStats = getClustMetrics(adata, neighborsKey, batchID = batchLabel,
								numNeighbors = nNeighborsKbet, 
								interations = inters, 
								cellsPerIter = cellsPerIter, 
								rep = pcaRep,
								n_pcs = numOutLayer)
	aeStats.to_csv(f"{outClustStatDir}stats_{neighborsKey}.csv")

	clustBatch = adata.obs[[neighborsKey,batchLabel]]
	vizFracNkBet(clustBatch=clustBatch, neighborsKey=neighborsKey, label=batchLabel, kbetScore=aeStats, outDir=outClustStatDir)


	
#	scale = pd.DataFrame(np.array([[adata.X.size, ]]), columns=["Size", "Time"])
#	scale.to_csv(f"{outClustStatDir}scale_{neighborsKey}.csv")


def getClusterMetricDF(labels_true, labels, neighborsKey):
	metricDict = {
				#   'Homogeneity': metrics.homogeneity_score,
				#	'Completeness': metrics.completeness_score,
				#	'V-measure': metrics.v_measure_score,
					'FM':metrics.fowlkes_mallows_score,
					'ARI': metrics.adjusted_rand_score,
				#	'Adjusted Mutual Information': metrics.adjusted_mutual_info_score
				}
	metricOut = [f"{metricDict[metricF](labels_true, labels):0.3}" for metricF in metricDict]
	metricDF = pd.DataFrame(metricOut, columns=[neighborsKey], index=list(metricDict.keys()))
	return(metricDF)


def heirSimi(adata, latent, cellLabel, allCellTypes, clustMet = "cosine"):
	latLen = range(adata.obsm[latent].shape[1])

	clustDist2 = pd.DataFrame(np.zeros((len(allCellTypes),len(latLen))), columns=latLen, index=allCellTypes)
	for ct in clustDist2.index:
		ctMean = np.mean(adata[adata.obs[cellLabel]==ct].obsm[latent], axis=0)
		clustDist2.loc[ct] = ctMean
	#clustDist2

	sns.clustermap(clustDist2, metric = clustMet)


def getAllStats(dirName,datasetName,paramName):
    #allStats = pd.DataFrame(np.zeros((4,6)),index=["ARI","FM","nKbet","LSS"],columns=["scVital","normal","BBKNN","Harmony","scVI","scDREAMER"])
    allStats = pd.DataFrame(np.zeros((4,6)),index=["Runtime\n(min)  ","ARI","FM","LSS"],columns=["scVital","normal","BBKNN","Harmony","scVI","scDREAMER"])

    lss = pd.read_csv(f"{dirName}/{datasetName}/{paramName}/figures/fullAucScores.csv",index_col=0)
    for lat in allStats.columns:
        #for stat in allStats.index:
        met = pd.read_csv(f"{dirName}/{datasetName}/{paramName}/figures/metrics_{lat}.csv",index_col=0)
        allStats.loc["ARI",lat] = met.loc["ARI",lat]    
        allStats.loc["FM",lat] = met.loc["FM",lat]
 
        try:
            scale = pd.read_csv(f"{dirName}/{datasetName}/{paramName}/figures/scale_{lat}.csv",index_col=0)
        except:
            scale = pd.read_csv(f"{dirName}/{datasetName}/{paramName}/figures/scale_No_Integration.csv",index_col=0)
            
        allStats.loc["Runtime\n(min)  ",lat] = (scale[lat][0])/60
        
        #kbet = pd.read_csv(f"{dirName}/{datasetName}/{paramName}/figures/stats_{lat}.csv",index_col=0)
        #allStats.loc["nKbet",lat] = sum(kbet["kBet"]>0.05)/len(kbet["kBet"])
        
        allStats.loc["LSS",lat] = lss.loc[:,f"X_{lat}"].values[0]
    
    return(allStats)

def getCmapValue(value, vals):
    maxVal = np.max(vals)
    if(maxVal > 1):
        return ((-value+maxVal)/(-np.min(vals)+maxVal))
    return value



#    FROM scib-metrics almost exatly
def plot_results_table(df, show = False, save_dir = None):
    """Plot the benchmarking results.
    Parameters
    ----------
    show
        Whether to show the plot.
    save_dir
        The directory to save the plot to. If `None`, the plot is not saved.
    """
    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.plots import bar
    embeds = list(df.index[:-1])
    num_embeds = len(embeds)
    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=cm.PRGn, num_stds=2.5)
    _LABELS = "labels"
    _BATCH = "batch"
    _METRIC_TYPE = "Metric Type"
    _AGGREGATE_SCORE = "Aggregate score"
    # Do not want to plot what kind of metric it is
    plot_df = df.drop(_METRIC_TYPE, axis=0)
    # Sort by total score
    plot_df = plot_df.sort_values(by="Total", ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    # Split columns by metric type, using df as it doesn't have the new method col
    score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
    other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
    ]
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn(plot_df[col]),
            group=df.loc[_METRIC_TYPE, col],
            formatter="{:.2f}",
        )
        for i, col in enumerate(other_cols)
    ]
    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=df.loc[_METRIC_TYPE, col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols)
    ]
    # Allow to manipulate text post-hoc (in illustrator)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
    if show:
        plt.show()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "scib_results.svg"), facecolor=ax.get_facecolor(), dpi=300)
        plt.close(fig)
    return tab




