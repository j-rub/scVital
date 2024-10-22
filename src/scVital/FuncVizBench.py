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

import FuncAUC as auc


colorDict = {"Mouse":"#F29242","Human":"#5AAA46","Zebrafish":"#4ad102",
			 "mouse":"#F29242","human":"#5AAA46",
			"Batch_1":"#d102c7","Batch_2":"#02cad1",
			"data_3p":"#d102c7","data_5p":"#02cad1",
			"m":"#d102c7","h":"#02cad1"}
#{"mouse":"r","UPS2236":"b","MFH9":"g", "ctl":"blue","wk1":"orange","wk3":"green","wk4":"red"}


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
		adata = auc.checkPairsAddSimple(adata, batchLabel, cellLabel)
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

def changeFileName(inFile, param, paramVal, pramSplit="_", valSplit = "~"):
	newFileName = ""
	for x in inFile.split(pramSplit):
		partFile = valSplit
		k = x.split(valSplit)
		if (k[0]==param):
			k[1] = str(paramVal)
		together = valSplit.join(k)
		newFileName = newFileName+pramSplit+together
	return(newFileName[1:])

def vizStats(statFile):
	stats = pd.read_csv(statFile,usecols=[1,2]).T
	vizStatsDF(stats, statFile)

def vizStatsDF(stats, statFile):
	stats = stats.round(3)
	
	# Create a figure and axes
	fig, ax = plt.subplots()

	# Set the number of rows and columns
	num_rows = stats.shape[0]
	num_cols = stats.shape[1]

	# Add a colorbar
	#cbar = plt.colorbar(im)
	for i in range(num_rows):
		for j in range(num_cols):
			value = stats.iloc[i, j]
			if(i==0):
				color = 'green' if value > 0.05 else 'red'
			else:
				color = 'green' if value > 1.25 else 'red'
			#print(f"values {value.round(3)} color {color}")
			rect = plt.Rectangle((j, i), 2, 2, facecolor=color, edgecolor='black', linewidth=1)
			ax.add_patch(rect)

	# Set the tick labels
	ax.set_xticks(np.arange(num_cols) + 0.5)
	ax.set_yticks(np.arange(num_rows) + 0.5)
	ax.set_xticklabels(stats.columns)
	ax.set_yticklabels(stats.index)

	ax.set_xticks(np.arange(num_cols)+1, minor = True)
	ax.set_yticks(np.arange(num_rows)+1, minor = True)

	# Loop over data dimensions and create text annotations
	for i in range(stats.shape[0]):
		for j in range(stats.shape[1]):
			text = ax.text(j+0.5, i+0.5, stats.values[i, j].round(3), ha='center', va='center', color='black', fontsize=12)

	# Set aspect ratio
	ax.set_aspect('equal')

	# Set title and labels
	ax.set_title("Stats")
	plt.xlabel("Clusters")
	plt.ylabel("Metrics")
	fig.savefig(".".join([statFile.split(".csv")[0],"png"]))
	plt.close(fig)

def vizMetrics(metricFile):
	metricDf = pd.read_csv(metricFile, index_col=0).T
	vizMetricsDF(metricDf, metricFile)

def vizMetricsDF(metricDf, metricFile):
	fig, ax = plt.subplots()

	num_rows = metricDf.shape[0]
	num_cols = metricDf.shape[1]

	paletteNum = 50
	rgCmap = sns.diverging_palette(10, 133, n=paletteNum+1)

	for i in range(num_rows):
		for j in range(num_cols):
			value = metricDf.iloc[i, j]
			rect = plt.Rectangle((j, i), 4, 4, edgecolor='black', linewidth=1, facecolor=rgCmap[int(float(value)*paletteNum)])
			ax.add_patch(rect)

	# Set the tick labels
	ax.set_xticks(np.arange(num_cols) + 0.5)
	ax.set_yticks(np.arange(num_rows) + 0.5)
	ax.set_xticklabels(metricDf.columns)
	ax.set_yticklabels(metricDf.index)

	ax.set_xticks(np.arange(num_cols)+1, minor = True)
	ax.set_yticks(np.arange(num_rows)+1, minor = True)

	# Loop over data dimensions and create text annotations
	for i in range(metricDf.shape[0]):
		for j in range(metricDf.shape[1]):
			text = ax.text(j+0.5, i+0.5, metricDf.values[i, j], ha='center', va='center', color='black', fontsize=12)

	# Set aspect ratio
	ax.set_aspect('equal')

	# Set title and labels
	ax.set_title("Metrics")
	plt.xlabel("Clusters")
	plt.ylabel("Metrics")
	fig.savefig(".".join([metricFile.split(".csv")[0],"png"]))
	plt.close(fig)

def vizFullViz(outClustStatDir, height=24, width=24, save=True):
	vizs=["normal","scVital","BBKNN","Harmony"]
	fig, axs = plt.subplots(len(vizs), 3, gridspec_kw={'width_ratios': [3, 1, 1]})
	for i, viz in enumerate(vizs):
		#if(viz=="ae"):
		#	axs[i,0].imshow(img.imread(f"{outClustStatDir}LossPlots.png"))
		#axs[i,1].imshow(img.imread(f"{outClustStatDir}latentHeatmap_{viz}.png"))
		axs[i,0].imshow(img.imread(f"{outClustStatDir}umap_{viz}_{viz}.svg"))
		vizStats(f"{outClustStatDir}stats_{viz}.csv")
		axs[i,1].imshow(img.imread(f"{outClustStatDir}stats_{viz}.png"))
		vizMetrics(f"{outClustStatDir}metrics_{viz}.csv")
		axs[i,2].imshow(img.imread(f"{outClustStatDir}metrics_{viz}.png"))
		#vizMetrics(f"{outClustStatDir}metrics_{viz}.csv")
		#axs[i,2].imshow(img.imread(f"{outClustStatDir}metrics_{viz}.png"))

	for x in range(len(axs)):
		for y in range(len(axs[1])):
			axs[x,y].xaxis.set_tick_params(labelbottom=False)
			axs[x,y].yaxis.set_tick_params(labelleft=False)
			axs[x,y].set_xticks([])
			axs[x,y].set_yticks([])
			axs[x,y].axis("off")

	fig.set_figheight(height)
	fig.set_figwidth(width)
	#fig.tight_layout()
	if(save):
		fig.savefig(f"{outClustStatDir}CombinedPlots.png")
		plt.close(fig)
	return(fig)

def plotMetricBarByData(df, label, outDir=None):
	numDatas = df.shape[0]
	dataNames = df.index

	numInteg = df.shape[1]
	integNames = df.columns

	index = np.arange(numDatas)
	width = 1/(numInteg+1)

	bars = np.empty(numInteg, dtype=object)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i, integ in enumerate(integNames):
		integVals = np.array(df.loc[:,integ])
		bar = ax.bar(index + width*i, integVals, width)#, color = 'r')
		bars[i] = bar

	ax.set_xlabel("Datasets")
	ax.set_ylabel(label)
	ax.set_title(f"{label}s")
	ax.set_xticks(index + (width*((numInteg-1)/2)), dataNames, rotation=80)
	legend = ax.legend(bars, integNames, loc='center right', bbox_to_anchor=(1.3,0.5))
	fig.savefig(f'{outDir}/{label}_barByData.svg', bbox_inches='tight')
	plt.close(fig)


def plotMetricBarByInteg(df, label, outDir=None):
	numDatas = df.shape[0]
	dataNames = df.index

	numInteg = df.shape[1]
	integNames = df.columns

	index = np.arange(numInteg)
	width = 1/(numDatas+1)

	bars = np.empty(numDatas, dtype=object)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i, data in enumerate(dataNames):
		dataVals = np.array(df.loc[data,:])
		bar = ax.bar(index + width*i, dataVals, width)#, color = 'r')
		bars[i] = bar

	ax.set_xlabel("Integration")
	ax.set_ylabel(label)
	ax.set_title(f"{label}s")
	ax.set_xticks(index + (width*((numDatas-1)/2)), integNames,rotation=80)
	legend = ax.legend(bars, dataNames, loc='center right',bbox_to_anchor=(1.3,0.5))
	fig.savefig(f'{outDir}/{label}_barByInteg.svg', bbox_inches='tight')
	plt.close(fig)

def plotScale(scaleDf, outDir=None):
	#scaleDf = (conDict["scale"]/60)

	dataOrder = np.argsort(scaleDf["Size"]).tolist()[::-1]
	datasets = np.array(scaleDf.index.tolist())[dataOrder]
	
	fig, ax = plt.subplots(figsize=(10, 6))
	
	for intAlg in scaleDf.columns[1:].tolist():
		scale = np.array(scaleDf[intAlg])[dataOrder]
		ax.plot(datasets, scale, label=intAlg)
	
	# Adding title and labels
	ax.set_title('Algorithm Performance Comparison')
	ax.set_xlabel('Datasets')
	ax.set_yscale('log')
	ax.set_ylabel('Time  log(minutes)')
	
	# Adding a legend
	ax.legend(loc='center right', bbox_to_anchor=(1.2,0.5))
	
	fig.savefig(f'{outDir}/scale.svg', bbox_inches='tight')
	plt.close(fig)

def df2StackBar(clustBatch, neighborsKey, label, ax):
	clustBatchCount = pd.DataFrame(clustBatch.value_counts(sort=False))
	clusters = np.unique(clustBatch[neighborsKey]).tolist()
	batches = np.unique(clustBatch[label]).tolist()

	counts = pd.DataFrame(np.zeros((len(clusters),len(batches))), index=clusters, columns=batches)

	for clust in clusters:
		for bat in batches:
			try:
				val = clustBatchCount.loc[(clust,bat)].iloc[0]
			except:
				val = 0
			counts.loc[clust,bat] = val

	numClust = len(clusters)#len(adata.obs[neighborsKey].cat.categories)
	rangeClusts = range(0,numClust)

	#pdb.set_trace()

	bott=np.zeros(numClust)
	for bat in counts:
		vals=counts[bat].values
		name=counts[bat].name
		ax.bar(rangeClusts, vals, bottom=bott, label=name)#, color=colorDict[name])
		bott = bott+vals

	#pdb.set_trace()

	ax.set_title(f"# of Cells of each Cluster by {label}") 
	#ax.set_xlabel("Cluster")#neighborsKey
	ax.set_ylabel("# cells")
	ax.legend(loc='center right', bbox_to_anchor=(1.3,0.5))



def vizFracNkBet(clustBatch, neighborsKey, label, kbetScore, outDir=None):
	rangeClusts = range(0,len(kbetScore))

	fig, axs = plt.subplots(2, 1, figsize=(5, 10))

	df2StackBar(clustBatch, neighborsKey, label, axs[0])

	axs[1].bar(rangeClusts, list(kbetScore["kBet"]))
	axs[1].axhline(0.05, color="black", linestyle="--")
	#axs[1].set_ylim(top=1)
	axs[1].set_title("kBet")
	axs[1].set_xlabel("Cluster")#neighborsKey
	axs[1].set_ylabel("kBet Score")

	#fig.legend()
	if(not outDir == None):
		plt.savefig(f'{outDir}/{neighborsKey}_ClustByBatchWkBet.svg')
		plt.close(fig)

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

def vizAllStats(allStats, name="", outDir=None):
    matrix = allStats.values
    xLabels = allStats.index
    yLabels = allStats.columns

    # Create a custom colormap for the first three rows
    RdYlGnCmap = matplotlib.colormaps['RdYlGn']

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    
    # Plot rectangles for each value
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = RdYlGnCmap(getCmapValue(value,matrix[i, :]))
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            textColor='black'
            if(0.212*color[0]+0.7152*color[1]+0.0722*color[3]<0.6):
                textColor='white'
            ax.text(j, i, f"{value:.2f}", ha='center', va='center', color=textColor)
    
    # Set axis labels
    ax.set_xticks(np.arange(-0.5,matrix.shape[1],0.5), [yLabels[i//2] if i%2 ==1 else "" for i,_ in enumerate(np.arange(-0.5,matrix.shape[1],0.5))])
    ax.set_yticks(np.arange(-0.5,matrix.shape[0],0.5), [xLabels[i//2] if i%2 ==1 else "" for i,_ in enumerate(np.arange(-0.5,matrix.shape[0],0.5))])
    ax.set_xticklabels(ax.get_xticklabels(),rotation=70)
    
    # Set plot title
    ax.set_title("Integration Statistics")
    #fig.colorbar(np.arange(0,1,0.2),ax=ax)
    
    # Show the plot
    plt.show()
    
    fig.savefig(f"{outDir}/allStats_{name}.svg", format="svg")


def plotOneUmap(title,x, y, c, edgecolors, name="", linewidths=0.2, s=2, alpha=0.5, outDir=None):
    fig, ax = plt.subplots(1, figsize= (4,4), dpi=300)

    ax.scatter(x, y, c=c, edgecolors=edgecolors, linewidths=linewidths, s=s, alpha=alpha)
    ax.set_xlabel('UMAP 0')
    ax.set_ylabel('UMAP 1')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    if("/" in title):
        title = "".join(title.split("/"))
    fig.savefig(f"{outDir}/umap_{title}_{name}.png", format="png")


def makeLegend(ctVals, btVals, cellTypeColorDict, batchColorDict, name="", outDir=None):
    
    ctColLab = cellTypeColorDict.values()
    btColLab = batchColorDict.values()

    ctLegendEle = [Line2D([0], [0], color=ctc, marker='o', lw=0, label=ctLabel) for ctLabel,ctc in ctColLab]
    spaceLegEle = [Line2D([0], [0], marker='o', lw=0, color='white', markeredgecolor='white', label="")]
    btLegendEle = [Line2D([0], [0], color=btc, marker='o', lw=0, markeredgecolor='black', label=btLabel) for btLabel,btc in btColLab]

    ctColors = np.array([cellTypeColorDict[ct][1] for ct in ctVals])
    btColors = np.array([batchColorDict[bt][1] for bt in btVals])

    legendEle = btLegendEle + spaceLegEle + ctLegendEle

    fig, ax = plt.subplots(1, dpi=300)
    plt.legend(handles=legendEle)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    fig.savefig(f"{outDir}/legend_{name}.svg", format="svg")
    
    return(legendEle, ctColors, btColors)

def plotInteg(inUmaps, titles, ctColors, btColors, shuff, outDir=None):
    for i, iUmap in enumerate(inUmaps):
        plotOneUmap(titles[i], x=iUmap[shuff, 0],y=iUmap[shuff, 1], c=ctColors[shuff], edgecolors=btColors[shuff])

def plotCbar(title,name, norm, cmap):
    fig, ax = plt.subplots(1, dpi=300)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    title = "".join(title.split("/"))
    fig.savefig(f"{outDir}/cbarleg_{title}_{name}.svg", format="svg")



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



#expTotal = 8
#for j,prop in enumerate(gProp):
#	valProp = np.round(prop)
#	sumProp = sum(valProp)
#	if(not sumProp == expTotal):
#		diff = expTotal - sumProp
#		ind = np.arange(len(valProp))
#		np.random.shuffle(ind)
#		for i in range(abs(diff).astype(int)):
#			valProp[ind[i]] = valProp[ind[i]] + 1*np.sign(diff)
#	gProp[j] = valProp



