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


import textwrap


def plotHVG(adata, minMean = 0.05, maxMean = 2.9, minDisp = 0.25, batchKey=None):
	fig, axs = plt.subplots(1, 2, figsize=(10, 5))

	means = adata.var[["means"]][adata.var[["means"]] > np.exp(-14)]#adata.var[["means"]],
	axs[0].hist(np.log(means), bins=100)#, log=True),
	axs[0].axvline(np.log(minMean), color='k', linestyle='dashed', linewidth=1)
	axs[0].axvline(np.log(maxMean), color='k', linestyle='dashed', linewidth=1)
	axs[0].set_title('Gene means counts')
	axs[0].set_xlabel('means')
	axs[0].set_ylabel('counts')

	dispNorm = adata.var[["dispersions_norm"]][adata.var[["dispersions_norm"]] > np.exp(-8)]#adata.var[["means"]],
	axs[1].hist(np.log(dispNorm), bins=100)#, log=True),
	axs[1].axvline(np.log(minDisp), color='k', linestyle='dashed', linewidth=1)
	axs[1].set_title('Gene dispersions counts')
	axs[1].set_xlabel('dispersions')
	axs[1].set_ylabel('counts')

	sc.pp.highly_variable_genes(adata, min_disp=minDisp, min_mean=minMean, max_mean=maxMean, batch_key=batchKey)
	print(sum(adata.var.highly_variable))
	if"highly_variable_intersection" in adata.var.names :
		print(sum(adata.var.highly_variable_intersection))



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

	bott=np.zeros(numClust)
	for bat in counts:
		vals=counts[bat].values
		name=counts[bat].name
		ax.bar(rangeClusts, vals, bottom=bott, label=name)#, color=colorDict[name])
		bott = bott+vals

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

