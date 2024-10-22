import os
import time
import numpy as np
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns

from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches

from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.manual_seed(18)

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

import AutoEncoder as ae
import Discriminator as dis 


def getAdataX(adata):
	adataX = adata.X
	if(isspmatrix(adataX)):
		adataX = adataX.todense()
	return(torch.tensor(adataX))

def getTrainLabel(inDataFile, inLabelFiles, obsLabel="batch", isAdata=True):
	if(isAdata):
		adata = sc.read_h5ad(inDataFile)
		adata.obs_names_make_unique()
		inData = getAdataX(adata)

		batch = np.array(adata.obs[obsLabel])
		uniqueBatch = np.unique(batch)
		numBatch = len(uniqueBatch)
		batchDict = dict(zip(uniqueBatch,range(0,numBatch)))
		batchNum = np.array([batchDict[x] for x in batch])

		inLabels = torch.tensor(batchNum)
		inLabels = inLabels.view(len(inLabels), 1)
		#TODO fix bug here
		#when multiple batches of different species
		geneType = np.full(len(adata.var_names),"all", dtype=object)
		batchSpeciesDict = {batch:"all" for batch in uniqueBatch}
		if(np.isin(["species"],adata.obs.columns.values)[0]):
			uniqueSpecies = adata.obs["species"].cat.categories
			batchSpecies = np.unique(["!".join(x) for x in adata.obs[[obsLabel,"species"]].values])
			batchSpeciesDict = {indBatchSpecies.split("!")[0]:indBatchSpecies.split("!")[1] for indBatchSpecies in batchSpecies}# Batch:species
			for i,gene in enumerate(adata.var_names):
				gsplit = gene.split("/")
				#lab = "u"
				if ' ' in gsplit:
					for j,g in enumerate(gsplit):
						if not g==" ":
							geneType[i] = uniqueSpecies[j] #uniqueBatch[j] #lab
				#else:
				#	lab = "all"
				#geneType[i] = lab
			rangeGenes = range(len(geneType))			
		batchSpecLabIndex = []
		for batchLab in uniqueBatch:
			speciesSpecGenes = np.logical_or(geneType=="all",geneType==batchSpeciesDict[batchLab])
			batchSpecLabIndex.append(list(speciesSpecGenes))
		
	else:
		inData = torch.tensor(np.loadtxt(inDataFile,delimiter="\t"), dtype=torch.float32)
		inLabels = torch.tensor(np.loadtxt(inLabelFiles[0],delimiter="\t",ndmin=2), dtype=torch.float32)
		batchSpecLabIndex = torch.tensor(np.loadtxt(inLabelFiles[1],delimiter="\t",ndmin=2), dtype=torch.float32)

	return(inData, inLabels, batchSpecLabIndex)

def addUbiqGenes(inData, addedGenes = 10, mean = 2.0, std=0.1):
	normHigh = torch.distributions.normal.Normal(torch.tensor([mean]), torch.tensor([std]))
	normH = normHigh.rsample(sample_shape=(inData.size()[0],addedGenes))[:,:,0]
	inData = torch.cat((inData, normH), dim=1)
	return inData

def getDataLoader(inData, inLabels, batchSize, trainPer):
	#inData, inLabels = getTrainLabel(inFileData, inFileLabel)
	LabeledData = torch.cat((inLabels,inData),axis=1)
	#trainSize = int(np.round(LabeledData.size()[0]*trainPer))
	#validSize = LabeledData.size()[0]-trainSize
	#ldTrainX, ldValX = torch.utils.data.random_split(LabeledData, [trainSize, validSize])
	#ldTrainDataLoader = DataLoader(LabeledData, batch_size=batchSize, shuffle=True)
	layer1Dim = inData.size()[1]
	numSpeices = len(inLabels.unique())
	#return(ldTrainDataLoader, ldValX, layer1Dim, numSpeices)
	return(LabeledData, layer1Dim, numSpeices)

def getDataLoader(inData, inLabels):
	LabeledData = torch.cat((inLabels,inData),axis=1)
	layer1Dim = inData.size()[1]
	numSpeices = len(inLabels.unique())
	return(LabeledData, layer1Dim, numSpeices)

def makeAutoencoder(layerDims, numSpeices=2):
	encoder = ae.Encoder(layerDims, numSpeices)
	decoder = ae.Decoder(layerDims, numSpeices)
	return ae.EncoderDecoder(encoder,decoder)

def makeDiscriminator(discriminatorDims, numSpeices=2):
	return dis.Discriminator(discriminatorDims, numSpeices)

def klCycle(start, stop, n_epoch, n_cycle=4):
	ratio = (n_cycle-1)/n_cycle
	kl = np.full(n_epoch,stop)
	period = n_epoch/n_cycle
	step = (stop-start)/(period*ratio) # linear schedule

	for c in range(n_cycle):
		v , i = start , 0
		while v <= stop and (int(i+c*period) < n_epoch):
			kl[int(i+c*period)] = v
			v += step
			i += 1
	return kl


def trainVAEABC(autoencoder, autoencoderOpt, reconstructionLossFunc, aeSchedLR,
				discriminator, discriminatorOpt, discriminatorLossFunc, discSchedLR,
				LabeledData, batchSize, numEpoch, 
				reconCoef = 10, klCoef = 0.1, discCoef = 1,
				numSpeices=2, discTrainIter=5):
	autoencoder.train()
	discriminator.train()

	ldTrainDataLoader = DataLoader(LabeledData, batch_size=int(batchSize), shuffle=True)
	numMiniBatch = len(ldTrainDataLoader)

	discEpochLoss = np.full(numEpoch, np.nan)
	reconstEpochLoss = np.full(numEpoch, np.nan)
	trickEpochLoss = np.full(numEpoch, np.nan)
	klDivEpochLoss = np.full(numEpoch, np.nan)
	aeTotalEpochLoss = np.full(numEpoch, np.nan)

	prevTotalLoss = np.inf

	#KL Divergenace Cyclical Annealing
	klDivCoeff = klCycle(0, klCoef, numEpoch)

	for iEp in range(numEpoch):

		discTrainLoss = np.full(numMiniBatch, np.nan) 
		reconstTrainLoss = np.full(numMiniBatch, np.nan)
		trickTrainLoss = np.full(numMiniBatch, np.nan)
		klDivTrainLoss = np.full(numMiniBatch, np.nan)
		aeTotalTrainLoss = np.full(numMiniBatch, np.nan)

		for iBatch, ldData in enumerate(ldTrainDataLoader):
			#Load data 
			bRealLabels, bData = ldData[:,0].to(torch.int64), ldData[:,1:].to(torch.float32)
			
			bRealLabOneHot = F.one_hot(bRealLabels, num_classes=numSpeices).float()
			labeledBData = torch.cat((bData, bRealLabOneHot),axis=1)
			labels = np.unique(bRealLabels)
			
			#Train discriminator
			#get mouse and human data in latent space
			encoder = autoencoder.getEncoder()
			#pdb.set_trace()
			for i in range(discTrainIter):
				bLatent = encoder(labeledBData) #bData
				
				#Optimize discriminator
				discriminatorOpt.zero_grad()

				bDiscPred = discriminator(bLatent)
				bDiscRealLoss = discriminatorLossFunc(bDiscPred, bRealLabels)
				bDiscRealLoss.backward()

				discriminatorOpt.step()
				discSchedLR.step(iEp + iBatch / numMiniBatch)

			#Train generator
			autoencoderOpt.zero_grad()
			
			#encode mouse and human data in latent space
			bReconstData = autoencoder(labeledBData, bRealLabels, numSpeices) #bData

			#added
			#split input data and reconstructed data by batch and batch-specific genes
			#calculate reconstruction on on relavent data
			bReconstLoss = torch.tensor(0.0)
			batchSpecLabIndex = autoencoder.getGeneIndex()
			allCells, allGenes = bData.shape

			for i in labels: # for every batch
				sCells = (bRealLabels==i).reshape((allCells,1))                                           #Cells realting to label
				sGenes = torch.tensor(batchSpecLabIndex[i]).reshape((1,allGenes))                         #Genes relating to label
				numCells, numGenes = torch.sum(sCells), torch.sum(sGenes,axis=1)                          #number of cells and genes in of the label
				sMask = sCells * sGenes                                                                   #tensor of the mask to only take the genes realting to labels for the same cells
				bDataMasked = torch.masked_select(bData, sMask).reshape((numCells,numGenes))              #apply mask to input data
				reconDataMasked = torch.masked_select(bReconstData, sMask).reshape((numCells,numGenes))   #apply mask to reconstructed data
				bReconstLoss += (numCells/allCells)*reconstructionLossFunc(reconDataMasked, bDataMasked)  #calcualte reconstruction with masks and update total
			
			#bReconstLoss = reconstructionLossFunc(bReconstData, bData) 
			
			encoder = autoencoder.getEncoder()
			bLatent = encoder(labeledBData) #bData

			#train discriminator and get preds try and subtract from total Loss
			bDiscPred = discriminator(bLatent)

			#bDiscWrongLoss = discriminatorLossFunc(bDiscPred, bRealLabels)
			bRealLabOneHot = F.one_hot(bRealLabels, num_classes=numSpeices).float()
			reconLatentEven = torch.ones_like(bRealLabOneHot)*(1/numSpeices)
			bDiscTrickLoss = discriminatorLossFunc(bDiscPred, reconLatentEven) #bRealLabels)
			
			#KL Div loss with N(0,1)
			klDivLoss = encoder.klDivLoss

			bRecTrickDiscLoss = reconCoef*bReconstLoss + klDivCoeff[iEp]*klDivLoss + discCoef*bDiscTrickLoss # + reconNbLoss
			bRecTrickDiscLoss.backward()
			
			#optimize generator
			autoencoderOpt.step()
			aeSchedLR.step(iEp + iBatch / numMiniBatch)

			discTrainLoss[iBatch] 		= bDiscRealLoss.item()
			reconstTrainLoss[iBatch] 	= reconCoef*bReconstLoss.item()
			trickTrainLoss[iBatch]		= discCoef*bDiscTrickLoss.item()
			klDivTrainLoss[iBatch]		= klDivCoeff[iEp]*klDivLoss.item()
			aeTotalTrainLoss[iBatch] 	= bRecTrickDiscLoss.item()

			#if iEp % 50 == 0 and iBatch % 20 == 0:
			#	print(f"Epoch={iEp}, batch={iBatch}, discr={np.nanmean(discTrainLoss):.4f}, total={np.nanmean(aeTotalTrainLoss):.4f}, recon={np.nanmean(reconstTrainLoss):.4f}, trick={np.nanmean(trickTrainLoss):.4f}, klDiv={np.nanmean(klDivTrainLoss):.4f}")

		discEpochLoss[iEp] 		= np.nanmean(discTrainLoss)
		reconstEpochLoss[iEp] 	= np.nanmean(reconstTrainLoss)
		trickEpochLoss[iEp] 	= np.nanmean(trickTrainLoss)
		klDivEpochLoss[iEp] 	= np.nanmean(klDivTrainLoss)
		aeTotalEpochLoss[iEp] 	= np.nanmean(aeTotalTrainLoss)
		
		#if iEp % 50 == 0:
		#	print(f"Epoch={iEp}, \t	discr={np.nanmean(discEpochLoss):.4f}, total={np.nanmean(aeTotalEpochLoss):.4f}, recon={np.nanmean(reconstEpochLoss):.4f}, trick={np.nanmean(trickEpochLoss):.4f}, klDiv={np.nanmean(klDivEpochLoss):.4f}")

		#Early Stopping
		totalLoss = np.mean(aeTotalEpochLoss[max(iEp-5,0):(iEp+1)])
		deltaLoss = np.abs(prevTotalLoss-totalLoss)
		if (deltaLoss < 1e-2 and iEp > 10):
			print(f' epoch:{iEp} delta:{deltaLoss}')
			break
		prevTotalLoss = totalLoss
		
		#aeSchedLR.step()
		#discSchedLR.step()

	lossDict = {"total":aeTotalEpochLoss,
				"recon":reconstEpochLoss,
				"trick":trickEpochLoss,
				"klDiv":klDivEpochLoss,
				"discr":discEpochLoss}

	return(autoencoder, discriminator, lossDict)


