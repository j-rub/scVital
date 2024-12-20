#!/usr/bin/env python
#run train DL model
#comments and error checking amde with the help of copilot

import os
import sys
import time
import pickle
import warnings

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as an
from scipy.sparse import isspmatrix
from matplotlib import pyplot as plt


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

#from scVital.autoencoder import EncoderDecoder, Encoder, Decoder
#from scVital.discriminator import Discriminator

def makeScVital(
	adata: an.AnnData,
	batchLabel: str,
	miniBatchSize: int = 512,
	numEpoch: int = 64,
	learningRate: float = 1e-3,
	hid1: int = 1024,
	hid2: int = 128,
	latentSize: int = 12,
	discHid: int = 6,
	reconCoef: float = 2e1,
	klCoef: float = 1e-1,
	discCoef: float = 1e0,
	discIter: int = 5,
	earlyStop: float = 1e-2,
	train: bool=False,
	seed: int = 18,
	verbose: bool = True
) -> 'scVitalModel':
	"""
	Run the scVital model with the specified parameters.

	Parameters:
	adata (an.AnnData): Annotated data matrix.
	batchLabel (str): Label for batch processing.
	miniBatchSize (int, optional): Size of mini-batches for training. Default is 512.
	numEpoch (int, optional): Number of epochs for training. Default is 64.
	learningRate (float, optional): Learning rate for the optimizer. Default is 1e-3.
	hid1 (int, optional): Number of units in the first hidden layer. Default is 1024.
	hid2 (int, optional): Number of units in the second hidden layer. Default is 128.
	latentSize (int, optional): Size of the latent space. Default is 12.
	discHid (int, optional): Number of units in the discriminator hidden layer. Default is 6.
	reconCoef (float, optional): Coefficient for reconstruction loss. Default is 2e1.
	klCoef (float, optional): Coefficient for KL divergence loss. Default is 1e-1.
	discCoef (float, optional): Coefficient for discriminator loss. Default is 1e0.
	discIter (int, optional): Number of iterations for discriminator training. Default is 5.
	earlyStop (float, optional): Delta error to trigger early stopping. Default is 1e-2.
	seed (int, optional): Random seed for reproducibility. Default is 18.
	verbose (bool, optional): Flag for verbosity. Default is True.

	Returns:
	scVital: An instance of the scVital class initialized with the specified parameters.
	
	Raises:
	ValueError: If any of the input parameters are invalid.
	"""
	# Check valid inputs
	if not isinstance(adata, an.AnnData):
		raise ValueError("adata must be an AnnData object")
	if not isinstance(batchLabel, str):
		raise ValueError("batchLabel must be a string")
	if not isinstance(miniBatchSize, int) or miniBatchSize <= 0:
		raise ValueError("miniBatchSize must be a positive integer")
	if not isinstance(numEpoch, int) or numEpoch <= 0:
		raise ValueError("numEpoch must be a positive integer")
	if not isinstance(learningRate, float) or learningRate <= 0:
		raise ValueError("learningRate must be a positive float")
	if not isinstance(hid1, int) or hid1 <= 0:
		raise ValueError("hid1 must be a positive integer")
	if not isinstance(hid2, int) or hid2 <= 0:
		raise ValueError("hid2 must be a positive integer")
	if not isinstance(latentSize, int) or latentSize <= 0:
		raise ValueError("latentSize must be a positive integer")
	if not isinstance(discHid, int) or discHid <= 0:
		raise ValueError("discHid must be a positive integer")
	if not isinstance(reconCoef, float) or reconCoef < 0:
		raise ValueError("reconCoef must be a non-negative float")
	if not isinstance(klCoef, float) or klCoef < 0:
		raise ValueError("klCoef must be a non-negative float")
	if not isinstance(discCoef, float) or discCoef < 0:
		raise ValueError("discCoef must be a non-negative float")
	if not isinstance(discIter, int) or discIter <= 0:
		raise ValueError("discIter must be a positive integer")
	if not isinstance(earlyStop, float) or earlyStop < 0:
		raise ValueError("earlyStop must be a non-negative float")
	if not isinstance(train, bool):
		raise ValueError("train must be a boolean")
	if not isinstance(seed, int) or seed < 0:
		raise ValueError("seed must be a non-negative integer")
	if not isinstance(verbose, bool):
		raise ValueError("verbose must be a boolean")

	# Issue a warning if the learning rate is unusually high
	if learningRate > 1:
		warnings.warn("The learning rate is unusually high and may cause instability.", UserWarning)
	
	# Issue a warning if the adata is very large
	if adata.shape[1] > 4000:
		warnings.warn("The adata object has many genes consider subsetting on highly variable genes", UserWarning)

	# Initialize the scVital model with the provided parameters
	scVitalData = scVitalModel(
		adata, batchLabel, miniBatchSize, numEpoch, learningRate,
		hid1, hid2, latentSize, discHid, reconCoef, klCoef, discCoef,
		discIter, earlyStop, seed, verbose
	)

	# Train is true then train 
	if(train):
		scVitalData.runTrainScVital(self)

	# Return the initialized scVital model
	return scVitalData


class scVitalModel(object):
	def __init__(self, adata, batchLabel, miniBatchSize, numEpoch, learningRate,
				hid1, hid2, latentSize, discHid, reconCoef, klCoef, discCoef, discIter, 
				earlyStop, seed, verbose):
		"""
		Initialize the model with the given parameters.

		Parameters:
		adata (AnnData): Annotated data matrix.
		batchLabel (str): Label for batch processing.
		miniBatchSize (int): Size of mini-batches for training.
		numEpoch (int): Number of epochs for training.
		learningRate (float): Learning rate for the optimizer.
		hid1 (int): Number of units in the first hidden layer.
		hid2 (int): Number of units in the second hidden layer.
		latentSize (int): Size of the latent space.
		discHid (int): Number of units in the discriminator hidden layer.
		reconCoef (float): Coefficient for reconstruction loss.
		klCoef (float): Coefficient for KL divergence loss.
		discCoef (float): Coefficient for discriminator loss.
		discIter (int): Number of iterations for discriminator training.
		earlyStop (float): Delta error to trigger early stopping.
		seed (int): Random seed for reproducibility.
		verbose (bool): Flag for verbosity.
		"""
		self.__adata = adata
		self.__batchLabel = batchLabel
		self.__miniBatchSize = miniBatchSize
		self.__numEpoch = numEpoch
		self.__learningRate = learningRate
		self.__hid1 = hid1
		self.__hid2 = hid2
		self.__latentSize = latentSize
		self.__discHid = discHid
		self.__reconCoef = reconCoef
		self.__klCoef = klCoef
		self.__discCoef = discCoef
		self.__discIter = discIter
		self.__earlyStop = earlyStop
		self.__seed = seed
		self.__verbose = verbose
		self.__lossDict = {"total":[],"recon":[],"trick":[],"klDiv":[],"discr":[]}
		# Set the random seed for reproducibility
		torch.manual_seed(seed)

		# Get training data and labels
		inData, batchSpecLabIndex = self._getTrainLabel(speciesLabel="species")

		# Prepare data by appending labels
		LabeledData, layer1Dim, numSpeices = self._getLabeledData(inData)
		self.__LabeledData = LabeledData
		self.__numSpeices = numSpeices

		# Define layer dimensions
		self.__layerDims = [layer1Dim, hid1, hid2, latentSize]
		self.__inDiscriminatorDims = [latentSize, discHid]

		# Adjust reconstruction coefficient
		self.__reconCoef = self.__reconCoef * (inData.shape[0] ** 0.5)

		# Initialize encoder and decoder
		encoder = Encoder(self.__layerDims, self.__numSpeices)
		decoder = Decoder(self.__layerDims, self.__numSpeices, geneIndexes=batchSpecLabIndex)

		# Initialize autoencoder
		self.__autoencoder = EncoderDecoder(encoder, decoder)
	
		# Initialize discriminator
		self.__discriminator = Discriminator(self.__inDiscriminatorDims, self.__numSpeices)

	def runTrainScVital(self):
		"""
		Train the scVital model, which includes an autoencoder and a discriminator, and store the results.

		This function initializes the encoder, decoder, autoencoder, and discriminator. It sets up the optimizers and learning rate schedulers,
		trains the models, and then stores the trained models and loss information. Finally, it prepares the data for evaluation and stores
		the latent representations and reconstructed data in an AnnData object.

		Attributes:
			self.__layerDims (list): Dimensions of the layers for the encoder and decoder.
			self.__numSpeices (int): Number of species (classes) in the dataset.
			self.__learningRate (float): Learning rate for the optimizers.
			self.__inDiscriminatorDims (list): Dimensions of the layers for the discriminator.
			self.__inLabels (torch.Tensor): Input labels for the data.
			self.__LabeledData (torch.Tensor): Labeled data for training.
			self.__adata (AnnData): AnnData object to store the results.
		"""
		# Initialize autoencoder
		autoencoderOpt = optim.AdamW(params=self.__autoencoder.parameters(), lr=self.__learningRate)
		aeSchedLR = optim.lr_scheduler.CosineAnnealingWarmRestarts(autoencoderOpt, T_0=5, T_mult=2)
		reconstructionLossFunc = torch.nn.MSELoss()

		# Initialize discriminator
		discriminatorOpt = optim.AdamW(params=self.__discriminator.parameters(), lr=self.__learningRate)
		discSchedLR = optim.lr_scheduler.CosineAnnealingWarmRestarts(discriminatorOpt, T_0=5, T_mult=2)
		discriminatorLossFunc = torch.nn.CrossEntropyLoss()

		# Train the model
		self._trainScVital(autoencoderOpt, reconstructionLossFunc, aeSchedLR, discriminatorOpt, discriminatorLossFunc, discSchedLR)

		# Set models to evaluation mode
		self.__autoencoder.eval()
		encoderOut = self.__autoencoder.getEncoder()
		encoderOut.eval()
		self.__discriminator.eval()

		# Prepare one-hot encoded labels
		LabOneHot = torch.reshape(F.one_hot(self.__inLabels.to(torch.int64), num_classes=self.__numSpeices).float(), (self.__LabeledData.shape[0], self.__numSpeices))#inData.shape[0]
		labOneHotInData = torch.cat((self.__LabeledData[:,1:], LabOneHot), axis=1)

		# Get latent representations and reconstructed data
		allEncOut = encoderOut(labOneHotInData)
		bLatent = allEncOut.detach().numpy()
		reconData = self.__autoencoder(labOneHotInData, self.__inLabels, self.__numSpeices).detach().numpy()

		# Store results in AnnData object
		self.__adata.obsm["X_scVital"] = bLatent
		self.__adata.layers["scVitalRecon"] = reconData


	def _trainScVital(self, autoencoderOpt, reconstructionLossFunc, aeSchedLR,
					 discriminatorOpt, discriminatorLossFunc, discSchedLR):
		self.__autoencoder.train()
		self.__discriminator.train()

		ldTrainDataLoader = DataLoader(self.__LabeledData, batch_size=self.__miniBatchSize, shuffle=True)
		numMiniBatch = len(ldTrainDataLoader)

		discEpochLoss = np.full(self.__numEpoch, np.nan)
		reconstEpochLoss = np.full(self.__numEpoch, np.nan)
		trickEpochLoss = np.full(self.__numEpoch, np.nan)
		klDivEpochLoss = np.full(self.__numEpoch, np.nan)
		aeTotalEpochLoss = np.full(self.__numEpoch, np.nan)

		prevTotalLoss = np.inf

		#KL Divergenace Cyclical Annealing
		klDivCoeff = self._klCycle(0, self.__klCoef, self.__numEpoch)

		for iEp in range(self.__numEpoch):

			discTrainLoss = np.full(numMiniBatch, np.nan) 
			reconstTrainLoss = np.full(numMiniBatch, np.nan)
			trickTrainLoss = np.full(numMiniBatch, np.nan)
			klDivTrainLoss = np.full(numMiniBatch, np.nan)
			aeTotalTrainLoss = np.full(numMiniBatch, np.nan)

			for iBatch, ldData in enumerate(ldTrainDataLoader):
				#Load data 
				bRealLabels, bData = ldData[:,0].to(torch.int64), ldData[:,1:].to(torch.float32)
				
				bRealLabOneHot = F.one_hot(bRealLabels, num_classes=self.__numSpeices).float()
				labeledBData = torch.cat((bData, bRealLabOneHot),axis=1)
				labels = np.unique(bRealLabels)
				
				#Train discriminator
				#get mouse and human data in latent space
				encoder = self.__autoencoder.getEncoder()
				#pdb.set_trace()
				for i in range(self.__discIter):
					bLatent = encoder(labeledBData) #bData
					
					#Optimize discriminator
					discriminatorOpt.zero_grad()

					bDiscPred = self.__discriminator(bLatent)
					bDiscRealLoss = discriminatorLossFunc(bDiscPred, bRealLabels)
					bDiscRealLoss.backward()

					discriminatorOpt.step()
					discSchedLR.step(iEp + iBatch / numMiniBatch)

				#Train generator
				autoencoderOpt.zero_grad()
				
				#encode mouse and human data in latent space
				bReconstData = self.__autoencoder(labeledBData, bRealLabels, self.__numSpeices) #bData

				#added
				#split input data and reconstructed data by batch and batch-specific genes
				#calculate reconstruction on on relavent data
				bReconstLoss = torch.tensor(0.0)
				batchSpecLabIndex = self.__autoencoder.getGeneIndex()
				allCells, allGenes = bData.shape

				for i in labels: # for every batch
					sCells = (bRealLabels==i).reshape((allCells,1))	   #Cells realting to label
					#sGenes = torch.tensor(bool(batchSpecLabIndex[i])).reshape((1, allGenes))

					sGenes = torch.tensor(batchSpecLabIndex[i]).reshape((1,allGenes))	 #Genes relating to label
					numCells, numGenes = torch.sum(sCells), torch.sum(sGenes,axis=1)	  #number of cells and genes in of the label
					sMask = sCells * sGenes	   #tensor of the mask to only take the genes realting to labels for the same cells
					bDataMasked = torch.masked_select(bData, sMask).reshape((numCells,numGenes))	  #apply mask to input data
					reconDataMasked = torch.masked_select(bReconstData, sMask).reshape((numCells,numGenes))   #apply mask to reconstructed data
					bReconstLoss += (numCells/allCells)*reconstructionLossFunc(reconDataMasked, bDataMasked)  #calcualte reconstruction with masks and update total
				
				#bReconstLoss = reconstructionLossFunc(bReconstData, bData) 
				
				encoder = self.__autoencoder.getEncoder()
				bLatent = encoder(labeledBData) #bData

				#train discriminator and get preds try and subtract from total Loss
				bDiscPred = self.__discriminator(bLatent)

				#bDiscWrongLoss = discriminatorLossFunc(bDiscPred, bRealLabels)
				bRealLabOneHot = F.one_hot(bRealLabels, num_classes=self.__numSpeices).float()
				reconLatentEven = torch.ones_like(bRealLabOneHot)*(1/self.__numSpeices)
				bDiscTrickLoss = discriminatorLossFunc(bDiscPred, reconLatentEven) #bRealLabels)
				
				#KL Div loss with N(0,1)
				klDivLoss = encoder.klDivLoss

				bRecTrickDiscLoss = self.__reconCoef*bReconstLoss + klDivCoeff[iEp]*klDivLoss + self.__discCoef*bDiscTrickLoss 
				bRecTrickDiscLoss.backward()
				
				#optimize generator
				autoencoderOpt.step()
				aeSchedLR.step(iEp + iBatch / numMiniBatch)

				discTrainLoss[iBatch] 		= bDiscRealLoss.item()
				reconstTrainLoss[iBatch] 	= self.__reconCoef*bReconstLoss.item()
				trickTrainLoss[iBatch]		= self.__discCoef*bDiscTrickLoss.item()
				klDivTrainLoss[iBatch]		= klDivCoeff[iEp]*klDivLoss.item()
				aeTotalTrainLoss[iBatch] 	= bRecTrickDiscLoss.item()

				if (self.__verbose and (iEp % 10 == 0 and iBatch % 8 == 0)):
					print(f"Epoch={iEp}, batch={iBatch}, discr={np.nanmean(discTrainLoss):.4f}, total={np.nanmean(aeTotalTrainLoss):.4f}, recon={np.nanmean(reconstTrainLoss):.4f}, trick={np.nanmean(trickTrainLoss):.4f}, klDiv={np.nanmean(klDivTrainLoss):.4f}")

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
			if (deltaLoss < self.__earlyStop and iEp > 10):
				print(f' epoch:{iEp} delta:{deltaLoss}')
				break
			prevTotalLoss = totalLoss
			
			#aeSchedLR.step()
			#discSchedLR.step()

		self.__lossDict = {"total":aeTotalEpochLoss,
					"recon":reconstEpochLoss,
					"trick":trickEpochLoss,
					"klDiv":klDivEpochLoss,
					"discr":discEpochLoss}
		#return


	def _getLabeledData(self, inData):
		"""
		Concatenate labels and data along the specified axis and return the labeled data,
		the dimension of the first layer, and the number of unique species.

		Parameters:
		inData (torch.Tensor): Input data tensor.
		inLabels (torch.Tensor): Input labels tensor.

		Returns:
		tuple: A tuple containing the labeled data, the dimension of the first layer, and the number of unique species.
		"""
		# Concatenate labels and data along the columns
		LabeledData = torch.cat((self.__inLabels, inData), axis=1)
		
		# Get the dimension of the first layer (number of features in the input data)
		layer1Dim = inData.size()[1]
		
		# Get the number of unique species from the labels
		numSpeices = len(self.__inLabels.unique())
		
		return LabeledData, layer1Dim, numSpeices

	@staticmethod
	def _klCycle(start, stop, n_epoch, n_cycle=4):
		"""
		Generate a KL divergence schedule that cycles between start and stop values over the specified number of epochs.

		Parameters:
		start (float): Starting value of the KL divergence.
		stop (float): Stopping value of the KL divergence.
		n_epoch (int): Total number of epochs.
		n_cycle (int): Number of cycles within the total epochs.

		Returns:
		numpy.ndarray: An array representing the KL divergence schedule.
		"""
		ratio = (n_cycle - 1) / n_cycle
		kl = np.full(n_epoch, stop)
		period = n_epoch / n_cycle
		step = (stop - start) / (period * ratio)  # Linear schedule

		for c in range(n_cycle):
			v, i = start, 0
			while v <= stop and (int(i + c * period) < n_epoch):
				kl[int(i + c * period)] = v
				v += step
				i += 1
		return kl

	def _getAdataX(self):
		"""
		Convert the AnnData object matrix to a dense tensor.

		Parameters:
		adata (AnnData): Annotated data matrix.

		Returns:
		torch.Tensor: Dense tensor representation of the data matrix.
		"""
		adataX = self.__adata.X
		if isspmatrix(adataX):
			adataX = adataX.todense()
		return torch.tensor(adataX)

	def _getTrainLabel(self, speciesLabel="species"):
		"""
		Prepare training data and labels, and generate batch-specific gene indices.

		Parameters:
		speciesLabel (str): Column name for species labels in the AnnData object.

		Returns:
		tuple: A tuple containing the input data, and batch-specific gene indices.
		"""
		# Convert AnnData object matrix to dense tensor
		inData = self._getAdataX()

		# Extract batch labels and create a dictionary mapping unique batches to indices
		batch = np.array(self.__adata.obs[self.__batchLabel])
		uniqueBatch = np.unique(batch)
		numBatch = len(uniqueBatch)
		batchDict = dict(zip(uniqueBatch,range(0,numBatch)))
		batchNum = np.array([batchDict[x] for x in batch])

		# Convert batch numbers to tensor and reshape
		inLabels = torch.tensor(batchNum)
		self.__inLabels = inLabels.view(len(inLabels), 1)

		# Initialize gene type array and batch-species dictionary
		geneType = np.full(len(self.__adata.var_names), "all", dtype=object)
		batchSpeciesDict = {batch: "all" for batch in uniqueBatch}

		# Check if species label exists in the AnnData object
		if np.isin([speciesLabel], self.__adata.obs.columns.values)[0]:
			uniqueSpecies = self.__adata.obs[speciesLabel].cat.categories
			batchSpecies = np.unique(["!".join(x) for x in self.__adata.obs[[self.__batchLabel, speciesLabel]].values])
			batchSpeciesDict = {indBatchSpecies.split("!")[0]: indBatchSpecies.split("!")[1] for indBatchSpecies in batchSpecies}
			
			# Assign gene types based on species
			for i, gene in enumerate(self.__adata.var_names):
				gsplit = gene.split("/")
				if len(gsplit) < 2:
					gInd = np.where(self.__adata.var_names.values == gsplit)[0][0]
					try:
						geneType[i] = self.__adata[(inData[:,gInd]>0).numpy(),:].obs[speciesLabel].unique()[0]
					except:
						geneType[i] = "none"
				if ' ' in gsplit:
					for j, g in enumerate(gsplit):
						if g != " ":
							geneType[i] = uniqueSpecies[j]
			rangeGenes = range(len(geneType))

		# Generate batch-specific gene indices
		batchSpecLabIndex = []
		for batchLab in uniqueBatch:
			speciesSpecGenes = np.logical_or(geneType == "all", geneType == batchSpeciesDict[batchLab])
			batchSpecLabIndex.append(np.array(list(speciesSpecGenes), dtype=bool))

		return inData, batchSpecLabIndex

	def saveDiscrim(self, outDiscFile):
		"""Save the discriminator to file."""
		torch.save(self.__discriminator, outDiscFile)

	def saveAutoenc(self, outVAEFile):
		"""Save the autoencoder to file."""
		torch.save(self.__autoencoderOut, outVAEFile)

	# Getters for the instance variables
	def getAdata(self):
		"""Return the annotated data matrix."""
		return self.__adata

	def getLabeledData(self):
		return self.__LabeledData

	def getBatchLabel(self):
		"""Return the batch label."""
		return self.__batchLabel

	def getMiniBatchSize(self):
		"""Return the mini-batch size."""
		return self.__miniBatchSize

	def getNumEpoch(self):
		"""Return the number of epochs."""
		return self.__numEpoch

	def getLearningRate(self):
		"""Return the learning rate."""
		return self.__learningRate

	def getLayerDims(self):
		"""Return a list of the dimentions of the hidden layers of scVital."""
		return [len(self.__adata.var_names), self.__hid1, self.__hid2, self.__latentSize] 

	def getDiscDims(self):
		"""Return a list of the dimentions of the hidden layers of discriminator scVital."""
		return [self.__latentSize, self.__discHid, self.__numSpeices] 

	def getLatentSize(self):
		"""Return the size of the latent space."""
		return self.__latentSize

	def getReconCoef(self):
		"""Return the coefficient for reconstruction loss."""
		return self.__reconCoef

	def getKlCoef(self):
		"""Return the coefficient for KL divergence loss."""
		return self.__klCoef

	def getDiscCoef(self):
		"""Return the coefficient for discriminator loss."""
		return self.__discCoef

	def getDiscIter(self):
		"""Return the number of iterations for discriminator training."""
		return self.__discIter

	def getEarlyStop(self):
		"""Return the delta error to trigger early stopping."""
		return self.__earlyStop

	def getAutoencoder(self):
		"""Return the Autoencoder."""
		return self.__autoencoder

	def getDisriminator(self):
		"""Return the Discriminator."""
		return self.__discriminator

	def getLossDict(self):
		"""Return the loss dicitonary of training."""
		return self.__lossDict

	def __str__(self):
		return (f"Model Parameters:\n"
			f"adata: {self.__adata}\n"
			f"batchLabel: {self.__batchLabel}\n"
			f"miniBatchSize: {self.__miniBatchSize}\n"
			f"numEpoch: {self.__numEpoch}\n"
			f"learningRate: {self.__learningRate}\n"
			f"reconCoef: {self.__reconCoef}\n"
			f"klCoef: {self.__klCoef}\n"
			f"discCoef: {self.__discCoef}\n"
			f"discIter: {self.__discIter}\n"
			f"earlyStop: {self.__earlyStop}\n"
			f"seed: {self.__seed}\n"
			f"verbose: {self.__verbose}\n"
			f"numSpeices: {self.__numSpeices}\n"
			f"layerDims: {self.__layerDims}\n"
			f"inDiscriminatorDims: {self.__inDiscriminatorDims}")


	def saveModel(self, filename):
		"""
		Save the scVitalModel object to a file.

		Parameters:
		model (scVitalModel): The model object to save.
		filename (str): The name of the file to save the model to.
		"""
		if(filename[-4:] != ".pkl"):
			filename = filename + ".pkl"
		with open(filename, 'wb') as file:
			pickle.dump(self, file)
		print(f"Model saved to {filename}")


	def plotLoss(self, save=False):
		lossFig, lossAxs = plt.subplots(3, 2)
		lossAxs[0, 0].plot(self.__lossDict["recon"]),
		lossAxs[0, 0].set_title('Recon Loss')

		lossAxs[0, 1].plot(self.__lossDict["discr"])
		lossAxs[0, 1].set_title('Disc Loss')

		lossAxs[1, 1].plot(self.__lossDict["trick"])
		lossAxs[1, 1].set_title('Trick Disc Loss')

		lossAxs[1, 0].plot(self.__lossDict["klDiv"])
		lossAxs[1, 0].set_title('KL Div Loss')

		lossAxs[2, 0].plot(self.__lossDict["total"])
		lossAxs[2, 0].set_title('Total Loss')

		lossFig.tight_layout(pad=1.0)
		if(save):
			lossFig.savefig(f"{save}/lossPlots.png")


def loadModel(filename):
	"""
	Load the scVitalModel object from a file.

	Parameters:
	filename (str): The name of the file to load the model from.

	Returns:
	scVitalModel: The loaded model object.
	"""
	with open(filename, 'rb') as file:
		scvitalModel = pickle.load(file)
	print(f"Model loaded from {filename}")
	return scvitalModel


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encIn, labels, num_classes): 
        latent = self.encoder(encIn)
        #labelsOneHot = F.one_hot(labels, num_classes=num_classes).float()
        labelsOneHot = torch.reshape(F.one_hot(labels.to(torch.int64), num_classes=num_classes).float(),(latent.shape[0], num_classes))
        encOutLabel = torch.cat((latent, labelsOneHot),axis=1)
        decOut = self.decoder(encOutLabel)
        return decOut
    
    def getEncoder(self):
        return self.encoder

    def getDecoder(self):
        return self.decoder
        
    def getGeneIndex(self):
        return self.decoder.getGeneIndexes()

class Encoder(nn.Module):
    def __init__(self, dims, numSpecies):#
        super(Encoder, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(dims[0] + numSpecies, dims[1], bias=True), # 
            nn.LayerNorm(dims[1]),
            #nn.BatchNorm1d(dims[1]),
            #nn.Dropout(0.05),
            nn.ReLU() #nn.LeakyReLU(0.1)
        )
        self.hidden1 = nn.Sequential(         
            nn.Linear(dims[1], dims[2], bias=True),
            nn.LayerNorm(dims[2]),
            #nn.BatchNorm1d(dims[2]),
            #nn.Dropout(0.05),
            nn.ReLU() #nn.LeakyReLU(0.1)
        )
        self.mu = nn.Sequential(
            nn.Linear(dims[2], dims[3], bias=True)
        )
        self.lnVar = nn.Sequential(
            nn.Linear(dims[2], dims[3], bias=True)
        )
        
        nn.init.kaiming_normal_(self.hidden0[0].weight,nonlinearity='relu')#,a=0.1,nonlinearity='leaky_relu')#
        nn.init.kaiming_normal_(self.hidden1[0].weight,nonlinearity='relu')#,a=0.1,nonlinearity='leaky_relu')#
        self.klDivLoss = 0

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        mean = self.mu(x)
        lnVar = self.lnVar(x)
        x = torch.exp(0.5*lnVar) * torch.randn_like(lnVar) + mean
        self.klDivLoss = torch.mean(0.5 * torch.sum(mean**2 + torch.exp(lnVar) - lnVar - 1,dim=1),dim=0)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, dims, numSpecies, **kwargs):
        self.geneIndexes = kwargs.get("geneIndexes",None)
        self.numSpecies = numSpecies
        super(Decoder, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(dims[3] + numSpecies, dims[2], bias=True),
            #nn.BatchNorm1d(dims[2]),
            nn.ReLU() #nn.LeakyReLU(0.1)
        )
        self.hidden1 = nn.Sequential(         
            nn.Linear(dims[2], dims[1], bias=True),
            #nn.BatchNorm1d(dims[1]),
            nn.ReLU() #nn.LeakyReLU(0.1)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(dims[1], dims[0], bias=True)#,dims[0] * 2
            #nn.LeakyReLU(0.1)
        )

        nn.init.kaiming_normal_(self.hidden0[0].weight,nonlinearity='relu')#,a=0.1,nonlinearity='leaky_relu')#
        nn.init.kaiming_normal_(self.hidden1[0].weight,nonlinearity='relu')#,a=0.1,nonlinearity='leaky_relu')#
        #nn.init.kaiming_normal_(self.hidden2[0].weight,nonlinearity='relu')#,a=0.1,nonlinearity='leaky_relu')#

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return x

    def getGeneIndexes(self):
        return self.geneIndexes
    
class Discriminator(nn.Module):
    def __init__(self, dims, numSpecies):
        super(Discriminator, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(dims[0], dims[1], bias=True),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(dims[1], numSpecies, bias=True)
        )
        #self.out = nn.Sequential(
        #    nn.Identity(numSpecies)
        #)
        
        nn.init.kaiming_normal_(self.hidden0[0].weight,nonlinearity='relu')#,a=0.1,nonlinearity='leaky_relu')#

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        #x = self.out(x)
        return x
        
