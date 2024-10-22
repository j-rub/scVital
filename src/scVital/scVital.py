#!/usr/bin/env python
#run train DL model
import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as an

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import AutoEncoder as ae
import Discriminator as dis 


def makeScVital(
	adata: an.AnnData,
	batchLabel: str,
	miniBatchSize: int = 512,
	numEpoch: int = 64,
	learningRate: float = 3e-1,
	hid1: int = 1024,
	hid2: int = 128,
	latentSize: int = 12,
	discHid: int = 6,
	reconCoef: float = 1,
	klCoef: float = 1e-1,
	discCoef: float = 1,
	discIter: int = 5,
	earlyStop: float = 1e-2,
	train: bool=False,
	seed: int = 18,
	verbose: bool = True
) -> scVital:
	"""
	Run the scVital model with the specified parameters.

	Parameters:
	adata (an.AnnData): Annotated data matrix.
	batchLabel (str): Label for batch processing.
	miniBatchSize (int, optional): Size of mini-batches for training. Default is 512.
	numEpoch (int, optional): Number of epochs for training. Default is 64.
	learningRate (float, optional): Learning rate for the optimizer. Default is 3e-1.
	hid1 (int, optional): Number of units in the first hidden layer. Default is 1024.
	hid2 (int, optional): Number of units in the second hidden layer. Default is 128.
	latentSize (int, optional): Size of the latent space. Default is 12.
	discHid (int, optional): Number of units in the discriminator hidden layer. Default is 6.
	reconCoef (float, optional): Coefficient for reconstruction loss. Default is 1.
	klCoef (float, optional): Coefficient for KL divergence loss. Default is 1e-1.
	discCoef (float, optional): Coefficient for discriminator loss. Default is 1.
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
	if adata.shape[0] > 4000:
		warnings.warn("The adata object has many genes consider subsetting on highly variable genes", UserWarning)

	# Initialize the scVital model with the provided parameters
	scVitalData = scVital(
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
		self.adata = adata
		self.batchLabel = batchLabel
		self.miniBatchSize = miniBatchSize
		self.numEpoch = numEpoch
		self.learningRate = learningRate
		self.hid1 = hid1
		self.hid2 = hid2
		self.latentSize = latentSize
		self.discHid = discHid
		self.reconCoef = reconCoef
		self.klCoef = klCoef
		self.discCoef = discCoef
		self.discIter = discIter
		self.earlyStop = earlyStop
		self.seed = seed
		self.verbose = verbose

		# Set the random seed for reproducibility
		torch.manual_seed(seed)

		# Get training data and labels
		inData, inLabels, batchSpecLabIndex = self._getTrainLabel(speciesLabel="species")

		self.inLabels = inLabels
		self.batchSpecLabIndex = batchSpecLabIndex

		# Prepare data by appending labels
		LabeledData, layer1Dim, numSpeices = self._getLabeledData(inData, inLabels)
		self.LabeledData = LabeledData
		self.numSpeices = numSpeices

		# Define layer dimensions
		self.layerDims = [layer1Dim, hid1, hid2, latentSize]
		self.inDiscriminatorDims = [latentSize, discHid]

		# Adjust reconstruction coefficient
		self.reconCoef = self.reconCoef * (inData.shape[0] ** 0.5)


	def runTrainScVital(self):
		"""
		Train the scVital model, which includes an autoencoder and a discriminator, and store the results.

		This function initializes the encoder, decoder, autoencoder, and discriminator. It sets up the optimizers and learning rate schedulers,
		trains the models, and then stores the trained models and loss information. Finally, it prepares the data for evaluation and stores
		the latent representations and reconstructed data in an AnnData object.

		Attributes:
			self.layerDims (list): Dimensions of the layers for the encoder and decoder.
			self.numSpeices (int): Number of species (classes) in the dataset.
			self.batchSpecLabIndex (list): Indexes for batch-specific labels.
			self.learningRate (float): Learning rate for the optimizers.
			self.inDiscriminatorDims (list): Dimensions of the layers for the discriminator.
			self.inLabels (torch.Tensor): Input labels for the data.
			self.LabeledData (torch.Tensor): Labeled data for training.
			self.adata (AnnData): AnnData object to store the results.
		"""
		# Initialize encoder and decoder
		encoder = ae.Encoder(self.layerDims, self.numSpeices)
		decoder = ae.Decoder(self.layerDims, self.numSpeices, geneIndexes=self.batchSpecLabIndex)

		# Initialize autoencoder
		autoencoder = ae.EncoderDecoder(encoder, decoder)
		autoencoderOpt = optim.AdamW(params=autoencoder.parameters(), lr=self.learningRate)
		aeSchedLR = optim.lr_scheduler.CosineAnnealingWarmRestarts(autoencoderOpt, T_0=5, T_mult=2)
		reconstructionLossFunc = torch.nn.MSELoss()

		# Initialize discriminator
		discriminator = dis.Discriminator(self.inDiscriminatorDims, self.numSpeices)
		discriminatorOpt = optim.AdamW(params=discriminator.parameters(), lr=self.learningRate)
		discSchedLR = optim.lr_scheduler.CosineAnnealingWarmRestarts(discriminatorOpt, T_0=5, T_mult=2)
		discriminatorLossFunc = torch.nn.CrossEntropyLoss()

		# Train the model
		autoencoderOut, discriminatorOut, lossDict = self._trainScVital(
			autoencoder, autoencoderOpt, reconstructionLossFunc, aeSchedLR,
			discriminator, discriminatorOpt, discriminatorLossFunc, discSchedLR
		)

		self.autoencoder = autoencoderOut
		self.discriminator = discriminatorOut
		self.lossDict = lossDict

		# Set models to evaluation mode
		autoencoderOut.eval()
		encoderOut = autoencoderOut.getEncoder()
		encoderOut.eval()
		discriminator.eval()

		# Prepare one-hot encoded labels
		LabOneHot = torch.reshape(F.one_hot(self.inLabels.to(torch.int64), num_classes=self.numSpeices).float(), (self.LabeledData.shape[0], self.numSpeices))#inData.shape[0]
		labOneHotInData = torch.cat((self.LabeledData, LabOneHot), axis=1)

		# Get latent representations and reconstructed data
		allEncOut = encoderOut(labOneHotInData)
		bLatent = allEncOut.detach().numpy()
		reconData = autoencoderOut(labOneHotInData, self.inLabels, self.numSpeices).detach().numpy()

		# Store results in AnnData object
		self.adata.obsm["X_scVital"] = bLatent
		self.adata.layers["scVitalRecon"] = reconData


	def _trainScVital(self, autoencoder, autoencoderOpt, reconstructionLossFunc, aeSchedLR,
					 discriminator, discriminatorOpt, discriminatorLossFunc, discSchedLR):
		autoencoder.train()
		discriminator.train()

		ldTrainDataLoader = DataLoader(self.LabeledData, batch_size=self.miniBatchSize, shuffle=True)
		numMiniBatch = len(ldTrainDataLoader)

		discEpochLoss = np.full(self.numEpoch, np.nan)
		reconstEpochLoss = np.full(self.numEpoch, np.nan)
		trickEpochLoss = np.full(self.numEpoch, np.nan)
		klDivEpochLoss = np.full(self.numEpoch, np.nan)
		aeTotalEpochLoss = np.full(self.numEpoch, np.nan)

		prevTotalLoss = np.inf

		#KL Divergenace Cyclical Annealing
		klDivCoeff = self._klCycle(0, self.klCoef, self.numEpoch)

		for iEp in range(self.numEpoch):

			discTrainLoss = np.full(numMiniBatch, np.nan) 
			reconstTrainLoss = np.full(numMiniBatch, np.nan)
			trickTrainLoss = np.full(numMiniBatch, np.nan)
			klDivTrainLoss = np.full(numMiniBatch, np.nan)
			aeTotalTrainLoss = np.full(numMiniBatch, np.nan)

			for iBatch, ldData in enumerate(ldTrainDataLoader):
				#Load data 
				bRealLabels, bData = ldData[:,0].to(torch.int64), ldData[:,1:].to(torch.float32)
				
				bRealLabOneHot = F.one_hot(bRealLabels, num_classes=self.numSpeices).float()
				labeledBData = torch.cat((bData, bRealLabOneHot),axis=1)
				labels = np.unique(bRealLabels)
				
				#Train discriminator
				#get mouse and human data in latent space
				encoder = autoencoder.getEncoder()
				#pdb.set_trace()
				for i in range(self.discIter):
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
				bReconstData = autoencoder(labeledBData, bRealLabels, self.numSpeices) #bData

				#added
				#split input data and reconstructed data by batch and batch-specific genes
				#calculate reconstruction on on relavent data
				bReconstLoss = torch.tensor(0.0)
				batchSpecLabIndex = autoencoder.getGeneIndex()
				allCells, allGenes = bData.shape

				for i in labels: # for every batch
					sCells = (bRealLabels==i).reshape((allCells,1))	   #Cells realting to label
					sGenes = torch.tensor(batchSpecLabIndex[i]).reshape((1,allGenes))	 #Genes relating to label
					numCells, numGenes = torch.sum(sCells), torch.sum(sGenes,axis=1)	  #number of cells and genes in of the label
					sMask = sCells * sGenes	   #tensor of the mask to only take the genes realting to labels for the same cells
					bDataMasked = torch.masked_select(bData, sMask).reshape((numCells,numGenes))	  #apply mask to input data
					reconDataMasked = torch.masked_select(bReconstData, sMask).reshape((numCells,numGenes))   #apply mask to reconstructed data
					bReconstLoss += (numCells/allCells)*reconstructionLossFunc(reconDataMasked, bDataMasked)  #calcualte reconstruction with masks and update total
				
				#bReconstLoss = reconstructionLossFunc(bReconstData, bData) 
				
				encoder = autoencoder.getEncoder()
				bLatent = encoder(labeledBData) #bData

				#train discriminator and get preds try and subtract from total Loss
				bDiscPred = discriminator(bLatent)

				#bDiscWrongLoss = discriminatorLossFunc(bDiscPred, bRealLabels)
				bRealLabOneHot = F.one_hot(bRealLabels, num_classes=self.numSpeices).float()
				reconLatentEven = torch.ones_like(bRealLabOneHot)*(1/self.numSpeices)
				bDiscTrickLoss = discriminatorLossFunc(bDiscPred, reconLatentEven) #bRealLabels)
				
				#KL Div loss with N(0,1)
				klDivLoss = encoder.klDivLoss

				bRecTrickDiscLoss = self.reconCoef*bReconstLoss + klDivCoeff[iEp]*klDivLoss + self.discCoef*bDiscTrickLoss 
				bRecTrickDiscLoss.backward()
				
				#optimize generator
				autoencoderOpt.step()
				aeSchedLR.step(iEp + iBatch / numMiniBatch)

				discTrainLoss[iBatch] 		= bDiscRealLoss.item()
				reconstTrainLoss[iBatch] 	= self.reconCoef*bReconstLoss.item()
				trickTrainLoss[iBatch]		= self.discCoef*bDiscTrickLoss.item()
				klDivTrainLoss[iBatch]		= klDivCoeff[iEp]*klDivLoss.item()
				aeTotalTrainLoss[iBatch] 	= bRecTrickDiscLoss.item()

				if (self.verbose and (iEp % 50 == 0 and iBatch % 20 == 0)):
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
			if (deltaLoss < self.earlyStop and iEp > 10):
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


	def _getLabeledData(self, inData, inLabels):
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
		LabeledData = torch.cat((inLabels, inData), axis=1)
		
		# Get the dimension of the first layer (number of features in the input data)
		layer1Dim = inData.size()[1]
		
		# Get the number of unique species from the labels
		numSpeices = len(inLabels.unique())
		
		return LabeledData, layer1Dim, numSpeices

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

	def _getAdataX(adata):
		"""
		Convert the AnnData object matrix to a dense tensor.

		Parameters:
		adata (AnnData): Annotated data matrix.

		Returns:
		torch.Tensor: Dense tensor representation of the data matrix.
		"""
		adataX = adata.X
		if isspmatrix(adataX):
			adataX = adataX.todense()
		return torch.tensor(adataX)

	def _getTrainLabel(self, speciesLabel="species"):
		"""
		Prepare training data and labels, and generate batch-specific gene indices.

		Parameters:
		speciesLabel (str): Column name for species labels in the AnnData object.

		Returns:
		tuple: A tuple containing the input data, labels, and batch-specific gene indices.
		"""
		# Convert AnnData object matrix to dense tensor
		inData = _getAdataX(self.adata)

		# Extract batch labels and create a dictionary mapping unique batches to indices
		batch = np.array(self.adata.obs[self.batchLabel])
		uniqueBatch = np.unique(batch)
		numBatch = len(uniqueBatch)
		batchDict = dict(zip(uniqueBatch,range(0,numBatch)))
		batchNum = np.array([batchDict[x] for x in batch])

		# Convert batch numbers to tensor and reshape
		inLabels = torch.tensor(batchNum)
		inLabels = inLabels.view(len(inLabels), 1)

		# Initialize gene type array and batch-species dictionary
		geneType = np.full(len(self.adata.var_names), "all", dtype=object)
		batchSpeciesDict = {batch: "all" for batch in uniqueBatch}

		# Check if species label exists in the AnnData object
		if np.isin([speciesLabel], self.adata.obs.columns.values)[0]:
			uniqueSpecies = self.adata.obs[speciesLabel].cat.categories
			batchSpecies = np.unique(["!".join(x) for x in self.adata.obs[[self.batchLabel, speciesLabel]].values])
			batchSpeciesDict = {indBatchSpecies.split("!")[0]: indBatchSpecies.split("!")[1] for indBatchSpecies in batchSpecies}
			
			# Assign gene types based on species
			for i, gene in enumerate(self.adata.var_names):
				gsplit = gene.split("/")
				if ' ' in gsplit:
					for j, g in enumerate(gsplit):
						if g != " ":
							geneType[i] = uniqueSpecies[j]
			rangeGenes = range(len(geneType))

		# Generate batch-specific gene indices
		batchSpecLabIndex = []
		for batchLab in uniqueBatch:
			speciesSpecGenes = np.logical_or(geneType == "all", geneType == batchSpeciesDict[batchLab])
			batchSpecLabIndex.append(list(speciesSpecGenes))

		return inData, inLabels, batchSpecLabIndex



	def saveDiscrim(self, outDiscFile):
		"""Save the discriminator to file."""
		torch.save(self.discriminator, outDiscFile)

	def saveAutoenc(self, outVAEFile):
	"""Save the autoencoder to file."""
		torch.save(self.autoencoderOut, outVAEFile)

	# Getters for the instance variables
	def get_adata(self):
	"""Return the annotated data matrix."""
		return self.adata

	def get_batchLabel(self):
	"""Return the batch label."""
		return self.batchLabel

	def get_miniBatchSize(self):
	"""Return the mini-batch size."""
		return self.miniBatchSize

	def get_numEpoch(self):
	"""Return the number of epochs."""
		return self.numEpoch

	def get_learningRate(self):
	"""Return the learning rate."""
		return self.learningRate

	def get_latent(self):
	"""Return the size of the latent space."""
		return self.latentSize

	def get_reconCoef(self):
	"""Return the coefficient for reconstruction loss."""
		return self.reconCoef

	def get_klCoef(self):
	"""Return the coefficient for KL divergence loss."""
		return self.klCoef

	def get_discCoef(self):
	"""Return the coefficient for discriminator loss."""
		return self.discCoef

	def get_discIter(self):
	"""Return the number of iterations for discriminator training."""
		return self.discIter

	def get_earlyStop(self):
	"""Return the delta error to trigger early stopping."""
		return self.earlyStop












