import sys
import os
import random
import time
import scanpy as sc
import pandas as pd
import numpy as np
import pdb

import FuncVizBench as vb
import FuncAUC as auc

outDir = snakemake.params["outDirect"]
outARIfile = snakemake.output["outARIfile"]
outFMfile = snakemake.output["outFMfile"]
outScaleFile = snakemake.output["outScaleFile"]

dataDirs = np.array(os.listdir(outDir))
#print(dataDirs)
datas = []#[dataDir.split("~")[1] for dataDir in os.listdir(outDir)]
for dataDir in os.listdir(outDir):
    if("~" in dataDir):
        datas = datas+[dataDir.split("~")[1]]
        
clutNameSet = set()
#print(datas)
for dataDir in os.listdir(outDir): #for every dataset
    if("~" in dataDir):
        for infoDir in os.listdir(os.path.join(outDir,dataDir)):#for every out file
            if(os.path.isdir(os.path.join(outDir,dataDir,infoDir)) and (not infoDir=="figures")):
                for file in os.listdir(os.path.join(outDir,dataDir,infoDir,"clusters")):
                    if(".csv" in file):
                        name = file.split(".")[0]
                        clutNameSet.add(f"{name}")
                break

clutNameList= list(clutNameSet)

fmDf = pd.DataFrame(np.zeros((len(datas),len(clutNameList))),
                 index=datas,
                 columns=clutNameList)
ariDf = pd.DataFrame(np.zeros((len(datas),len(clutNameList))),
                 index=datas,
                 columns=clutNameList)
scaleDf = pd.DataFrame(np.zeros((len(datas),1+len(clutNameList))),
                 index=datas,
                 columns=["Size"]+clutNameList)

conDict = {"FM":fmDf,
           "ARI":ariDf,
           "scale":scaleDf}

for dataDir in os.listdir(outDir): #for every dataset
    if("~" in dataDir):
        dataName = dataDir.split("~")[1]
        for infoDir in os.listdir(os.path.join(outDir,dataDir)):#for every out file
            if(os.path.isdir(os.path.join(outDir,dataDir,infoDir)) and (not infoDir=="figures")):
                metricDir = os.path.join(outDir,dataDir,infoDir,"figures","")
                figsFiles = np.array(os.listdir(metricDir))
                metricsFilesBool = np.array([".csv" in figsFile for figsFile in figsFiles])
                metricsFiles = figsFiles[metricsFilesBool]
                for metFile in metricsFiles:
                    fullMetFile = metricDir + metFile
                    infoDF = pd.read_csv(fullMetFile, index_col=0)
                    if("scale" in fullMetFile):
                        conDict["scale"].loc[dataName,["Size", infoDF.columns[1]]] = infoDF.iloc[0,:]
                    elif("metrics" in fullMetFile):
                        integ = infoDF.columns[0]
                        for metric in infoDF.index:
                            conDict[metric].loc[dataName,integ] = infoDF.loc[metric,integ]

print(conDict)

vb.plotMetricBarByData(conDict["ARI"], label="ARI", outDir=outDir)
vb.plotMetricBarByData(conDict["FM"], label="FM", outDir=outDir)

vb.plotMetricBarByInteg(conDict["ARI"], label="ARI", outDir=outDir)
vb.plotMetricBarByInteg(conDict["FM"], label="FM", outDir=outDir)

vb.plotScale(conDict["scale"], outDir=outDir)

print("fin")
