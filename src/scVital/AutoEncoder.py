import torch
from torch import nn
import torch.nn.functional as F

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

