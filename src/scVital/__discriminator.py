from torch import nn
    
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
        