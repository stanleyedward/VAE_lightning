import torch
import torch.nn.functional as F 
from torch import nn 
import pytorch_lightning as pl

#VAE architecture
class VAE(pl.LightningModule):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == '__main__':
    vae = VAE()
    #print(vae(torch.rand(4,784)))
    trainer = pl.Trainer(fast_dev_run=True)# runs a single batch through training and testing loop to check for errors, basically compiliing your code 
    trainer.fit(vae) #will give errors since we havent defined any training step




