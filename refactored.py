from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import torch
import torch.nn.functional as F 
from torch import nn 
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
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

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def forward(self, z):
        # mu, logvar = self.encode(x.view(-1, 784))
        # z = self.reparameterize(mu, logvar)
        # return self.decode(z), mu, logvar
        return self.decode(z)

    def training_step(self, batch, batch_idx):
        #data = data.to(device) #lightning manages devices directly 
        #optimizer.zero_grad() #lightning updates optimizers directly 
        x, _ = batch #will error if you dont split
        # we move the forward() here in the training step because its actually whats ahpepening in the training step
        mu, logvar = self.encode(x.view(-1, 784)) 
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        # return self.decode(z), mu, logvar
        # recon_batch, mu, logvar = self(x)
        loss = self.loss_function(x_hat, x, mu, logvar)
        #loss.backward() #lightning automates the backward prop as well
        #train_loss += loss.item() #lightning also aggregates the loss automatically like this
        #optimizer.step() #lightning updates optimizers directly 
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        x, _ = batch
        # data = data.to(device)
        
        # recon_batch, mu, logvar = self(x)
        # val_loss = self.loss_function(recon_batch, x, mu, logvar).item()
        
        #from the updated train_step
        mu, logvar = self.encode(x.view(-1, 784)) 
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        val_loss = self.loss_function(x_hat, x, mu, logvar)
            
        # if batch_idx == 0:
        #     n = min(x.size(0), 8)
        #     comparison = torch.cat([x[:n],
        #                             x_hat.view(args.batch_size, 1, 28, 28)[:n]])
        #     path = 'results/reconstruction_' + str(self.current_epoch) + '.png'
        #     save_image(comparison.cpu(), path, nrow=n)
        
        #deleted since we are using logger below!

        return {'val_loss': val_loss, 
                'x_hat': x_hat}
    
    def on_validation_epoch_end(self, outputs): #for logging the end of every epoch and not only batch
        
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        x_hat = outputs[-1]['x_hat']
   
        grid = torchvision.utils.make_grid(x_hat)
        self.logger.experiment.add_image('images', grid, 0)
        
        log = {'avg_val_loss': val_loss}
        return {'log': log}

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def train_dataloader(self):
        
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
        # batch_size=args.batch_size, shuffle=True, **kwargs)
        batch_size=args.batch_size, shuffle=True) #removed kwargs since we dont need it anymore after putting train_dataloader inside the model class
        return train_loader
    
    def val_dataloader(self):

        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            # batch_size=args.batch_size, shuffle=False, **kwargs)
            batch_size=args.batch_size, shuffle=False)#removed kwargs since we dont need it anymore after putting val_dataloader inside the model class
        return val_loader
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--cuda', default=False)
    args = parser.parse_args()
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    

    
    
    vae = VAE()
    #print(vae(torch.rand(4,784)))
    trainer = pl.Trainer(fast_dev_run=True)# runs a single batch through training and testing loop to check for errors, basically compiliing your code 
    # trainer = pl.Trainer()
    # trainer = pl.Trainer(limit_train_batches=0.1)#if we wanna train on 10% data without seeing how the reconstruction looks like 
    # trainer.fit(vae, train_dataloaders=train_loader, val_dataloaders=val_loader) #will give errors if we havent defined any training step and dataloader
    trainer.fit(vae) #since we added the train_dataloader and test_dataloader function inside the model we dont need to specify




