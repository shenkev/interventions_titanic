#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
sns.set()

# In[9]:


titanic_training = pd.read_csv('train.csv')
titanic_training.head()

# In[1]:


from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="./logs/myrun")

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
pyro.enable_validation(True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# In[2]:


def check_dim(x):
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    return x

class DecoderC(nn.Module):
    def __init__(self, U_c_dim, num_classes):
        super(DecoderC, self).__init__()
        self.fc1 = nn.Linear(U_c_dim + 2, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, U_c, A, S):

        A, S = [check_dim(x) for x in [A, S]]
        
        logits = self.fc1(torch.cat([U_c, A, S], dim=1))
        probs = self.softmax(logits)
        return probs
    
class DecoderY(nn.Module):
    def __init__(self, U_y_dim, num_classes):
        super(DecoderY, self).__init__()
        self.fc1 = nn.Linear(U_y_dim + 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, U_y, A, S, C):

        A, S, C = [check_dim(x) for x in [A, S, C]]
        
        logits = self.fc1(torch.cat([U_y, A, S, C], dim=1))
        probs = self.sigmoid(logits)
        return probs

class EncoderC(nn.Module):
    def __init__(self, U_c_dim, num_classes, hid_dim=5):
        super(EncoderC, self).__init__()
        self.m1 = nn.Linear(3, U_c_dim)
        self.v1 = nn.Linear(3, hid_dim)
        self.v2 = nn.Linear(hid_dim, U_c_dim)
        self.v2.bias.data.fill_(1.0)

    def forward(self, CAS):

        mean_c = self.m1(CAS)
        std_c = torch.exp(self.v2(F.tanh(self.v1(CAS))))
        return mean_c, std_c
    
class EncoderY(nn.Module):
    def __init__(self, U_y_dim, num_classes, hid_dim=5):
        super(EncoderY, self).__init__()
        self.m1 = nn.Linear(4, U_y_dim)
        self.v1 = nn.Linear(4, hid_dim)
        self.v2 = nn.Linear(hid_dim, U_y_dim)
        self.v2.bias.data.fill_(1.0)

    def forward(self, YCAS):

        mean_y = self.m1(YCAS)
        # I found variance had to be nonlinear for it not to collapse to 0
        std_y = torch.exp(self.v2(F.tanh(self.v1(YCAS))))
        return mean_y, std_y
    
def to_one_hot(idxs, num_classes):
    
    idxs = idxs.reshape(-1, 1)
    return (idxs == torch.arange(num_classes).reshape(1, num_classes).to(device)).float()

def one_hot_to_idx(one_hot):
    return one_hot.nonzero()[:, -1].float()

# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)

# In[3]:


class LinearVAE(nn.Module):

    def __init__(self, U_c_dim=1, U_y_dim=1, num_classes=3):
        super(LinearVAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder_c = EncoderC(U_c_dim, num_classes)
        self.encoder_y = EncoderY(U_y_dim, num_classes)
        self.decoder_c = DecoderC(U_c_dim, num_classes)
        self.decoder_y = DecoderY(U_y_dim, num_classes)

        # self.encoder_c.apply(init_weights)
        # self.encoder_y.apply(init_weights)
        # self.decoder_c.apply(init_weights)
        # self.decoder_y.apply(init_weights)

        self.U_c_dim = U_c_dim
        self.U_y_dim = U_y_dim
        self.num_classes = num_classes

    # define the model p(x|z)p(z)
    def model(self, x):
        pyro.module("decoder_c", self.decoder_c)
        pyro.module("decoder_y", self.decoder_y)

        with pyro.plate("data", x.shape[0]):
            # x is (Outcome, Class, Age, Sex)
            
            # prior on U_c
            mean_c = x.new_zeros(torch.Size((x.shape[0], self.U_c_dim)))
            std_c = x.new_ones(torch.Size((x.shape[0], self.U_c_dim)))
            U_c = pyro.sample("U_c", dist.Normal(mean_c, std_c).to_event(1))
            
            # prior on U_y
            mean_y = x.new_zeros(torch.Size((x.shape[0], self.U_y_dim)))
            std_y = x.new_ones(torch.Size((x.shape[0], self.U_y_dim)))
            U_y = pyro.sample("U_y", dist.Normal(mean_y, std_y).to_event(1))
            
            # prior on Age
            mean_a = 29.7*x.new_ones(torch.Size((x.shape[0], 1)))
            std_a = 14.5*x.new_ones(torch.Size((x.shape[0], 1)))
            A = pyro.sample("Age", dist.Normal(mean_a, std_a).to_event(1))
            
            # prior on Sex
            prob_s = 0.6476*x.new_ones(torch.Size((x.shape[0], 1)))
            S = pyro.sample("Sex", dist.Bernoulli(prob_s).to_event(1))
            
            # decode the latent code z
            C_probs = self.decoder_c(U_c, A, S)
            C = pyro.sample("Class", dist.Multinomial(probs=C_probs).to_event(1), obs=to_one_hot(x[:, 1], self.num_classes))
            C = one_hot_to_idx(C)
            
            # score against actual outcome
            Y = self.decoder_y(U_y, A, S, C)
            pyro.sample("Outcome", dist.Bernoulli(probs=Y).to_event(1), obs=x[:, 0].reshape(-1, 1))
            
    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        pyro.module("encoder_c", self.encoder_c)
        pyro.module("encoder_y", self.encoder_y)
                        
        with pyro.plate("data", x.shape[0]):
            # x is (Outcome, Class, Age, Sex)

            # posterior on U_c            
            mean_c, std_c = self.encoder_c(x[:, 1:])
            pyro.sample("U_c", dist.Normal(mean_c, std_c).to_event(1))
                        
            # posterior on U_y           
            mean_y, std_y = self.encoder_y(x)
            pyro.sample("U_y", dist.Normal(mean_y, std_y).to_event(1))


# In[4]:


class TitanicDataset(Dataset):

    def __init__(self, data):

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# In[14]:


def train(train_loader, svi):

    epoch_loss = 0

    for batch in tqdm(train_loader):
        batch = batch.float().to(device)
        epoch_loss += svi.step(batch)

    return epoch_loss/len(train_loader.dataset)

def evaluate(test_loader, svi, vae, epoch):

    test_loss = 0.

    for i, batch in enumerate(tqdm(test_loader)):
        batch = batch.float().to(device)
        test_loss += svi.evaluate_loss(batch)

        if i == 0:
            mean_c, std_c = vae.encoder_c(batch[:, 1:])
            mean_y, std_y = vae.encoder_y(batch)
            C_probs = vae.decoder_c(mean_c, batch[:, 2:3], batch[:, 3:4])
            Y = vae.decoder_y(mean_y, batch[:, 2:3], batch[:, 3:4], C_probs.argmax(dim=1).unsqueeze(1).float())
 
            for j in range(5):
                print(" Y, P, A, S: {}\n Uc, Uy: {} var {}, {} var {} \n Class Probs: {}\n Y Pred: {}".format(
                    batch[j], mean_c[j].item(), std_c[j].item(), mean_y[j].item(), std_y[j].item(), C_probs[j], Y[j].item()))

    return test_loss / len(test_loader.dataset)


# In[10]:


data = titanic_training[~titanic_training["Age"].isna()][["Survived", "Pclass", "Age", "Sex"]]
data["Sex"] = data["Sex"].map({'male': 1, 'female': 0})
data["Pclass"] = data["Pclass"] - 1
data = data.values

np.random.shuffle(data)
train_data, test_data = data[:int(0.7*len(data)),:], data[int(0.7*len(data)):,:]
train_dataset, test_dataset = [TitanicDataset(x) for x in [train_data, test_data]]

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)


# In[15]:


num_epochs = 1000
pyro.clear_param_store()
train_elbo = []
test_elbo = []

vae = LinearVAE().to(device)
scheduler = pyro.optim.StepLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': 0.1}, 'step_size': 400, 'gamma': 0.1 })

svi = SVI(vae.model, vae.guide, scheduler, loss=Trace_ELBO())

for epoch in range(num_epochs):

    total_epoch_loss_train = train(train_loader, svi)
    train_elbo.append(-total_epoch_loss_train)
    writer.add_scalar('ELBO/train', -total_epoch_loss_train, epoch)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % 4 == 0:
        # report test diagnostics
        total_epoch_loss_test = evaluate(test_loader, svi, vae, epoch)
        test_elbo.append(-total_epoch_loss_test)
        writer.add_scalar('ELBO/test', -total_epoch_loss_test, epoch)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))


print("done")