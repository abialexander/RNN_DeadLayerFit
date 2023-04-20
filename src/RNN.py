import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
import os
import argparse
import time
import math
import random
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import gzip
import pickle
import numpy as np
from torch.autograd import Variable
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import torchsnooper 
import h5py
import re
from tqdm import tqdm
from torch.cuda.amp import autocast

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#try increasing no of linear layers, from 2 to >2
#try increasing dropout rate

class FCNet(nn.Module):
    "fully connected part of Neural Network"
    def __init__(self, first_unit, last_unit):
        super(FCNet, self).__init__()
        
        #Number of channels in each fully connected layers
        fc1, fc2 = (first_unit, int(first_unit*0.25))
        do = 0.2
        self.fcnet = nn.Sequential(
            torch.nn.Linear(fc1, fc2),
            torch.nn.BatchNorm1d(fc2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(do),
#             torch.nn.Linear(fc2, fc3),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout(do),
#             torch.nn.Linear(fc3, fc4),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout(do),
            torch.nn.Linear(fc2, last_unit),
        )
    def forward(self, x):
        return self.fcnet(x)
    
class QuantileLoss(nn.Module):
    """Quantile regression loss function.
    https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/"""
    def __init__(self, q):
        super(QuantileLoss, self).__init__()
        
        self.q = q #quantile [0,1]
               
    def forward(self, yi, yip):
        #p=prediction, yi=truth value, q=quantile
        #For a set of values (i.e. a batch) the loss is the average
        Loss = torch.mean(torch.max(self.q*(yi-yip), (self.q-1)*(yi-yip)))
        return Loss
    


class RNN(nn.Module):
    "RNN Model"
    def __init__(self,histlen,num_class, get_attention = False, attention_mechanism = "normal"):
        super(RNN, self).__init__()
        
        bidirec = True
        self.bidirec = bidirec
        self.num_class = num_class
        feed_in_dim = 512
        self.seg = 1
        self.seq_len = histlen // self.seg
        if bidirec:
            self.RNNLayer = torch.nn.GRU(input_size = self.seg, hidden_size = feed_in_dim//2,num_layers=2, batch_first=True,bidirectional=True,dropout=0.2)
            feed_in_dim *= 2
        else:
            self.RNNLayer = torch.nn.GRU(input_size = self.seg, hidden_size = feed_in_dim//2,num_layers=2, batch_first=True,bidirectional=False,dropout=0.2)
        self.fcnet = FCNet(feed_in_dim,self.num_class) #only 2 classes/decisions to make: FCCD1>FCCD2 and DLF1>DLF2
        self.attention_weight = nn.Linear(feed_in_dim//2, feed_in_dim//2, bias=False)
        self.get_attention = get_attention
        self.attention_mechanism = attention_mechanism #="normal" or " cosine"

#     @torchsnooper.snoop() #uncomment for troubleshooting if training fails
    def forward(self, x):
        x = x.view(-1,self.seq_len,self.seg)
        bsize = x.size(0)
        output, hidden = self.RNNLayer(x)
        if self.bidirec:
            hidden =  hidden[-2:]
            hidden = hidden.transpose(0,1).reshape(bsize,-1)
        else:
            hidden =  hidden[-1]
        
        
        #Attention Mechanism
        if self.attention_mechanism == "normal":
            hidden_attention = hidden.unsqueeze(-1) #[batch, channel]
            w_attention = self.attention_weight(output) # [batch, seq_len, channel] * [channel, channel] -> [batch, seq_len, channel]
            w_attention = torch.einsum("ijl,ilm->ijm",w_attention,hidden_attention).squeeze(-1)   # [batch, seq_len, channel] * [batch, channel] -> [batch, seq_len]
            attention_score = torch.softmax(w_attention,dim=-1) #Softmax over seq_len dimension
        
#         #try other attention mechanism
        elif self.attention_mechanism == "cosine":
            inner_product = torch.einsum("ijl,il->ij",output, hidden)
            output_norm = torch.linalg.norm(output,dim=-1)
            hidden_norm = torch.linalg.norm(hidden,dim=-1).unsqueeze(-1).expand(output_norm.size())
            attention_score = torch.softmax(inner_product/(output_norm*hidden_norm),dim=-1) #Softmax over seq_len dimension
        
        if self.get_attention:
            return attention_score
        
        context = torch.sum(attention_score.unsqueeze(-1).expand(*output.size()) * output,dim=1) #Sum over seq_len dimension with attention score multiplied to output
        x = self.fcnet(torch.cat([context,hidden],dim=-1)) #concatenate context vector with last hidden state output
        x = torch.sigmoid(x) #forces NNoutput to be 0-1, means we can use BCE loss function and not BCEwithlogitloss
        # assert 0
        return x