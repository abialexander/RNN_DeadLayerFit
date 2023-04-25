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


def plotTrainingData(folder, bins=50):
    FCCD_list, DLF_list = [],[]
    for filename in os.listdir(folder):
        m = re.search('FCCD(.+?)mm_DLF', filename)
        n = re.search('DLF(.+?)_frac', filename)
        a = (float(m.group(1)),float(n.group(1)))
        FCCD_list.append(float(m.group(1)))
        DLF_list.append(float(n.group(1)))
    no_files = len(FCCD_list)
    print("no files: ", str(no_files))
    fig, (ax_FCCD, ax_DLF) = plt.subplots(1, 2, figsize=(12,4))
    ax_FCCD.hist(FCCD_list,bins, histtype="step")
    ax_DLF.hist(DLF_list,bins, histtype="step")
    ax_FCCD.set_xlabel("FCCD / mm")
    ax_DLF.set_xlabel("DLF")
    ax_FCCD.set_ylabel("# training files / bin")
    ax_DLF.set_ylabel("# training files / bin")
    info = "\n".join([r'#files: '+str(no_files), r'#bins: '+str(bins)])
    ax_FCCD.text(0.05, 0.1, info, transform=ax_FCCD.transAxes, fontsize=10,verticalalignment='top')
    ax_DLF.text(0.05, 0.1, info, transform=ax_DLF.transAxes, fontsize=10,verticalalignment='top')
#     fig.suptitle(folder)

    fig2, ax2 = plt.subplots()
    ax2.scatter(FCCD_list, DLF_list, s=4)
    ax2.set_xlabel("FCCD / mm")
    ax2.set_ylabel("DLF")
#     fig2.suptitle(folder)
    
    plt.show()


class DL_Dataset(Dataset):
    "class for the dataset: MC energy spectra with different FCCD and DLF labels"

    CodePath = os.path.dirname(os.path.abspath("__file__"))

    def __init__(self, path, restrict_dataset = False, restrict_dict = None, size=1000, path_MC2 = None):
        
        
        self.size = size #this is the no. pairs of MC spectra to sample 
        self.hist_length = 900 #Energy Spectrum # of bins #binning: 0.5 keV bins from 0-450 keV:
        self.energy_bin = np.linspace(0,450.0,self.hist_length+1) # Only look at events between 0 and 450 keV
        self.restrict_dataset = restrict_dataset
        self.restrict_dict = restrict_dict
        
        #---------------------------------
        #MC1 (and MC2 if pathMC2 = None)
        #---------------------------------
        self.path = path
        self.event_dict = {}
        count = 0
        # Loop through all the files
        for filename in os.listdir(path):
            if count > 111111:
                break
            count += 1

            m = re.search('FCCD(.+?)mm_DLF', filename)
            n = re.search('DLF(.+?)_frac', filename)
            a = (float(m.group(1)),float(n.group(1)))
        
            if a:
                self.event_dict[a] = os.path.join(path, filename)
                        
        self.event_list = list(self.event_dict.keys()) #list of [FCCD, DLF]
        self.data_size = len(self.event_list) #this is the no. MC spectra
        self.scaler = self.build_scaler() #performs a ~normalisation on each bin so that they are all significant
         
        #---------------------------------
        #MC2 (IF MC1 AND MC2 FROM DIFFERENT PATHS)
        #---------------------------------
        self.path_MC2 = path_MC2
        if self.path_MC2 is not None:
            self.event_dict_MC2 = {}
            count = 0
            # Loop through all the files
            for filename in os.listdir(path_MC2):
                if count > 111111:
                    break
                count += 1

                m = re.search('FCCD(.+?)mm_DLF', filename)
                n = re.search('DLF(.+?)_frac', filename)
                a = (float(m.group(1)),float(n.group(1)))

                if a:
                    self.event_dict_MC2[a] = os.path.join(path_MC2, filename)
            
            self.event_list_MC2 = list(self.event_dict_MC2.keys()) #list of [FCCD, DLF]
            self.data_size_MC2 = len(self.event_list_MC2) #this is the no. MC spectra
            self.scaler_MC2 = self.build_scaler(MC2=True) #performs a ~normalisation on each bin so that they are all significant
   
        
    def __len__(self):
        return self.size
    
    def get_histlen(self):
        return self.hist_length

    def build_scaler(self, MC2=False):
        "applies standard scaler to each spectrum, converting them in to 0-centered with 1 standard deviation."
        hist_array = []
        
        if MC2 == False:   
            for i in tqdm(range(self.data_size)):
                FCCD, DLF = self.event_list[i][0], self.event_list[i][1]
                dead_layer_address = self.event_dict[(FCCD, DLF)]
                hist_array.append(self.get_hist_magnitude(dead_layer_address).reshape(1,-1))
        else:   
            for i in tqdm(range(self.data_size_MC2)):
                FCCD, DLF = self.event_list_MC2[i][0], self.event_list_MC2[i][1]
                dead_layer_address = self.event_dict_MC2[(FCCD, DLF)]
                hist_array.append(self.get_hist_magnitude(dead_layer_address).reshape(1,-1))            
        
        hist_array = np.concatenate(hist_array,axis=0)
        print(hist_array.shape)
        scaler = StandardScaler()
        scaler.fit(hist_array)
        return scaler
    
    
    def get_hist_magnitude(self,h5_address,MC=True):
        "This gets the energy spectrum of each file"
        df =  pd.read_hdf(h5_address, key="energy_hist")
        counts = df[0].to_numpy()
        bins = self.energy_bin #size 901, 0-450keV 0.5keV width
          
        if MC == True: #data doesnt need normalising, only MC
            counts = self.normalise_MC_counts(counts) 
        return counts
    
    
    def normalise_MC_counts(self, counts):
        "Normalises MC histograms to data histogram"
        #normalise MC to data`
        data_time = 30*60 #30 mins, 30*60s
        Ba133_activity = 116.1*10**3 #Bq
        data_evts = data_time*Ba133_activity
        MC_evts = 10**8
        MC_solidangle_fraction = 1/6 #30 degrees solid angle in MC
        scaling = MC_evts/MC_solidangle_fraction/data_evts
        
        counts_normalised = counts/scaling
        
        return counts_normalised
        

    def __getitem__(self, idx):
        
        #OLD: 1st spectra read idx, 2nd spectra is random, then compute difference
        #NEW: both spectra random
        
        #1st spectrum
        idx = np.random.randint(self.data_size)
        FCCD, DLF = self.event_list[idx][0], self.event_list[idx][1]
        dead_layer_address = self.event_dict[(FCCD, DLF)]
        spectrum_original = self.get_hist_magnitude(dead_layer_address)
        spectrum = self.scaler.transform(self.get_hist_magnitude(dead_layer_address).reshape(1,-1))
        
        #2nd spectrum
        if self.path_MC2 is None:
            idx2 = np.random.randint(self.data_size)
            while idx2 == idx:
                idx2 = np.random.randint(self.data_size) #ensures we dont have same ind
            FCCD2, DLF2 = self.event_list[idx2][0], self.event_list[idx2][1]
        else:
            idx2 = np.random.randint(self.data_size_MC2)
            FCCD2, DLF2 = self.event_list_MC2[idx2][0], self.event_list_MC2[idx2][1]
          
        FCCD_diff, DLF_diff = FCCD-FCCD2, DLF - DLF2

        #for restricted datasets, ensure FCCDdiff and DLFdiff satisfy given restriction
        if self.restrict_dataset == True: 
            while abs(FCCD_diff) > self.restrict_dict["maxFCCDdiff"] or abs(DLF_diff) > self.restrict_dict["maxDLFdiff"]:
                if self.path_MC2 is None:
                    idx2 = np.random.randint(self.data_size)
                    while idx2 == idx:
                        idx2 = np.random.randint(self.data_size) #ensures we dont have same ind
                    FCCD2, DLF2 = self.event_list[idx2][0], self.event_list[idx2][1]
                else:
                    idx2 = np.random.randint(self.data_size_MC2)
                    FCCD2, DLF2 = self.event_list_MC2[idx2][0], self.event_list_MC2[idx2][1]
                FCCD_diff, DLF_diff = FCCD-FCCD2, DLF - DLF2

        if self.path_MC2 is None:
            dead_layer_address2 = self.event_dict[(FCCD2, DLF2)]
            spectrum2 = self.scaler.transform(self.get_hist_magnitude(dead_layer_address2).reshape(1,-1))
        else:
            dead_layer_address2 = self.event_dict_MC2[(FCCD2, DLF2)]
            spectrum2 = self.scaler_MC2.transform(self.get_hist_magnitude(dead_layer_address2).reshape(1,-1))     
        
        #compute difference and make binary label
        if FCCD_diff >=0:
            FCCD_diff_label = 1
        else:
            FCCD_diff_label = 0
        if DLF_diff >=0:
            DLF_diff_label = 1
        else:
            DLF_diff_label = 0
        
        if self.path_MC2 is not None: #when DLF2=1, dont want binary label
            DLF_diff_label = DLF 
            
        spectrum_diff = spectrum - spectrum2
        
        #extras = info needed to investigate specific trials
        extras = {"FCCD1": FCCD, "FCCD2": FCCD2, "FCCD_diff": FCCD_diff, "DLF1": DLF, "DLF2": DLF2, "DLF_diff": DLF_diff}
        
        return spectrum_diff, FCCD_diff_label, DLF_diff_label, extras, spectrum_original
    
        
    def get_data(self):
        return self.scaler.transform(self.get_hist_magnitude(DATA_PATH, MC=False).reshape(1,-1))
    
    def get_hist_range(self):
        return self.energy_bin
    
    def get_scaler(self):
        return self.scaler
    


#Load dataset
def load_data(batch_size, restrict_dataset = False, restrict_dict = None, size=1000, path = None, path_MC2 = None):
    "function to load the dataset"
    
    CodePath = os.path.dirname(os.path.abspath("__file__"))
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A/"
    MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
    
    if restrict_dataset == True and restrict_dict is None:
        print("You must use kwarg restrict_dict in order to restrict dataset")
        return 0

    if path is None:
        path = MC_PATH
    dataset = DL_Dataset(restrict_dataset = restrict_dataset, restrict_dict=restrict_dict, size=size, path=path, path_MC2 = path_MC2)
    validation_split = .3 #Split data set into training & testing with 7:3 ratio
    shuffle_dataset = True
    random_seed= 42222

    dataset_size = int(len(dataset))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    test_loader = data_utils.DataLoader(dataset, batch_size=batch_size,sampler=valid_sampler,  drop_last=True)

    return train_loader,test_loader, dataset  
    