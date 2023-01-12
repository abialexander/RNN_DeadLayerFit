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

from src.RNN import *
from src.data import *


def train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, fulldataset=True, saveRNN=False, plot_training_results = False, RNN_ID = None):
    "function to create and train the RNN from a given dataset (test and train loader)"
    
    if RNN_ID is None and saveRNN==True:
        print("You must set RNN_ID to save RNN!")
        sys.exit()
    
    print(RNN_ID)
        
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CodePath = os.path.dirname(os.path.abspath("__file__"))
    
    BATCH_SIZE = train_loader.batch_size
    
    #Define RNN classifier
    RNNclassifier = RNN(dataset.get_histlen(),2) #only 2 classes
    RNNclassifier.to(DEVICE)

    print("#params", sum(x.numel() for x in RNNclassifier.parameters()))


    # Define categorical cross entropy loss
    RNNcriterion = torch.nn.BCELoss() #use binary cross entropy loss: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
    RNNcriterion = RNNcriterion.to(DEVICE)

    # Use lower learning rate at the first 400 iteration to "warm up" the attention mechanism
    warmup_size = 400
    print("Warmup Size: %d"%(warmup_size))
    lmbda = lambda epoch: min((epoch+1)**-0.5, (epoch+1)*warmup_size**-1.5)
    RNNoptimizer = torch.optim.AdamW(RNNclassifier.parameters(),lr=LEARNING_RATE, betas=(0.9, 0.98),eps=1e-9) #can change optimiser - read pytorch different optimisers

    scheduler = torch.optim.lr_scheduler.LambdaLR(RNNoptimizer, lr_lambda=lmbda)

    FCCD_accuracy_values, DLF_accuracy_values = [], []
    loss_values = []

    print("- - - - - - - - - - -")
    for epoch in range(NUM_EPOCHS):
        print("")
        print("EPOCH: ", epoch+1, "/",NUM_EPOCHS)
        print("Training network...")
        for i, (spectrum_diff, FCCDLabel, DLFLabel, extras, spectrum) in enumerate(train_loader):

            #RNN in train mode
            RNNclassifier.train() 

            #Send inputs and labels to DEVICE
            spectrum_diff = spectrum_diff.to(DEVICE).float()
            FCCDLabel = FCCDLabel.to(DEVICE).float() #(batch_size,)
            DLFLabel = DLFLabel.to(DEVICE).float() #(batch_size,)
            concat_labels = torch.stack([FCCDLabel, DLFLabel], dim=1) # (batch_size, 2)

            #Train RNN Classifier
            RNNoutputs  = RNNclassifier(spectrum_diff) #(batch_size, 2) 

            #Calculate Loss
            RNNloss = RNNcriterion(RNNoutputs, concat_labels)

            #Back-propagate Loss
            RNNloss.backward()

            # Perform gradient descent to update parameters
            RNNoptimizer.step()        # update parameters of net
            RNNoptimizer.zero_grad()   # reset gradient to 0

            scheduler.step()

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch+1, NUM_EPOCHS, i+1, len(train_loader),
                RNNloss.item(), end=""),end="")

        loss_values.append(RNNloss.item())

        print("")

        #================================================================================================

        RNNclassifier.eval() #RNN in evaluate mode



        # Plots per epoch
        fig, (ax_FCCD, ax_DLF) = plt.subplots(1, 2, figsize=(12,4))
        FCCD_labels_all = []
        FCCD_RNNoutputs_all = []
        FCCD_RNNoutputs_1 = [] #outputs where input label is 1
        FCCD_RNNoutputs_0 = [] #outputs where input label is 0

        DLF_labels_all = []
        DLF_RNNoutputs_all = []
        DLF_RNNoutputs_1 = [] #outputs where input label is 1
        DLF_RNNoutputs_0 = [] #outputs where input label is 0


        print("Testing/Validating...")
        for spectrum_diff, FCCDLabel, DLFLabel, extras, spectrum in tqdm(test_loader): 

            RNNclassifier.eval()
            labels = torch.stack([FCCDLabel, DLFLabel], dim=1) # (batch_size, 2)
            lb_data_in = labels.cpu().data.numpy()

            with torch.no_grad():
                spectrum_diff = spectrum_diff.to(DEVICE).float()
                outputs = RNNclassifier(spectrum_diff)
                outputs = outputs.cpu().data.numpy()

                for i in range(BATCH_SIZE):

                    FCCD_label = FCCDLabel[i].item()
                    DLF_label = DLFLabel[i].item()

                    RNNoutput_FCCD = outputs[i][0]
                    RNNoutput_DLF = outputs[i][1]


                    FCCD_labels_all.append(FCCD_label)
                    FCCD_RNNoutputs_all.append(RNNoutput_FCCD)

                    DLF_labels_all.append(DLF_label)
                    DLF_RNNoutputs_all.append(RNNoutput_DLF)

                    if FCCD_label == 1:
                        FCCD_RNNoutputs_1.append(RNNoutput_FCCD)
                    else:
                        FCCD_RNNoutputs_0.append(RNNoutput_FCCD)

                    if DLF_label == 1:
                        DLF_RNNoutputs_1.append(RNNoutput_DLF)
                    else:
                        DLF_RNNoutputs_0.append(RNNoutput_DLF)


        bins = np.linspace(0,1,201)  
        ax_FCCD.hist(np.array(FCCD_RNNoutputs_1), bins=bins, label = "label = 1", histtype="step")
        ax_FCCD.hist(np.array(FCCD_RNNoutputs_0), bins=bins, label = "label = 0", histtype="step")
        ax_FCCD.legend()
        ax_FCCD.set_xlabel("FCCD RNNoutput")
        ax_FCCD.set_yscale("log")

        ax_DLF.hist(np.array(DLF_RNNoutputs_1), bins=bins, label = "label = 1", histtype="step")
        ax_DLF.hist(np.array(DLF_RNNoutputs_0), bins=bins, label = "label = 0", histtype="step")
        ax_DLF.legend()
        ax_DLF.set_xlabel("DLF RNNoutput")
        ax_DLF.set_yscale("log")

#         plt.show()

        #Print accuracy after each epoch with a default boundary at 0.5
        accuracy_FCCD, precision_FCCD, recall_FCCD = compute_accuracy(0.5, FCCD_labels_all, FCCD_RNNoutputs_all)
        FCCD_accuracy_values.append(accuracy_FCCD)
        print("accuracy_FCCD: ", accuracy_FCCD)
        accuracy_DLF, precision_DLF, recall_DLF = compute_accuracy(0.5, DLF_labels_all, DLF_RNNoutputs_all)
        DLF_accuracy_values.append(accuracy_DLF)
        print("accuracy_DLF: ", accuracy_DLF)
    
    print("")
    print("Training complete.")
    print("")
        
    if saveRNN == True:
        model_path = CodePath+"/saved_models/"+RNN_ID+"/"+RNN_ID+".pkl"
        torch.save(RNNclassifier.state_dict(), model_path)
        print("Saving RNN at "+model_path)
        
    if plot_training_results == True:
        training_results(NUM_EPOCHS, loss_values, FCCD_accuracy_values, DLF_accuracy_values, save_plots = True, RNN_ID = RNN_ID)
    

    return FCCD_accuracy_values, DLF_accuracy_values, loss_values


def training_results(NUM_EPOCHS, loss_values, FCCD_accuracy_values, DLF_accuracy_values, save_plots = False, RNN_ID = None):
    "function to plot the loss_values and training accuracies as a function of epoch"
    
    CodePath = os.path.dirname(os.path.abspath("__file__"))
    
    # Loss function plot
    plt.figure()
    plt.plot(np.arange(NUM_EPOCHS).astype(int), loss_values)
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross Entropy Loss [a.u.]")
    plt.ylim(0,1)
    if save_plots == True:
        plt.savefig(CodePath+"/saved_models/"+RNN_ID+"/plots/training_loss_values.png")

    # accuracy plot
    plt.figure()
    plt.plot(np.arange(NUM_EPOCHS).astype(int), FCCD_accuracy_values, label = "FCCD")
    plt.plot(np.arange(NUM_EPOCHS).astype(int), DLF_accuracy_values, label = "DLF")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    if save_plots == True:
        plt.savefig(CodePath+"/saved_models/"+RNN_ID+"/plots/training_accuracies.png")
    
    
    
def compute_accuracy(cut,labels,outputs, print_results = False):
    "function to compute the accuracy, precision and recall of a binary classifier from its outputs and labels"
    
    #positive -> value of 1
    total = len(labels)

    TP, TN, FP, FN = 0, 0, 0, 0
    for i, label in enumerate(labels):

        if (label == 1) & (outputs[i] > cut):
            TP += 1

        if (label == 0) & (outputs[i] > cut):
            FP += 1

        if (label == 0) & (outputs[i] < cut):
            TN += 1

        if (label == 1) & (outputs[i] < cut):
            FN += 1
    
    
    #checking
    if print_results == True:
        print("total: ", total)
        print("TP: ", TP)
        print("FP: ", FP)
        print("TN: ", TN)
        print("FN: ", FN)
    

    accuracy = (TP+TN)/total #No. correct predictions / total number predictions
    precision = TP/(TP+FP) #What proportion of positive identifications was actually correct?
    recall = TP/(TP+FN) #What proportion of actual positives was identified correctly?
    
    return accuracy, precision, recall



# THIS FUNCTION IS NOT CURRENTLY USED
#This function gets the false positive rate, true positive rate, cutting threshold and area under curve using the given signal and background array
def get_roc(sig, bkg):
    testY = np.array([1]*len(sig) + [0]*len(bkg))
    predY = np.array(sig+bkg)
    auc = roc_auc_score(testY, predY)
    fpr, tpr, thr = roc_curve(testY, predY)
    return fpr,tpr,thr,auc


def test_RNN(RNNclassifier, test_loader, RNN_ID=None, performance_plots = False, misclassified_trials_plots = False, save_results = False, train_restricted_test_fulldataset = False):
    "function to test a trained RNN on given test dataset"
    
    if RNN_ID is None and save_results==True:
        print("You must set RNN_ID to save results!")
        return 0
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CodePath = os.path.dirname(os.path.abspath("__file__"))
    
    BATCH_SIZE = test_loader.batch_size
    
    RNNclassifier.eval()
    RNNclassifier.to(DEVICE)

    FCCD_labels_all = []
    FCCD_RNNoutputs_all = []
    FCCD_RNNoutputs_1 = [] #outputs where input label is 1
    FCCD_RNNoutputs_0 = [] #outputs where input label is 0

    DLF_labels_all = []
    DLF_RNNoutputs_all = []
    DLF_RNNoutputs_1 = [] #outputs where input label is 1
    DLF_RNNoutputs_0 = [] #outputs where input label is 0

    FCCD_RNNoutput_cut = 0.5
    DLF_RNNoutput_cut = 0.5

    #misclassified trials
    FCCD_misclassified_FCCD_diff = []
    FCCD_misclassified_FCCD1 = []
    FCCD_misclassified_FCCD2 = []
    FCCD_misclassified_FCCD_RNNoutput =[]
    DLF_misclassified_DLF_diff = []
    DLF_misclassified_DLF1 = []
    DLF_misclassified_DLF2 = []
    DLF_misclassified_DLF_RNNoutput =[]


    # Run test data set through trained RNN
    for j, (spectrum_diff, FCCDLabel, DLFLabel, extras, spectrum) in enumerate(tqdm(test_loader)): 

        labels = torch.stack([FCCDLabel, DLFLabel], dim=1) # (batch_size, 2)!
        lb_data_in = labels.cpu().data.numpy()

        with torch.no_grad():
            spectrum_diff = spectrum_diff.to(DEVICE).float()
            outputs = RNNclassifier(spectrum_diff)
            outputs = outputs.cpu().data.numpy()

            for i in range(BATCH_SIZE):

                FCCD_label = FCCDLabel[i].item()
                DLF_label = DLFLabel[i].item()

                RNNoutput_FCCD = outputs[i][0]
                RNNoutput_DLF = outputs[i][1]

                FCCD_labels_all.append(FCCD_label)
                FCCD_RNNoutputs_all.append(RNNoutput_FCCD)

                DLF_labels_all.append(DLF_label)
                DLF_RNNoutputs_all.append(RNNoutput_DLF)

                if FCCD_label == 1:
                    FCCD_RNNoutputs_1.append(RNNoutput_FCCD)
                else:
                    FCCD_RNNoutputs_0.append(RNNoutput_FCCD)

                if DLF_label == 1:
                    DLF_RNNoutputs_1.append(RNNoutput_DLF)
                else:
                    DLF_RNNoutputs_0.append(RNNoutput_DLF)


                #misclassified trials
                FCCD_pred = 1 if RNNoutput_FCCD > FCCD_RNNoutput_cut else 0
                DLF_pred = 1 if RNNoutput_DLF > DLF_RNNoutput_cut else 0
                FCCDmisclassified, DLFmisclassified = False, False
                if FCCD_pred != FCCD_label:
                    FCCDmisclassified = True
                    FCCD_diff, FCCD1, FCCD2 = extras["FCCD_diff"][i].item(), extras["FCCD1"][i].item(), extras["FCCD2"][i].item()
                    FCCD_misclassified_FCCD_diff.append(FCCD_diff)
                    FCCD_misclassified_FCCD1.append(FCCD1)
                    FCCD_misclassified_FCCD2.append(FCCD2)
                    FCCD_misclassified_FCCD_RNNoutput.append(RNNoutput_FCCD)
                if DLF_pred != DLF_label:
                    DLFmisclassified = True
                    DLF_diff, DLF1, DLF2 = extras["DLF_diff"][i].item(), extras["DLF1"][i].item(), extras["DLF2"][i].item()
                    DLF_misclassified_DLF_diff.append(DLF_diff)
                    DLF_misclassified_DLF1.append(DLF1)
                    DLF_misclassified_DLF2.append(DLF2)
                    DLF_misclassified_DLF_RNNoutput.append(RNNoutput_DLF)

                if FCCDmisclassified is True and DLFmisclassified is True:
                    print("DLF and FCCD misclassified for same trial:")
                    print("j: ", j, ", i: ", i)
                    print("FCCD1: ", FCCD1, ", FCCD2: ", FCCD2, ", FCCD_diff: ", FCCD_diff, ", RNNoutput: ", RNNoutput_FCCD)
                    print("DLF1: ", DLF1, ", DLF2: ", DLF2, ", DLF_diff: ", DLF_diff, ", RNNoutput: ", RNNoutput_DLF)


    #Compute accuracy
    print("FCCD accuracies: ")
    accuracy_FCCD, precision_FCCD, recall_FCCD = compute_accuracy(FCCD_RNNoutput_cut, FCCD_labels_all, FCCD_RNNoutputs_all, print_results=True)
    print("accuracy: ", accuracy_FCCD)
    print("precision: ", precision_FCCD)
    print("recall: ", recall_FCCD)
    print("")
    print("DLF accuracies: ")
    accuracy_DLF, precision_DLF, recall_DLF = compute_accuracy(DLF_RNNoutput_cut, DLF_labels_all, DLF_RNNoutputs_all, print_results=True)
    print("accuracy: ", accuracy_DLF)
    print("precision: ", precision_DLF)
    print("recall: ", recall_DLF)
    accuracies = {"accuracy_FCCD":accuracy_FCCD, "precision_FCCD":precision_FCCD, "recall_FCCD":recall_FCCD, "accuracy_DLF":accuracy_DLF, "precision_DLF":precision_DLF, "recall_DLF":recall_DLF}
    if save_results == True:
        if train_restricted_test_fulldataset == True:
            fn = CodePath+"/saved_models/"+RNN_ID+"/test_accuracies_fulldataset.json"
        else:
            fn = CodePath+"/saved_models/"+RNN_ID+"/test_accuracies.json"
        with open(fn, "w") as outfile:
            json.dump(accuracies, outfile, indent=4)
    
    # Performance Plots
    if performance_plots == True:
        fig, (ax_FCCD, ax_DLF) = plt.subplots(1, 2, figsize=(12,4))
        bins = np.linspace(0,1,201)  
        counts, bins, bars = ax_FCCD.hist(np.array(FCCD_RNNoutputs_1), bins=bins, label = "label = 1", histtype="step")
        ax_FCCD.hist(np.array(FCCD_RNNoutputs_0), bins=bins, label = "label = 0", histtype="step")
        ax_FCCD.legend()
        ax_FCCD.set_xlabel("FCCD RNNoutput")
        ax_FCCD.vlines(FCCD_RNNoutput_cut, min(counts), 2*max(counts), linestyles="dashed", color="gray", label ="cut")
        ax_FCCD.set_yscale("log")

        ax_DLF.hist(np.array(DLF_RNNoutputs_1), bins=bins, label = "label = 1", histtype="step")
        counts, bins, bars = ax_DLF.hist(np.array(DLF_RNNoutputs_0), bins=bins, label = "label = 0", histtype="step")
        ax_DLF.legend()
        ax_DLF.set_xlabel("DLF RNNoutput")
        ax_DLF.vlines(DLF_RNNoutput_cut, min(counts), 2*max(counts), linestyles="dashed", color="gray", label ="cut")
        ax_DLF.set_yscale("log")
        
        if save_results == True:
            if train_restricted_test_fulldataset == True:
                fn = CodePath+"/saved_models/"+RNN_ID+"/plots/test_RNN_performance_fulldataset.png"
            else:
                fn = CodePath+"/saved_models/"+RNN_ID+"/plots/test_RNN_performance.png"
            plt.savefig(fn)

#         plt.show()
    
    if misclassified_trials_plots == True:

        print("Total # misclassified trials FCCD: ", len(FCCD_misclassified_FCCD1), " /", len(FCCD_labels_all))
        print("Total # misclassified trials DLF: ", len(DLF_misclassified_DLF1), " /", len(FCCD_labels_all))

        #FCCD
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,4))
        fig.suptitle("Misclassified trials: FCCD", fontsize=12)
        bins = 20
        ax1.hist(np.array(FCCD_misclassified_FCCD_diff), bins=bins, label = "FCCD_diff", histtype="step")
        ax1.set_xlabel("FCCD diff / mm")
        ax1.set_xlim(-2,2)
        ax2.hist(np.array(FCCD_misclassified_FCCD1), bins=bins, label = "FCCD1", histtype="step")
        ax2.set_xlabel("FCCD 1 / mm")
        ax2.set_xlim(0,2)
        ax3.hist(np.array(FCCD_misclassified_FCCD2), bins=bins, label = "FCCD2", histtype="step")
        ax3.set_xlabel("FCCD 2 / mm")
        ax3.set_xlim(0,2)
        ax4.hist(np.array(FCCD_misclassified_FCCD_RNNoutput), bins=bins, label = "RNNoutput", histtype="step")
        ax4.set_xlabel("RNNoutput")
        ax4.set_xlim(0,1)
        ax1.text(0.05, 0.95, 'total: '+str(len(FCCD_misclassified_FCCD1)), transform=ax1.transAxes, fontsize=10,verticalalignment='top')
        ax1.set_ylabel("Frequency")
        if save_results == True:
            if train_restricted_test_fulldataset == True:
                fn = CodePath+"/saved_models/"+RNN_ID+"/plots/test_misclassified_hist_FCCD_fulldataset.png"
            else:
                fn = CodePath+"/saved_models/"+RNN_ID+"/plots/test_misclassified_hist_FCCD.png"
    
            plt.savefig(fn)

        #DLF
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,4))
        fig.suptitle("Misclassified trials: DLF", fontsize=12)
        bins = 20
        ax1.hist(np.array(DLF_misclassified_DLF_diff), bins=bins, label = "DLF_diff", histtype="step", color="orange")
        ax1.set_xlabel("DLF diff")
        ax1.set_xlim(-1,1)
        ax2.hist(np.array(DLF_misclassified_DLF1), bins=bins, label = "DLF1", histtype="step", color="orange")
        ax2.set_xlabel("DLF 1")
        ax2.set_xlim(0,1)
        ax3.hist(np.array(DLF_misclassified_DLF2), bins=bins, label = "DLF2", histtype="step", color="orange")
        ax3.set_xlabel("DLF 2")
        ax3.set_xlim(0,1)
        ax4.hist(np.array(DLF_misclassified_DLF_RNNoutput), bins=bins, label = "RNNoutput", histtype="step", color="orange")
        ax4.set_xlabel("RNNoutput")
        ax4.set_xlim(0,1)
        ax1.text(0.05, 0.95, 'total: '+str(len(DLF_misclassified_DLF1)), transform=ax1.transAxes, fontsize=10,verticalalignment='top')
        ax1.set_ylabel("Frequency")
        if save_results == True:
            if train_restricted_test_fulldataset == True:
                fn = CodePath+"/saved_models/"+RNN_ID+"/plots/test_misclassified_hist_DLF_fulldataset.png"
            else:
                fn = CodePath+"/saved_models/"+RNN_ID+"/plots/test_misclassified_hist_DLF.png"
            plt.savefig(fn)
        
        
        #FCCD and DLF
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        fig.suptitle("Misclassified trials", fontsize=12)
        ax1.scatter(FCCD_misclassified_FCCD1, FCCD_misclassified_FCCD2, label="trials")
        ax1.set_xlabel("FCCD 1 / mm")
        ax1.set_ylabel("FCCD 2 / mm")
        ax1.set_xlim(0,2)
        ax1.set_ylim(0,2)
        ax1.axline((0, 0), slope=1, color='grey', linestyle="dashed", label="y=x line")
        ax1.text(0.05, 0.95, 'total: '+str(len(FCCD_misclassified_FCCD1)), transform=ax1.transAxes, fontsize=10,verticalalignment='top')
        ax1.legend(loc="lower right")
        ax2.scatter(DLF_misclassified_DLF1, DLF_misclassified_DLF2, color="orange", label="trials")
        ax2.set_xlabel("DLF 1")
        ax2.set_ylabel("DLF 2")
        ax2.set_xlim(0,1)
        ax2.set_ylim(0,1)
        ax2.axline((0, 0), slope=1, color='grey', linestyle="dashed", label="y=x line")
        ax2.text(0.05, 0.95, 'total: '+str(len(DLF_misclassified_DLF1)), transform=ax2.transAxes, fontsize=10,verticalalignment='top')
        ax2.legend(loc="lower right")
        if save_results == True:
            if train_restricted_test_fulldataset == True:
                fn=CodePath+"/saved_models/"+RNN_ID+"/plots/test_misclassified_scatter_fulldataset.png"
            else:
                fn=CodePath+"/saved_models/"+RNN_ID+"/plots/test_misclassified_scatter.png"
            plt.savefig(fn)
    
    
    return accuracies


def plot_attention(spectrum, attscore, labels, ax= None, fig=None):
    '''
    This function plots the attention score distribution on given spectrum
    '''
    
    from matplotlib import cm
    from matplotlib import gridspec
    colormap_normal = cm.get_cmap("cool")
    
    spectrum=np.array(spectrum)
    attscore = np.array(attscore)
    fig, ax = plt.subplots(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[8,1]) 

    plt.subplot(gs[0])
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    len_spectrum = len(spectrum)
    plt.bar(np.linspace(0,450,len_spectrum), spectrum, width=1.5, color=colormap_normal(rescale(attscore)))
    plt.xlabel("Energy / keV")
    plt.ylabel("Counts")
    plt.yscale("log")

    loss_ax_scale = fig.add_subplot(gs[1])
    loss_ax_scale.set_xticks([])
    loss_ax_scale.tick_params(length=0)
    plt.yticks([1,72], ["High Attention", "Low Attention"], rotation=90)  # Set text labels and properties.

    loss_scale = np.linspace(1.0, 0.0, 100)

    for i in range(0,1):
        loss_scale = np.vstack((loss_scale,loss_scale))
    loss_scale = loss_ax_scale.imshow(np.transpose(loss_scale),cmap=colormap_normal, interpolation='nearest')

    plt.tight_layout()
    
    info_str = '\n'.join((r'FCCD 1=%s'%(labels["FCCD1"]), r'FCCD 2=%s'%(labels["FCCD2"]), r'DLF 1=%s'%(labels["DLF1"]), r'DLF 2=%s'%(labels["DLF2"])))
    plt.text(0, 0.98, info_str, transform=ax.transAxes, fontsize=10 ,verticalalignment='center') 
    
    return fig
    




