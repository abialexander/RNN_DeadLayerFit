from src.data import *
from src.RNN import *
from src.training import *
from tabulate import tabulate

def compareRNNs(maxFCCDdiff_list, NUM_EPOCHS_list,LEARNING_RATE = 0.005, dataset_size = 10000, RNN_ID_start="RNN_MC2DLF1"):
    """
    Compare the performance metrics of saved RNN models and output tables.
    Currently only for quantileRegressionDLF=True
    """
    
    CodePath = os.path.dirname(os.path.abspath("__file__"))
    
    accuracy_FCCD_list, accuracy_DLF_list = [], []
    roc_auc_FCCD_list = []
    RMSE_FCCDmisclassified_FCCD_list = []
    RMSE_DLFmisclassified_FCCD_list = []

    accuracy_FCCD_fulldataset_list, accuracy_DLF_fulldataset_list = [], []
    roc_auc_FCCD_fulldataset_list = []
    
    for ind, maxFCCDdiff in enumerate(maxFCCDdiff_list):
        NUM_EPOCHS = NUM_EPOCHS_list[ind]
        if maxFCCDdiff == "NA":
            RNN_ID =RNN_ID_start+"_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"
        else:
            RNN_ID =RNN_ID_start+"_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

        #Restricted Dataset
        accuracies_path = CodePath+"/saved_models/"+RNN_ID+"/test_accuracies.json"
        with open(accuracies_path) as json_file:
            data=json.load(json_file)
        accuracy_FCCD_list.append(round(data["accuracy_FCCD"],3))
        accuracy_DLF_list.append(round(data["accuracy_DLF"],3))
        roc_auc_FCCD_list.append(round(data["roc_auc_FCCD"],3))
        RMSE_FCCDmisclassified_FCCD_list.append(round(data["RMSEmisclassified"]["RMSE_FCCDmisclassified_FCCD"],3))
        RMSE_DLFmisclassified_FCCD_list.append(round(data["RMSEmisclassified"]["RMSE_DLFmisclassified_FCCD"],3))
        
        #Full Dataset
        if maxFCCDdiff == "NA":
            accuracies_path = CodePath+"/saved_models/"+RNN_ID+"/test_accuracies.json"
        else:
            accuracies_path = CodePath+"/saved_models/"+RNN_ID+"/test_accuracies_fulldataset.json"
        with open(accuracies_path) as json_file:
            data=json.load(json_file)
        accuracy_FCCD_fulldataset_list.append(round(data["accuracy_FCCD"],3))
        accuracy_DLF_fulldataset_list.append(round(data["accuracy_DLF"],3))
        roc_auc_FCCD_fulldataset_list.append(round(data["roc_auc_FCCD"],3))
    
    headers = [r"Max($\Delta_{FCCD}$) /mm", r"Accuracy FCCD", r"Accuracy DLF",  r"roc auc FCCD"]
    print("Evaluated on restricted test dataset")
    print(tabulate(list(zip(maxFCCDdiff_list, accuracy_FCCD_list, accuracy_DLF_list ,roc_auc_FCCD_list)), headers=headers,tablefmt="double_outline"))   
    print(tabulate(list(zip(maxFCCDdiff_list, accuracy_FCCD_list, accuracy_DLF_list ,roc_auc_FCCD_list)), headers=headers,tablefmt="latex_raw"))
    print("")
    print("Evaluated on full dataset")
    print(tabulate(list(zip(maxFCCDdiff_list, accuracy_FCCD_fulldataset_list, accuracy_DLF_fulldataset_list ,roc_auc_FCCD_fulldataset_list)), headers=headers,tablefmt="double_outline"))   
    print(tabulate(list(zip(maxFCCDdiff_list, accuracy_FCCD_fulldataset_list, accuracy_DLF_fulldataset_list ,roc_auc_FCCD_fulldataset_list)), headers=headers,tablefmt="latex_raw"))
    
    print("")
    headers = [r"Max($\Delta_{FCCD}$) /mm", r"RMSE FCCD Misclassified / mm", r"RMSE DLF Misclassified / mm"]
    print("Evaluated on restricted test dataset")
    print(tabulate(list(zip(maxFCCDdiff_list,RMSE_FCCDmisclassified_FCCD_list, RMSE_DLFmisclassified_FCCD_list)), headers=headers,tablefmt="double_outline"))   
    print(tabulate(list(zip(RMSE_FCCDmisclassified_FCCD_list, RMSE_DLFmisclassified_FCCD_list)), headers=headers,tablefmt="latex_raw"))
    
    #use: https://quicklatex.com/ to view latex table
    
def plotMultipleSpectra(spectra_list, label_list):
    "Plot multiple spectra on same hist"
    
    binwidth = 0.5 #keV
    bins = np.arange(5,450+binwidth,binwidth)
    bins_centres = np.delete(bins+binwidth/2,-1)
    
    plt.figure()
    for ind, spectrum in enumerate(spectra_list):
        plt.plot(bins_centres, spectrum, label=label_list[ind])
    
    plt.ylabel("Counts / "+str(binwidth)+" keV")
    plt.yscale("log")
    plt.xlim(0,450)
    plt.xlabel("Energy / keV")
    plt.tight_layout()
    plt.legend()
    plt.show()

def plotSpectrumDiff(spectrum_diff):
    "Plot spectrum difference"
    
    binwidth = 0.5 #keV
    bins = np.arange(5,450+binwidth,binwidth)
    bins_centres = np.delete(bins+binwidth/2,-1)
    plt.figure()
    plt.plot(bins_centres, spectrum_diff)
    plt.ylabel("Counts / "+str(binwidth)+" keV")
#     plt.yscale("log")
    plt.xlim(0,450)
    plt.xlabel("Energy / keV")
    plt.tight_layout()
    plt.hlines(0,0,450)
#     plt.ylim(-10,10)
    plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    