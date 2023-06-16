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

def find_hist_quantile(counts, bins, q):
    "find quantile bin of a histogram"
    pdf = counts/sum(counts) #convert hist counts to a pdf normalised to 1
    cs = np.cumsum(pdf)
    bin_idx = np.where(cs > q)[0][0]
    bin_quantile = bins[bin_idx]
    print(bin_quantile)
    return bin_quantile

def plot_FCCD_pdf(RNN_ID, q=0.9):
    "plot the pdf of RNN FCCD output on test training data, with quantiles"
    
    CodePath = os.path.dirname(os.path.abspath("__file__"))
    
    #open hist
    CodePath = os.path.dirname(os.path.abspath("__file__"))
    df = pd.read_csv(CodePath+"/saved_models/"+RNN_ID+"/FCCD_test_classification_hist.csv")
    bins_centres, counts0, counts1 = df.bins_centres, df.counts0, df.counts1
    bins = np.linspace(0,1,201) 
    
    #compute 10% and 90% quantiles for exclusion region
    pdf1_q_low = find_hist_quantile(counts1, bins, 1-q)
    pdf0_q_up = find_hist_quantile(counts0, bins, q)
    
    #plot
    fig, ax = plt.subplots()
    plt.step(bins_centres, counts0/sum(counts0), linestyle='-',linewidth=1, label = "FCCD truth label = 0", color="blue")
    plt.step(bins_centres, counts1/sum(counts1), linestyle='-',linewidth=1, label = "FCCD truth label = 1", color="orange")
    plt.vlines(pdf1_q_low, min(counts1), 1/2, ls="-.", color="orange")
    plt.vlines(pdf0_q_up, min(counts1), 1/2, ls="-.", color="blue")
    # plt.Rectangle( [pdf0_q90,min(counts1)], pdf1_q10-pdf0_q90, 1/2, color="grey") #, angle=0.0 ÷, rotation_point='xy')
    plt.legend(loc="upper left", title="RNN output region of uncertainty: "+ str(pdf0_q_up)+" - "+str(pdf1_q_low), title_fontsize=9, fontsize=9)
    plt.xlabel("RNN Output FCCD")
    plt.ylabel("PDF")
    plt.ylim(0.5*10**(-3), 2)
    plt.yscale("log")
    plt.savefig(CodePath+"/saved_models/"+RNN_ID+"/plots/FCCD_test_classification_pdf.png")
    
    return pdf1_q_low, pdf0_q_up    
    

# def plot_FCCD_pdf(RNN_ID):
#     "plot the pdf of RNN FCCD output on test training data, with quantiles"
    
#     #open hist
#     CodePath = os.path.dirname(os.path.abspath("__file__"))
#     df = pd.read_csv(CodePath+"/saved_models/"+RNN_ID+"/FCCD_test_classification_hist.csv")
#     bins_centres, counts0, counts1 = df.bins_centres, df.counts0, df.counts1
#     bins = np.linspace(0,1,201) 
    
#     #compute 10% and 90% quantiles for exclusion region
#     pdf1_q10 = find_hist_quantile(counts1, bins, 0.1)
#     pdf0_q90 = find_hist_quantile(counts0, bins, 0.9)
    
#     #plot
#     fig, ax = plt.figure()
#     plt.step(bins_centres, counts0/sum(counts0), linestyle='-',linewidth=1, label = "FCCD truth label = 0", color="blue")
#     plt.step(bins_centres, counts1/sum(counts1), linestyle='-',linewidth=1, label = "FCCD truth label = 1", color="orange")
#     plt.vlines(pdf1_q10, min(counts1), 1/2, ls="-.", color="orange")
#     plt.vlines(pdf0_q90, min(counts1), 1/2, ls="-.", color="blue")
#     # plt.Rectangle( [pdf0_q90,min(counts1)], pdf1_q10-pdf0_q90, 1/2, color="grey") #, angle=0.0 ÷, rotation_point='xy')
#     plt.legend(loc="upper left", title="RNN output region of uncertainty: "+ str(pdf0_q90)+" - "+str(pdf1_q10), title_fontsize=9, fontsize=9)
#     plt.xlabel("RNN Output FCCD")
#     plt.ylabel("PDF")
#     plt.ylim(0.5*10**(-3), 2)
#     plt.yscale("log")
#     plt.savefig(CodePath+"/saved_models/"+RNN_ID+"/plots/FCCD_test_classification_pdf.png")

    
def compare_attention(maxFCCDdiff_list, NUM_EPOCHS_list, test_loader, RNN_ID_start="RNN_MC2DLF1", 
                      LEARNING_RATE = 0.005, dataset_size = 10000, savePlots= False, maxE=450):
    
    CodePath = os.path.dirname(os.path.abspath("__file__"))
    
    fig, axs = plt.subplots(nrows=len(maxFCCDdiff_list)+1, sharex=True, figsize=(8.27, 11.69)) #, figsize=(8,6))
    binwidth = 0.5 #keV
    bins = np.arange(5,maxE+binwidth,binwidth)
    bins_centres = np.delete(bins+binwidth/2,-1)
    average_attention_list = []
    for ind, maxFCCDdiff in enumerate(maxFCCDdiff_list):
        NUM_EPOCHS = NUM_EPOCHS_list[ind]
        if maxFCCDdiff == "NA":
            RNN_ID =RNN_ID_start+"_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"
        else:
            RNN_ID =RNN_ID_start+"_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"
        df = pd.read_csv(CodePath+"/saved_models/"+RNN_ID+"/average_attention_hist.csv")
        average_attention = df.average_attention
        average_attention_list.append(average_attention)

        axs[ind].plot(bins_centres, average_attention, label = "maxFCCDdiff: "+str(maxFCCDdiff)+" mm")
        axs[ind].set_ylabel("Attention")
        axs[ind].set_yscale("log")
        axs[ind].set_xlim(0,maxE)
        axs[ind].legend(loc="lower right", fontsize=8)
        axs[ind].grid()
        axs[ind].minorticks_on()
        
    #get random spectra to plot:
    test_spectrum_diff, test_FCCDLabel, test_DLFLabel, test_extras, test_spectrum = next(iter(test_loader))
    test_spectrum = test_spectrum[1].numpy()
    axs[-1].plot(bins_centres, test_spectrum, color="red", label="Example Spectrum")
    axs[-1].legend(loc="lower right", fontsize=8)
    axs[-1].set_xlim(0,maxE)
    axs[-1].set_ylabel("Spectrum")
    axs[-1].set_yscale("log")
    axs[-1].set_xlabel("Energy / keV")
    axs[-1].set_xlim(0,maxE)
    axs[-1].grid()
    axs[-1].minorticks_on()
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
#     plt.suptitle(RNN_ID)

    if savePlots == True:
        plt.savefig(CodePath+"/Results/"+RNN_ID_start+"_AttentionComparison.pdf")    
    
    

    
    
    
    
    
    
    