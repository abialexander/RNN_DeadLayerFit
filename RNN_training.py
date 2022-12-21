from src.data import *
from src.RNN import *
from src.functions import *



def main():

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CodePath = os.path.dirname(os.path.abspath("__file__"))
    
    
#     #============TRAINING 1===================
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
#     RNN_ID = "RNN_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"
    
#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #Load dataset
#     BATCH_SIZE = 4 
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size)
    
#     #train RNN
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID)
    
#     #=================TRAINING 2 (max FCCDdiff=0.5)===================
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
   
#     maxFCCDdiff = 0.5
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #Load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, restrict_dataset=True, restrict_dict = restrict_dict, size=dataset_size)
    
#     #train RNN
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID)
    
#     #=================TRAINING 3 (max FCCDdiff=0.25)===================
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
   
#     maxFCCDdiff = 0.25
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #Load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, restrict_dataset=True, restrict_dict = restrict_dict, size=dataset_size)
    
#     #train RNN
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID)
    
#     #=================TRAINING 4 (max FCCDdiff=0.25 and maxDLF=0.25)===================
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
   
#     maxFCCDdiff = 0.25
#     maxDLFdiff = 0.25 
#     RNN_ID ="RNN_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_maxDLFdiff"+str(maxDLFdiff)+"_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #Load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, restrict_dataset=True, restrict_dict = restrict_dict, size=dataset_size)
    
#     #train RNN
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID)
    
    
    #=================TRAINING 5 (max FCCDdiff=0.1)===================
    NUM_EPOCHS = 30 #can try increasing
    LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
    dataset_size = 10000
   
    maxFCCDdiff = 0.1
    maxDLFdiff = 1.0 #i.e. no restriction
    RNN_ID ="RNN_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

    #initialise directories to save
    if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
        os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
    #Load dataset
    BATCH_SIZE = 4 
    restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
    train_loader, test_loader, dataset = load_data(BATCH_SIZE, restrict_dataset=True, restrict_dict = restrict_dict, size=dataset_size)
    
    #train RNN
    FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID)



if __name__ == "__main__":
    main()