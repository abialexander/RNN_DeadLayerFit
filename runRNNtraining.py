from src.data import *
from src.RNN import *
from src.training import *



def main():

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CodePath = os.path.dirname(os.path.abspath("__file__"))
    
    
#     #============TRAINING 1===================
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
#     attention_mechanism="normal"
#     RNN_ID = "RNN_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"
    
#     attention_mechanism="cosine"
#     RNN_ID = RNN_ID+"_"+attention_mechanism
    
#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #Load dataset
#     BATCH_SIZE = 4 
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size)
    
#     #train RNN
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, attention_mechanism=attention_mechanism)
    
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
    
    
#     #=================TRAINING 5 (max FCCDdiff=0.1)===================
#     NUM_EPOCHS = 30 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
   
#     maxFCCDdiff = 0.1
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
    
    
#     #=================TRAINING 6 (max FCCDdiff=0.05)===================
#     NUM_EPOCHS = 40 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
#     maxFCCDdiff = 0.05
#     maxDLFdiff = 1.0 #i.e. no restriction
#     attention_mechanism="normal"
#     RNN_ID ="RNN_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     attention_mechanism="cosine"
#     RNN_ID = RNN_ID+"_"+attention_mechanism
   
#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #Load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, restrict_dataset=True, restrict_dict = restrict_dict, size=dataset_size)
    
#     #train RNN
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, attention_mechanism=attention_mechanism)
    
    
#     #=================TRAINING 7 (max FCCDdiff=0.01)===================
#     NUM_EPOCHS = 80 #can try increasing
#     LEARNING_RATE = 0.01 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
#     maxFCCDdiff = 0.01
#     maxDLFdiff = 1.0 #i.e. no restriction
#     attention_mechanism="normal"
#     RNN_ID ="RNN_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

# #     attention_mechanism="cosine"
# #     RNN_ID = RNN_ID+"_"+attention_mechanism
   
#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #Load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, restrict_dataset=True, restrict_dict = restrict_dict, size=dataset_size)
    
#     #train RNN
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, attention_mechanism=attention_mechanism)
    

    
#     #=================TRAINING 8 (max FCCDdiff=0.005)===================
#     NUM_EPOCHS = 40 #can try increasing
#     LEARNING_RATE = 0.01 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
#     maxFCCDdiff = 0.005
#     maxDLFdiff = 1.0 #i.e. no restriction
#     attention_mechanism="normal"
#     RNN_ID ="RNN_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

# #     attention_mechanism="cosine"
# #     RNN_ID = RNN_ID+"_"+attention_mechanism
   
#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #Load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, restrict_dataset=True, restrict_dict = restrict_dict, size=dataset_size)
    
#     #train RNN
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, attention_mechanism=attention_mechanism)



#     #=================TRAINING 9 FCCD ONLY (FCCD ONLY max FCCDdiff=0.5)===================
#     CodePath = os.path.dirname(os.path.abspath("__file__"))
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_FCCDonly/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000
#     maxFCCDdiff = 0.05
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_FCCDonly_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load restricted dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, restrict_dataset=True, restrict_dict = restrict_dict, size=dataset_size, path = MC_PATH)
    
#     #run training
#     FCCD_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, FCCDonly = True)
    

#     #=================TRAINING 10 MC2 DLF=1, Quantile Regression, no restriction===================    
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000
#     RNN_ID = "RNN_MC2DLF1_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)


#     #=================TRAINING 11 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.5===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.5
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 12 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.25===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.25
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
    
#     #=================TRAINING 13 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.1===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.1
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
    
#     #=================TRAINING 14 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.05===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.05
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 15 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.01===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.01
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)


#     # All norm1_ratioInput
#     #=================TRAINING 16 MC2 DLF=1, Quantile Regression, no restriction, ===================    
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 30 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000
#     RNN_ID = "RNN_MC2DLF1_norm1_ratioInput_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, normaliseSpectraUnity = True, ratioSpectraRNNInput = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
    
#     #=================TRAINING 17 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.5===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 30 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.5
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_ratioInput_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, ratioSpectraRNNInput = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 18 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.25===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 30 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.25
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_ratioInput_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, ratioSpectraRNNInput = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
#     #=================TRAINING 19 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.1===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 30 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.1
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_ratioInput_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, ratioSpectraRNNInput = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True) 

#     #=================TRAINING 20 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.05===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 30 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.05
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_ratioInput_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, ratioSpectraRNNInput = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
#     #=================TRAINING 21 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.01===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 30 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.01
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_ratioInput_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, ratioSpectraRNNInput = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)




#     # All norm1, separate scalers
# #     #=================TRAINING 22 MC2 DLF=1, Quantile Regression, no restriction, ===================    
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000
#     RNN_ID = "RNN_MC2DLF1_norm1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, normaliseSpectraUnity = True, separate_scalers=True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 23 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.5===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.5
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
#     #=================TRAINING 24 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.25===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.25
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 25 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.1===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.1
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 26 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.05===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.05
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 27 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.01===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.01
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_norm1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, normaliseSpectraUnity = True, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)


#     # all separate scalers
# #     #=================TRAINING 28 MC2 DLF=1, Quantile Regression, no restriction, ===================    
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000
#     RNN_ID = "RNN_MC2DLF1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, separate_scalers=True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 29 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.5===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.5
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
#     #=================TRAINING 30 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.25===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.25
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
#     #=================TRAINING 31 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.1===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.1
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
#     #=================TRAINING 32 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.05===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.05
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)    
#     #=================TRAINING 33 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.01===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.01
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_2scalers_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, separate_scalers = True)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)    
    
    
    #maxE=425
# #=================TRAINING 34 MC2 DLF=1, Quantile Regression, no restriction===================    
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000
#     RNN_ID = "RNN_MC2DLF1_maxE425_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, maxE=425)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 35 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.5===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.5
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_maxE425_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=425)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
#     #=================TRAINING 36 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.25===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.25
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_maxE425_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=425)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)   

#     #=================TRAINING 37 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.1===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.1
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_maxE425_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=425)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 38 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.05===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.05
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_maxE425_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=425)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 39 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.01===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.01
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_maxE425_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=425)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)


    #maxE=400
# #=================TRAINING 40 MC2 DLF=1, Quantile Regression, no restriction===================    
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000
#     RNN_ID = "RNN_MC2DLF1_maxE400_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_fulldataset_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, maxE=400)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)
    
#     #=================TRAINING 41 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.5===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.5
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_maxE400_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=400)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 42 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.25===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.25
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_maxE400_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=400)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 43 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.1===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.1
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_maxE400_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=400)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

#     #=================TRAINING 44 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.05===================
#     MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
#     MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
#     NUM_EPOCHS = 20 #can try increasing
#     LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
#     dataset_size = 10000 #10000

#     maxFCCDdiff = 0.05
#     maxDLFdiff = 1.0 #i.e. no restriction
#     RNN_ID ="RNN_MC2DLF1_maxE400_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

#     #initialise directories to save
#     if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
#         os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
#     #load dataset
#     BATCH_SIZE = 4 
#     restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
#     train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=400)

#     #run training
#     FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)

    #=================TRAINING 45 MC2 DLF=1, Quantile Regression, restriction max FCCD 0.01===================
    MC_PATH = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/"
    MC_PATH_FCCDonly = CodePath+"/data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/"
    
    NUM_EPOCHS = 20 #can try increasing
    LEARNING_RATE = 0.005 #0.01 #try modifying learning rate #0.001 too low for 30 epochs, 0.01 may be too high
    dataset_size = 10000 #10000

    maxFCCDdiff = 0.01
    maxDLFdiff = 1.0 #i.e. no restriction
    RNN_ID ="RNN_MC2DLF1_maxE400_"+str(NUM_EPOCHS)+"epochs_LR"+str(LEARNING_RATE)+"_maxFCCDdiff"+str(maxFCCDdiff)+"mm_"+str(dataset_size)+"trials"

    #initialise directories to save
    if not os.path.exists(CodePath+"/saved_models/"+RNN_ID+"/plots/"):
        os.makedirs(CodePath+"/saved_models/"+RNN_ID+"/plots/")
    
    #load dataset
    BATCH_SIZE = 4 
    restrict_dict = {"maxFCCDdiff": maxFCCDdiff, "maxDLFdiff": maxDLFdiff}
    train_loader, test_loader, dataset = load_data(BATCH_SIZE, size=dataset_size, path_MC2 = MC_PATH_FCCDonly, restrict_dataset=True, restrict_dict = restrict_dict, maxE=400)

    #run training
    FCCD_accuracy_values, DLF_accuracy_values, loss_values = train_RNN(dataset, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, saveRNN=True, plot_training_results = True, RNN_ID = RNN_ID, quantileRegressionDLF=True)


















if __name__ == "__main__":
    main()