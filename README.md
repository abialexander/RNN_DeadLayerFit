# RNN Dead Layer Fit Project

## Project Aim:
Use a Recurrent Neural Network (RNN) as a classifier to make decisions about 
the dead layer parameters of high purity germanium detectors used in the LEGEND Experiment.
### Original Plan
1) Generating MC spectrums with different [FCCD Width, Dead Layer Ratio, transfer function], normalize them to contain the same number of events -> Done
2) Randomly pull out 2 MC spectrum(MC1 and MC2) and subtract them to obtain MCdiff = (MC1-MC2)
3) Feed MCdiff into RNN and train to answer questions: is FCCD Width of MC1>MC2? is dead layer ration of MC1>MC2? etc. 
    - RNN: https://github.com/legend-exp/gem/blob/master/NetworkPSA/NetworkPSA_RNN.ipynb
4) Use the traditional peak-fitting technology you and Valentina developed, this will give you a MC spectrum given certain dead layer parameters, let’s call it MCbest
5) Calculate a new difference by doing MC_newdiff = Data-MCbest, then feeding MC_newdiff into the trained RNN. RNN will tell you if the [FCCD width, dead layer ratio,…] in MCbest is too low/too high.
6) Looking at the attention score, this will tell you which part of the spectrum the network has used to make the decision.
7) Trying to understand why a too low/too high FCCD width would have that kind of effect on that part of the spectrum, is it because of the MC simulation? transfer function? etc.

## Training Data
The training data are post-processed MC simulations of a germanium detector characterised at the HADES underground laboratory with different dead layer parameters (the labels). 
The detectors are exposed to uncollimated Ba-133, at 78.5 mm above top of cryostat and the data is the energy spectrum recorded by the detector.
For each detector (just V05268A for now), randomly generated combinations of FCCD (Full Charge Collection Depth) and DLF (Dead Layer Fraction) 
for a linear transition layer were created. 
This labeled data is stored as binned histograms. 

- `data/V05268A_data/training_data_V05268A_5000randomFCCDs_randomDLFs/`: ~5000 random FCCDs and random DLFs
- `data/V05268A_data/training_data_V05268A_5000randomFCCDs_DLF1/`: 5000 random FCCDs, all DLFs=1.0
- `data/V05268A_data/data_hist_Ba_V05268A.h5`: single experimental data file

### RNN and Workflow
Functions and classes: `src/`
- `src/data.py`: Classes and functions for handling of the training dataset
- `src/RNN.py`: Classes and functions for the RNN
- `src/training.py`: Classes and functions for training and evaluating of the RNN

The workflow is controlled by jupyter notebooks that use the functions and classes from `/src/'
- `RNN_DeadLayerFit.ipynb`: Original notebook,  for fitting FCCD and DLF, where MC 2 DLF not fixed
- `RNN_DeadLayerFit_MC2DLF1.ipynb`: Training where MC2 has a fixed DLF=1 and a quantile regression loss fucntion is used on DLF Label
- `RNN_DeadLayerFit_FCCDonly.ipynb`: Training where all training data has DLF=1.0, and only FCCD label is used

### Results
Different versions of the RNN (i.e. with different training parameters or trained with different restrictions of training data) 
are saved as pytorch pkl files in `saved_models/<RNN_ID>/`. 
In this folder we also save the results for each RNN - i.e. plots and json files with performance metrics 
(e.g. accuracy, recall and precision).
These are excluded from the github (they are in the .gitignore) with the exception of RNN_ID = `RNN_MC2DLF1_20epochs_LR0.005_fulldataset_10000trials`.




