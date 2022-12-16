# RNN Dead Layer Fit Project

### Project Aim:
Use a Recurrent Neural Network (RNN) as a classifier to make decisions about 
the dead layer parameters of high purity germanium detectors used in the LEGEND Experiment.

### Training Data
The training data are post-processed MC simulations of a germanium detector characterised at the HADES underground laboratory with different dead layer parameters (the labels). 
The detectors are exposed to uncollimated Ba-133, at 78.5 mm above top of cryostat and the data is the energy spectrum recorded by the detector.

For each detector (just V05268A for now), 1000 randomly generated combinations of FCCD (Full Charge Collection Depth) and DLF (Dead Layer Fraction) 
for a linear transition layer were created. This labeled data is stored in `data/<detector_name>_data/training_data_<detector_name>/` as binned histograms. 
In this same folder (`data/<detector_name>_data/`) there is also the single experimental data file. See `data/V05268A_data/README.md` for more info.

### RNN and Workflow
The entire workflow is currently contained to the jupyter notebook `RNN_DeadLayerFit.ipynb`. This contains all the classes (i.e. dataset and RNN) and functions
used to train and test RNNs. Different versions of the RNN (i.e. with different training parameters or trained with different restrictions of training data) 
are saved as pytorch pkl files in `saved_models/<RNN_ID>/'. In this folder we also save the results for each RNN - i.e. plots and json files with performance metrics 
(e.g. accuracy, recall and precision).
