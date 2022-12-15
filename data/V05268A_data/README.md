## General details:
- Detector: V05268A
- Source: uncollimated Ba-133, at 78.5 mm above top of cryostat

## Post-Processed MC: the training data
- 1000 randomly generated combinations of FCCD and DLF (Dead Layer Fraction) for a linear transition layer. Stored in `training_data_V05268A/`
- Not a grid search in the end, but 1000 pairs of (FCCD,DLF) were randomly generated between 0-2 and 0-1 respectively. These pairs are stored in `1000random_FCCDs_DLFs.json`


## Opening the data and training data files:
- both the data (`data_hist_Ba_V05268A.h5`) and simulations (`training_data_V05268A/`) are energy histograms stored in hdf5 files.
- e.g. open with pandas: `df =  pd.read_hdf(<path>, key="energy_hist")`
- binning: 0.5 keV bins from 0-450 keV:
    ` binwidth = 0.5 #keV`
    ` bins = np.arange(0,450+binwidth,binwidth)`
    ` bins_centres = np.delete(bins+binwidth/2,-1)`

## Normalisation of the Simulations:
- The simulations/training data need to be normalised to the same number of events as the data. The best way to do this is based on the source activity and run time (see below). You should divide the simulation histogram counts by this scaling factor.
    ` #normalise MC to data`
    ` data_time = 30*60 #30 mins, 30*60s`
    ` Ba133_activity = 116.1*10**3 #Bq`
    ` data_evts = data_time*Ba133_activity`
    ` MC_evts = 10**8`
    ` MC_solidangle_fraction = 1/6 #30 degrees solid angle in MC`
    ` scaling = MC_evts/MC_solidangle_fraction/data_evts`