
# Predict Umatching Compositions for Compositional Zero-Shot Learning

**Note:** Most of the code is borrowed from [https://github.com/ExplainableML/czsl](https://github.com/ExplainableML/co-cge).

<p align="center">
  <img src="utils/dependencymodelingflow.png" />
</p>

## Setup 

1. Clone the repo 

2. We recommend using Anaconda for environment setup. To create the environment and activate it, please run:
```
    conda env create --file environment.yml
    conda activate czsl
```

3. Go to the cloned repo and open a terminal. Download the datasets and embeddings, specifying the desired path (e.g. `DATA_ROOT` in the example):
```
    bash ./utils/download_data.sh ROOT_FOLDER
    mkdir logs
```


## Training
To train Co-CGE for OW-CZSL, the command is simply:
```
    python train.py --config configs/ld/MODEL_NAME/CONFIG_FILE --fast_eval 
```
where `MODEL_NAME` is one of cocge, kgsp, compcos and `CONFIG_FILE` is the path to the configuration file of the model. We suggest to use the `fast_eval` flag to speed up the test phase.

Note that the folder `configs` contains configuration files for Absence Modeling and all other methods, i.e. Absence modeling in `configs/ld`, CompCos in `configs/compcos`, and the other methods in `configs/baselines`.  

To train for OW-CZSL a non-open world method, just add `--open_world` after the command. E.g. for running SymNet in the open world scenario on Mit-States, the command is:
```
    python train.py --config configs/baselines/mit/symnet.yml --open_world
```
**Note 1:** Not all methods are compatible with the `fast_eval` (e.g. SymNet is one)
**Note 2:** To create a new config, all the available arguments are indicated in `flags.py`. 

## Test
 

**Closed World.** To test a model, the code is simple:
```
    python test.py --logpath LOG_DIR
```
where `LOG_DIR` is the directory containing the logs of a model.

**Open World.** To test a model in the open world setting, run:
```
    python test.py --logpath LOG_DIR --open_world --fast_eval
```

To test a model trained for OW-CZSL in the closed setting, run:
```
    python test.py --logpath LOG_DIR --open_world --fast_eval --closed_eval
```


## References
If you use this code, please cite
```

```
