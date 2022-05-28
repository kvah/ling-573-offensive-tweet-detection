# ling-573-group-repo
## Task Description

### Primary Task
An end-to-end system for classifying English tweets as offensive or non-offensive, based on the [OffensEval 2019 Shared Task](https://sites.google.com/site/offensevalsharedtask/offenseval2019) (subtask A).

### Adaptation Task
An end-to-end system for classifying Greek tweets as offensive or non-offensive, based on the [OffensEval 2020 Shared Task](https://sites.google.com/site/offensevalsharedtask/results-and-paper-submission) (subtask A).


## Changes in D4

### Primary Task
#### Embeddings and Classification
- GloVe embedding + Bidirectional LSTM -> RoBERTa-base model
- Model finetuning and hypertuning

### Adaptation Task
#### Additional Preproccessing
- Removing diacritics
- Convert unicode data into ASCII characters
- Lemmatization
#### Embeddings and Classification
- XML-RoBERTa model
- Model finetuning and hypertuning

## Instructions

### 1. Prerequisites
#### Install Anaconda
If necessary, download and install anaconda by running the following commands:
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
```

#### Download best models for primary and adaptation tasks
- Download the [best model for primary task](https://drive.google.com/drive/u/2/folders/1KYS1PpH_jKT4wz94Kut1H7wnGopEI5Rb) and place the entire folder (containing `config.json` and `pytorch.bin`) in `models/`
- Download the [best model for adaptation task](https://drive.google.com/drive/folders/1-BlV1p9GvdiQblCWJ_M-yjh4nszmYypw) and place the entire folder (containing `config.json` and `pytorch.bin`) in `models/`
- Note that the model for **primary task** (the folder containing `config.json` and `pytorch.bin`) should be named `finetune_roberta` and the model for **adaptation task** should be named `finetune_xlmr_large_final_greek`


<!-- ### 2. Create the conda environment and run the following commands

- If the conda environment was not created previously, run the following:
``` 
conda create --prefix ./573_gpu python=3.8
conda activate ./573_gpu
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda env update --prefix ./573_gpu --file env.yml --prune
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --force-reinstall
```

- If the conda environment exists and you want to update the dependencies, run the following:
``` 
conda activate ./573_gpu
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda env update --prefix ./573_gpu --file env.yml --prune
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --force-reinstall
```
Note: Creating or updating the conda environment can sometimes take a while (30-60 min) -->

### 2. Run the Condor Script

```
condor_submit D4.cmd
```

**Notes:** 
- For the purposes of this deliverable, preprocessing and training are commented out from the main script (`D4_run.sh`).
- The condor script activates an existing conda environment. No additional action needed to create conda environment.


In summary, the pipeline:
1. Pre-processes SOLID Greek data.
2. Finetunes pretained model (XML-RoBERTa) on Greek training data.
3. Runs finetuned model predictions on Greek data and save output predictions in `outputs/D4/adaptation/evaltest/D4_preds.csv`
4. Saves the final f1-score in `results/D4/adaptation/evaltest/D4_scores.out`
