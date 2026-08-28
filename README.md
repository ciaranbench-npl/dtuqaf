# From Confidence Scores to Decision Consequences: A Decision-Theoretic Approach to Communicating Uncertainty for Atrial-Fibrillation Detection

## Overview

This repository contains a research workflow for training and evaluating a binary time-series classifier, recalibrating its predicted probabilities using Venn-Abers predictors, and assessing prediction reliability under simulated dataset shift.

The workflow supports:

- training a convolutional binary classifier on unmodified time-series data;
- evaluating the trained model on clean and noise-augmented test data;
- generating held-out calibration predictions;
- constructing Venn-Abers probability intervals;
- comparing probability-based and expected-cost-based rejection strategies;
- calculating calibration, and cost-coverage measures; and
- producing figures and summary statistics for reproducibility studies.

## Dataset
The dataset and preprocessing procedures used in this study are described by Moulaeifard et al.: 
```@article{moulaeifard2025machine, title={Machine-learning for photoplethysmography analysis: Benchmarking feature, image, and signal-based approaches}, author={Moulaeifard, Mohammad and Coquelin, Loic and Rinkevi{\v{c}}ius, Mantas and Solo{\v{s}}enko, Andrius and Pfeffer, Oskar and Bench, Ciaran and Hegemann, Nando and Vardanega, Sara and Nandi, Manasi and Alastruey, Jordi and others}, journal={arXiv preprint arXiv:2502.19949}, year={2025} }```


Other code is adapted from source code released under the Apache License 2.0 and associated with the publication by Bench et al. below. The Apache License 2.0 permits use, modification, and distribution subject to its terms, including preservation of the applicable copyright, licence, and attribution notices. The adapted code may contain modifications made for the present analysis and should not be treated as an exact reproduction of the original implementation.
```
@article{bench2025uncertainty,
  title={Uncertainty quantification with approximate variational learning for wearable photoplethysmography prediction tasks},
  author={Bench, Ciaran and Desai, Vivek and Moulaeifard, Mohammad and Strodthoff, Nils and Aston, Philip and Thompson, Andrew},
  journal={Machine Learning: Health},
  volume={1},
  number={1},
  pages={015013},
  year={2025},
  publisher={IOP Publishing},
  doi={10.1088/3049-477X/ae0b74}
}
```

## Method Summary

### 1. Model training

The training workflow:

1. parses command-line settings;
2. loads metadata and time-series signals;
3. creates training, validation, calibration, and test splits;
4. normalises the input data and prepares PyTorch DataLoaders;
5. initialises the binary classification model, optimiser, scheduler, and loss function;
6. trains and validates the model for the requested number of epochs; and
7. saves model checkpoints, hyperparameters, and training logs.


### 2. Model evaluation

The evaluation workflow:

1. loads saved hyperparameters and a model checkpoint;
2. selects calibration, clean-test, or noise-augmented test data;
3. reconstructs the trained model;
4. runs inference; and
5. saves probabilities, labels, and any configured uncertainty outputs.

### 3. Venn-Abers and decision-theoretic analysis

Held-out calibration probabilities and labels are used to construct a Venn-Abers interval for each test prediction. The analysis then compares:

- a naive rejection rule based on probability magnitude or distance from the decision threshold; and
- a cost-aware rejection rule based on expected decision cost.

The analysis can be repeated for clean and shifted test data, different decision thresholds, alternative cost assumptions, and different retained-data fractions.


## Requirements

The code is designed for a Linux-based Python environment with optional GPU acceleration.

Use the dependency versions supplied with the repository where available.

## Environment Setup
See requirements.txt

## Data Preparation

The expected dataset contains time-series signals and corresponding binary labels. A typical input directory contains:

```text
<DATA_DIR>/
├── metadata.csv
└── signals.npy
```

The data-loading code creates four non-overlapping subsets:

- training;
- validation;
- calibration; and
- test.

The calibration subset must remain separate from the training, validation, and test subsets because it is used to fit the Venn-Abers recalibration procedure.

### Data privacy

Do not commit raw data, participant identifiers, access tokens, credentials, internal paths, or restricted metadata to the repository. Use access-controlled storage and replace local paths with placeholders in scripts, notebooks, logs, and examples.

## Configuration

Arguments are defined in `argparser.py`. Relevant options include:

```text
--data_dir          Path to the dataset directory
--checkpoint-path   Path to a trained model checkpoint
--calib             Evaluate the held-out calibration subset
--addnoise          Add Gaussian noise during test evaluation
--noise             Standard deviation of the Gaussian noise
```


## Running the Workflow

The commands below use placeholders intentionally. Adjust script names only if they differ in your checkout.

### 1. Train the classifier

```bash
python AF_model/train_sgd_mcd_af.py \
  --data_dir <DATA_DIR> \
  <ADDITIONAL_TRAINING_ARGUMENTS>
```

Expected outputs include:

```text
checkpoints_<configuration>/
├── checkpoint_epoch_<N>.pth
├── hparams.npy
└── training_log.txt
```

### 2. Generate calibration predictions

```bash
python AF_model/eval_sgd_mcd_af_no_sample.py \
  --data_dir <DATA_DIR> \
  --checkpoint-path <CHECKPOINT_PATH> \
  --calib
```

Move or write the generated files to:

```text
calib_preds/
```

### 3. Evaluate the clean test set

```bash
python AF_model/eval_sgd_mcd_af_no_sample.py \
  --data_dir <DATA_DIR> \
  --checkpoint-path <CHECKPOINT_PATH>
```

Store the generated files in:

```text
test_preds/
```

### 4. Evaluate the noise-augmented test set

```bash
python AF_model/eval_sgd_mcd_af_no_sample.py \
  --data_dir <DATA_DIR> \
  --checkpoint-path <CHECKPOINT_PATH> \
  --addnoise \
  --noise <NOISE_STANDARD_DEVIATION>
```

Store the generated files in:

```text
noise_preds/
```

### 5. Run the analysis notebook

Start Jupyter:

```bash
jupyter lab
```

Open the required notebook under `analysis/`, then set its input paths to the calibration, clean-test, and noise-augmented prediction directories.

For full reproducibility, restart the kernel and run all cells from top to bottom.

## Inputs and Outputs

### Training inputs

- `metadata.csv` containing labels and required metadata;
- `signals.npy` containing the time-series signals;
- command-line model and training settings.

### Evaluation inputs

- dataset directory;
- trained model checkpoint;
- saved hyperparameters;
- calibration/test selection;
- optional noise settings.

### Model outputs

Evaluation will generate NumPy files such as:

```text
preds_mcd_af.npy
 gts_mcd_af.npy
entropy_corrupted_mcd_af.npy
H2_mcd_af.npy
```

Only outputs used by the current analysis should be retained and documented. Remove obsolete uncertainty outputs if they are not part of the reproducible workflow.

### Analysis outputs

The notebooks may produce:

- Venn-Abers probability intervals;
- collapsed calibrated probabilities;
- calibration and reliability statistics;
- reliability diagrams;
- retained and rejected confusion counts;
- cost-coverage curves;
- area under the cost-coverage curve; and
- saved figure files.




