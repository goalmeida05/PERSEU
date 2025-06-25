# PerseuCPP: A Machine Learning Strategy to Predict Cell-Penetrating Peptides and Their Uptake Efficiency

**PerseuCPP** is a machine learning pipeline developed to predict whether a given peptide sequence has cell-penetrating properties and estimate its uptake efficiency.  
This repository contains all code, datasets, models, and outputs necessary to reproduce the training process, make new predictions, and evaluate model performance.

---

## Repository Structure


├── DATASETS/              # Datasets used for training and testing

├── PERFORMANCE/           # Performance metrics for each dataset

├── RESULTS/               # Prediction results on each dataset

├── PERSEU_MODEL.pkl       # Trained model for CPP classification

├── PERSEU-EFFICIENCY.pkl  # (Optional) Efficiency prediction model (currently not fully integrated)

├── PerseuCPP.py           # Main script: feature extraction, model training, and prediction

├── aminos.json            # Amino acid group and ID mappings

├── utils.py               # Helper functions for feature extraction

├── wrong predicted sequences.csv

├── tests-results.csv

├── cpp-test.fasta         # Example file for testing

└── README.md              # This file




---

## Dependencies

To run this project, you need to have **Python 3.8 or higher** installed.

We recommend creating a virtual environment to avoid conflicts.  
You can set up your environment by running:

```bash
python -m venv perseu-env
source perseu-env/bin/activate  # On Windows use: perseu-env\Scripts\activate
```
```bash
pip install -r requirements.txt
```
## How to Run PERSEUcpp

### 1. Predicting New Sequences

Execute the main script:

```bash
python PerseuCPP.py
```

You will see the following menu:
      1 - TRAINING MODEL

      2 - TESTING MODEL

Choose option 2 - TESTING MODEL if you want to classify new sequences using the pretrained model (PERSEU_MODEL.pkl included in this repository).

You will be asked to enter the file path to your dataset.

Accepted input formats:

* FASTA: Standard FASTA format.
* CSV: CSV file containing a single column with peptide sequences.

Example:
Data path: cpp-test.fasta

The prediction results will be saved into the RESULTS/ directory as results-cpp-mlcpp.csv, containing:
* seq: the original peptide sequence
* prob: probability score of being a CPP
* Classification: predicted class (1 = CPP, 0 = non-CPP)


### 2. Training a New Model (Optional)

If you wish to retrain the model using your own datasets, select option 1 - TRAINING MODEL after executing PerseuCPP.py.

You will be prompted to provide:
* The path to your positive dataset (sequences labeled as CPPs)
* The path to your negative dataset (sequences labeled as non-CPPs)

```bash
Positives path: DATASETS/positives.fasta
Negatives path: DATASETS/negatives.fasta
```
The pipeline will:

* Extract features
* Generate the training matrix
* Perform repeated 10x10 cross-validation
* Display and save performance metrics
* Save the new trained model as model.pkl


## Notes on Feature Extraction
Several handcrafted features are calculated, including:

* Amino acid composition
* Dipeptide and tripeptide frequencies
* Physicochemical properties (molecular weight, isoelectric point, net charge, hydropathy)
* CKSAAGP features (K-spaced amino acid group pairs with k=1)
* Atomic Composition 

Sequences containing ambiguous amino acids (X, B, Z, J, O, U, *, -) are automatically filtered.
