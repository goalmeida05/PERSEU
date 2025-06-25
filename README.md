# PERSEUcpp
PERSEUcpp: A machine learning strategy to predict cell-penetrating peptides and their uptake efficiency
PERSEUcpp is a machine learning pipeline developed to predict whether a given peptide sequence has cell-penetrating properties and estimate its uptake efficiency.
This repository contains all code, datasets, models, and outputs necessary to reproduce the training process, make new predictions, and evaluate model performance.

# Repository Structure
.
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

Our source code is titled PERSEUcpp.py, and in it, you will find how the all descriptors calculations were done and how the model was trained.

To run PERSEUcpp, you only need to run PERSEUcpp.py and enter the path of the desired file. For example, if you have the file cpps-test.fasta, simply enter its full name and submit it. The CPPs and their respective efficiencies will be predicted.

The model accepts both FASTA and CSV files. For the FASTA format, you must follow the standard FASTA file structure. For the CSV format, the file should contain a single column with only the sequences.
