# Extended Stochastic Block Models for Recommender Systems

Code for my Matser's thesis adapting **Extended Stochastic Block Model** (Legramanti et al. 2020) to recommender systems. 

---
## Project Description

This repository contains the code and analysis for my Master’s thesis, which explores **Extended Stochastic Block Models (ESBMs)** and their application to **recommender systems**. A stochastic block model (SBM) is a generative statistical model for networks, often used to cluster nodes into communities based on their connection patterns. Extended Stochastic Block Models (ESBMs) are a Bayesian nonparametric version of SBMs.

The project extends this version of SBMs to **weighted, bipartite networks** and introduces a **degree-corrected extension** to capture user/item popularity heterogeneity while leveraging nonparametric Bayesian modeling for community detection.

The codebase serves two main purposes:
1. **Simulation Studies:** Scripts to reproduce the synthetic experiments in the thesis, demonstrating the performance of ESBMs under different network settings.
2. **Analysis of real-world dataset from Goodreads:** An empirical study applying ESBM and its degree-corrected variant to a large-scale **Goodreads dataset** of book ratings.


### Data
The dataset is not stored in this repository due to its size. However:
- A **script is provided** to preprocess and build the dataset from the raw files.
- The raw Goodreads ratings data can be downloaded from [this page](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (called 'goodreads_interactions_dedup.json').

### So... why does this matter? 
Recommendation algorithms decide what we watch, read, or buy but they are often black boxes. The goal of this project is to an interpretable algorithm modeling user-item interactions as networks. Instead of blindly factorizing a ratings matrix, we discover communities of similar users and items, making recommendations easier to interpret and explain. Plus, with degree correction, we can adjust for super popular items or very active users making recommendations more accurate.

It is also worth pointing out that this thesis was motivated by an application to recommender systems but the methods we derive can be applied to any other type of weighted bipartite graph (e.g. flower-pollinators networks or citation networks)

---

## 📂 Repository Structure

```text
├── data/               # Raw and processed datasets
├── results/            # Outputs from analysis
│   ├── figures/        # Plots and visualizations
│   ├── tables/         # Tabular results
│   └── text/           # PDF version of thesis
├── src/                # Source code
│   ├── analysis/       # Modeling and statistical methods
│   │   ├── models/     # Model definitions
│   │   └── utilities/  # Helper functions
│   └── pipeline/       # Data loading and preprocessing
├── tests/              # Unit and integration tests
└── README.md           # this file :)
```
---

## Installation
Clone the repository and install requirements:

(pip)
```bash
git clone https://github.com/lorenzo-costa/DCESBM.git
cd DCESBM
pip install -r requirements.txt
```

(conda)
```bash
git clone https://github.com/lorenzo-costa/DCESBM.git
cd DCESBM
conda env create -f environment.yml
conda activate DCESBM
```

---

## Usage
An example of how the model works can be found in the notebook `example.ipynb'.

To run the simulations and data analysis:
1. Preprocess the data: place Goodreads dataset in data/raw and run
```bash
python src/pipeline/pre_processing_functs.py
``` 
Note:
- this is not needed for simulations
- requires internet connection
- may be quite slow

Alternatively you can dowload the data using
```bash
gdown 'https://drive.google.com/uc?id=1h7UdSq9_WEj9k7HD-pOOeXtro5xXmKXZ' -O data/processed/dataset_clean.csv
```

2. Run simulations. This populates the folder results with figures and tables from simulations
```bash
python src/analysis/simulations.py
```
3. Run data analysis. This populates the folder results with figures and tables from 
analysis of Goodreads dataset
```bash
python src/analysis/book_analysis.py
```
Note: this takes A LOT of time


To run the whole analysis with one command run:
```bash
make all
```

or to skip the processing step run
```bash
make no_process
```

or to run a smaller version of the analysis
```bash
make small
```
