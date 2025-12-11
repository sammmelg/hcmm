# $H_{class}$ Meteor Model: Meteor Classification Pipeline

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Last Commit](https://img.shields.io/github/last-commit/sammmelg/hcmm)
![Issues](https://img.shields.io/github/issues/sammmelg/hcmm)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains tools for cleaning Global Meteor Network (GMN) data 
(https://globalmeteornetwork.org/data/traj_summary_data/), applying a trained normalization, 
factor analysis, and Gaussian Mixture Model (GMM) to classify meteor events. The project currently supports 
the 3-cluster and 11-cluster models and includes utilities for generating summary tables and plots.

## Features

- Cleans and filters raw GMN meteor data  
- Normalizes features using stored model coefficients  
- Computes factor-analysis scores  
- Analytically applies a pre-trained GMM to determine $H_{class}$  
- Outputs posterior probabilities and hard $H_{class}$ labels  
- Optional summaries and pie chart

## Installation

Clone the repository:

```bash
git clone https://github.com/sammmelg/hcmm.git
cd hcmm
```

## Useage
### gmnDataConverter.py
This function will take a raw text file downloaded from the GMN data 
website and create 2 $.csv$ files:

- A cleaned version of the raw data with easily read column headers. 
This raw data can be used for further analysis with model classified meteors.
- A model data file that contains computed values ($E_\mathrm{beg}$ and $\rho_\mathrm{beg}$)
using Western Meteor Py Library (wmpl; Vida et al. 2019) along with the other 11 features used 
in the model. 

```bash
python gmnDataConverter.py \
    -path /path/to/raw_gmn_file.txt \
    -savefile cleaned_output.csv
```

### Classifier.py
This script reads in the model data and raw data and applies the GMM model to determine 
$H_{class}$ assignment for each meteor. Outputs include: a classification summary that 
breaks down events by shower along with the number/percentage of events assigned to 
each $H_{class}$ (classification_summary.csv), a summary of each event's classification (event_summary.csv),
and a pie chart of the distribution of events in each $H_{class}$ (classification_distribution.jpg).

```bash
python Classifier.py \
    -n_classes 3 \
    -modeldata path/to/model_data.csv \
    -rawdata path/to/raw_data.csv \
    -threshold 0.5 \
    -save_output \
    -save_path results/
```