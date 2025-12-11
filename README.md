# $H_{class}$ Meteor Classification Pipeline

This repository contains tools for cleaning Global Meteor Network (GMN) data 
(https://globalmeteornetwork.org/data/traj_summary_data/), applying a trained normalization, 
factor analysis, and Gaussian Mixture Model (GMM) to classify meteor events. The project supports 
both 3-cluster and 11-cluster models and includes utilities for generating summary tables and plots.

## Features

- Cleans and filters raw GMN meteor datasets  
- Normalizes features using stored model coefficients  
- Computes factor-analysis scores  
- Applies pre-trained GMM models for classification  
- Outputs posterior probabilities and hard $H_{class}$ labels  
- Optional visualizations (pie charts, summaries, etc.)  

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>