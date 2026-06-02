# Energy-Aware Deep Learning for Network Intrusion Detection

This repository contains the implementation used in a thesis project on the trade-off between detection performance and energy consumption in deep learning models for network intrusion detection.

## Models

The following models are implemented:

- Feedforward Neural Network (FFNN)
- Long Short-Term Memory network (LSTM)
- One-Dimensional Convolutional Neural Network (1D-CNN)

## Dataset

The experiments use the NSL-KDD dataset.

Download the following files and place them in the `data/` folder:

- KDDTrain+.txt
- KDDTest+.txt

Expected folder structure:

data/
  KDDTrain+.txt
  KDDTest+.txt

## Setup

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

## Run experiments

python main.py

The main thesis results were reported using seed 42. Additional runs were conducted using seeds 1 and 123 to assess result stability.

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Energy consumption
- CO2 emissions

Energy consumption and CO2 emissions are measured using CodeCarbon.

## Notes

The implemented models are lightweight model prototypes and are not intended to represent complete deployable NIDS systems. The purpose of the repository is to support reproducibility of the experimental evaluation.