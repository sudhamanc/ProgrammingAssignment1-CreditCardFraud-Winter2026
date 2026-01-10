# Programming Assignment 1: Credit Card Fraud - Winter 2026

## CS 614 - Applications of Machine Learning

This repository contains the implementation of a binary classification neural network for credit card fraud detection using PyTorch.

## Dataset

Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Note**: The dataset is not included in this repository. Download it from the link above and place it in a `datasets/` folder.

## Project Structure

- `HW1.ipynb` - Main Jupyter notebook with complete implementation
- `.gitignore` - Git ignore file for Python/Jupyter projects
- `README.md` - This file

## Implementation Overview

### 1. Data Preprocessing
- Z-score standardization using training set statistics
- Train/test split (67%/33%)

### 2. Neural Network Architecture
- Input layer: 30 features
- Hidden layer: 16 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation
- Loss function: Binary Cross Entropy
- Optimizer: Adagrad

### 3. Evaluation Metrics
- Accuracy
- Precision (for fraud detection)
- Recall (for fraud detection)

## Results

### Class Distribution
- Legitimate transactions: ~99.82%
- Fraudulent transactions: ~0.18%
- Highly imbalanced dataset

### Model Performance
- Training Accuracy: ~99.92%
- Testing Accuracy: ~99.92%
- Training Precision (Fraud): ~88.19%
- Training Recall (Fraud): ~62.20%
- Testing Precision (Fraud): ~82.64%
- Testing Recall (Fraud): ~64.10%

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib scikit-learn torch
```

3. Download the dataset and place it in `datasets/creditcard.csv`

4. Run the Jupyter notebook:
```bash
jupyter notebook HW1.ipynb
```

## Key Findings

- The model achieves high accuracy but moderate recall for fraud detection
- Precision is relatively high (~83-88%), meaning predictions of fraud are reliable
- Recall is moderate (~62-64%), meaning the model misses about 36-38% of actual fraud cases
- The class imbalance makes accuracy less meaningful than precision/recall
- For production fraud detection, higher recall would be prioritized

## Technologies Used

- Python 3.13
- PyTorch 2.9.1
- NumPy 2.4.0
- Pandas 2.3.3
- Matplotlib 3.10.8
- Scikit-learn 1.8.0

## Author

CS 614 Student - Winter 2026
