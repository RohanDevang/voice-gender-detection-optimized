# Voice Gender Detection Optimized

## Overview

This repository contains optimized code for training and using a support‑vector machine (SVM) classifier to detect gender (male/female) from voice audio files. It includes utilities for converting audio file formats, extracting features, training and evaluating a model, and performing predictions on new audio samples.

## Files

- **feature_extraction.py** – Provides functions to extract Mel‑Frequency Cepstral Coefficient (MFCC) features from audio files and load entire datasets. It uses `librosa` to load audio, computes MFCCs, and returns the mean across frames. The `load_dataset` function parallelizes feature extraction and returns features and labels.

- **convert_audio.py** – A utility script that batch converts `.m4a` files in a specified directory to `.wav` using `pydub`. It creates the output directory if necessary and skips files that already have a converted `.wav`.

- **train_model.py** – A training script that loads the dataset, performs a stratified train/test split, constructs a pipeline with `StandardScaler` and an SVM classifier (`SVC`), uses `GridSearchCV` to tune hyper‑parameters, evaluates the best model on the test set, prints metrics (accuracy, precision, recall, F1 score), plots the confusion matrix, and saves the trained model to a file.

- **predict_gender.py** – A lightweight inference script that loads a saved SVM model and uses MFCC features extracted from a new audio file to predict the speaker’s gender. It prints “Male” or “Female” accordingly.

## Usage

1. **Convert M4A to WAV** (if needed):
   ```bash
   python convert_audio.py /path/to/m4a/files /path/to/output/wav
   ```

2. **Train a model**:
   ```bash
   python train_model.py /path/to/dataset --num_mfcc 20 --model_out svm_model.pkl
   ```

3. **Predict gender for a new audio file**:
   ```bash
   python predict_gender.py /path/to/audio.wav svm_model.pkl
   ```

## Dataset structure

The dataset directory should have subdirectories for each gender label (e.g., `male` and `female`), each containing `.wav` files for that class.
