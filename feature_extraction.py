"""
Feature extraction utilities for voice gender detection.

This module provides functions to convert raw audio files into numerical
representations (Mel‑Frequency Cepstral Coefficients, or MFCCs) and
to load entire datasets of labelled audio files into numpy arrays.

Key features:

* Uses ``librosa`` to load audio and compute MFCCs.
* Computes the mean MFCC over time for a fixed‑length representation.
* Supports parallel feature extraction using ``concurrent.futures``.
* Provides a configurable number of MFCC coefficients and worker threads.

Example
-------

>>> from feature_extraction import load_dataset
>>> X, y = load_dataset("/path/to/dataset", num_mfcc=20, n_jobs=8)

"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple, Sequence

import librosa
import numpy as np

def extract_mfcc(audio_file: str, num_mfcc: int = 13) -> np.ndarray:
    """Extract MFCC features from an audio file.

    Parameters
    ----------
    audio_file : str
        Path to the audio file (WAV format recommended).
    num_mfcc : int, optional
        Number of MFCC coefficients to compute, by default 13.

    Returns
    -------
    np.ndarray
        A 1‑D numpy array containing the mean of each MFCC coefficient
        across time. The length of the array is ``num_mfcc``.
    """
    # Load the entire audio file. ``sr=None`` preserves the original sampling rate.
    y, sr = librosa.load(audio_file, sr=None)
    # Compute MFCCs.  The result is a 2‑D array with shape (n_mfcc, n_frames).
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    # Take the mean across time frames to obtain a fixed‑length vector.
    return np.mean(mfccs, axis=1)


def _process_file(task: Tuple[str, int, int]) -> Tuple[np.ndarray, int]:
    """Helper function to process a single audio file.

    Parameters
    ----------
    task : tuple
        A tuple containing the file path, the numeric label and the number
        of MFCC coefficients.

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple of (feature_vector, label).
    """
    file_path, label, num_mfcc = task
    features = extract_mfcc(file_path, num_mfcc=num_mfcc)
    return features, label


def load_dataset(
    dataset_path: str,
    genders: Sequence[str] = ("male", "female"),
    num_mfcc: int = 13,
    n_jobs: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset of labelled audio files and extract MFCC features.

    The dataset directory is expected to contain subfolders for each
    gender (e.g. ``male`` and ``female``). Each subfolder should contain
    ``.wav`` files for that class.  Feature extraction is performed in
    parallel using a thread pool.

    Parameters
    ----------
    dataset_path : str
        Root directory containing subfolders for each class/gender.
    genders : Sequence[str], optional
        Names of subfolders corresponding to class labels. The index of
        each name in this sequence determines the integer label, e.g.
        0 for ``male`` and 1 for ``female``.
    num_mfcc : int, optional
        Number of MFCC coefficients to compute for each audio file.
    n_jobs : int, optional
        Number of worker threads to use for parallel extraction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two numpy arrays: ``X`` containing feature vectors of shape
        (n_samples, num_mfcc) and ``y`` containing integer labels of shape
        (n_samples,).
    """
    tasks: list[Tuple[str, int, int]] = []
    for label, gender in enumerate(genders):
        folder_path = os.path.join(dataset_path, gender)
        if not os.path.isdir(folder_path):
            # Skip missing folders
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".wav"):
                file_path = os.path.join(folder_path, fname)
                tasks.append((file_path, label, num_mfcc))

    X: list[np.ndarray] = []
    y: list[int] = []
    # Use ThreadPoolExecutor to parallelize feature extraction.  CPU‑bound tasks
    # may also benefit from ProcessPoolExecutor, but thread pool avoids
    # heavy inter‑process communication when loading audio files.
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for features, label in executor.map(_process_file, tasks):
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)
