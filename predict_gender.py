"""
Inference script for the voice gender classifier.

Given a trained model and a new audio file, this script computes MFCC
features and predicts the gender using the saved model.  The script
returns a human‑readable label rather than printing raw feature vectors.

Example
-------

>>> python predict_gender.py path/to/audio.wav svm_model.pkl

This will print either ``Male`` or ``Female``.
"""

from __future__ import annotations

import argparse
from typing import Optional

import joblib

from feature_extraction import extract_mfcc


def predict_gender(audio_file: str, model_path: str, num_mfcc: int = 13) -> str:
    """Predict the gender of a speaker from an audio file.

    Parameters
    ----------
    audio_file : str
        Path to the audio file to classify.
    model_path : str
        Path to a saved scikit‑learn model (pickled via joblib).
    num_mfcc : int, optional
        Number of MFCC coefficients to compute (should match the training
        configuration).

    Returns
    -------
    str
        "Female" if the predicted label is 1, otherwise "Male".
    """
    # Load the trained model.
    model = joblib.load(model_path)
    # Extract MFCC features from the new audio file.
    mfcc_features = extract_mfcc(audio_file, num_mfcc=num_mfcc)
    # The model expects a 2‑D array, so wrap the vector in a list.
    label = model.predict([mfcc_features])[0]
    return "Female" if label == 1 else "Male"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict the gender from a WAV file using a trained model.")
    parser.add_argument(
        "audio_file",
        help="Path to the WAV file to classify.",
    )
    parser.add_argument(
        "model_path",
        help="Path to the saved classifier (joblib .pkl file).",
    )
    parser.add_argument(
        "--num_mfcc",
        type=int,
        default=13,
        help="Number of MFCC coefficients to compute (default: 13).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    prediction = predict_gender(args.audio_file, args.model_path, num_mfcc=args.num_mfcc)
    print(prediction)


if __name__ == "__main__":
    main()
