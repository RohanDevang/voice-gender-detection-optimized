"""
Audio format conversion utility.

This module provides a simple function to convert all ``.m4a`` files in a
directory to the ``.wav`` format using ``pydub``.  It also includes a
command‑line interface for ad‑hoc conversion of folders.

Example
-------

To convert a folder of m4a files into wav files in a new directory:

>>> python convert_audio.py /path/to/m4a/files /path/to/output/wav

Converted files will be written only if the corresponding ``.wav`` does
not already exist.  Existing files are skipped silently.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from pydub import AudioSegment


def convert_m4a_to_wav(input_folder: str, output_folder: str) -> None:
    """Convert all M4A audio files in a directory to WAV format.

    Parameters
    ----------
    input_folder : str
        Directory containing ``.m4a`` files to convert.
    output_folder : str
        Directory where converted ``.wav`` files should be written.  It will
        be created if it does not exist.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if not file_name.lower().endswith(".m4a"):
            continue
        input_path = os.path.join(input_folder, file_name)
        output_filename = os.path.splitext(file_name)[0] + ".wav"
        output_path = os.path.join(output_folder, output_filename)
        # Skip conversion if the output file already exists to save time.
        if os.path.exists(output_path):
            continue
        # Load the audio and export to WAV.
        audio = AudioSegment.from_file(input_path, format="m4a")
        audio.export(output_path, format="wav")
        print(f"Converted: {file_name} -> {output_filename}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch convert M4A files to WAV.")
    parser.add_argument(
        "input_folder",
        help="Path to a folder containing .m4a files to convert.",
    )
    parser.add_argument(
        "output_folder",
        help="Path to the folder where converted .wav files will be stored.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    convert_m4a_to_wav(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
