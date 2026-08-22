#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Disclaimer: This code is provided solely for the purpose of reproducing
# results as described in the associated work. The authors and contributors
# are not responsible for any consequences arising from the use of this code
# or its outputs beyond this intended purpose. Use at your own risk.

"""Plot clean and noisy AF test signals using the existing dataloaders.

This script calls:

    create_dataloaders_AF()
    create_dataloaders_AF_noise_aug()

The plotted arrays are retrieved from each test dataset through
CustomDataset.__getitem__(). They therefore receive the same normalisation as
the tensors supplied to the model during evaluation.
"""

import os
import sys
from unittest.mock import patch

# Select a non-interactive backend before importing pyplot. This allows the
# script to run on an HPC node without a graphical display.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from prepare_dataset import (
    create_dataloaders_AF,
    create_dataloaders_AF_noise_aug,
)


def get_clean_test_dataloader(data_dir, batch_size=64):
    """Return the clean test dataloader from create_dataloaders_AF()."""
    # get_args_AF() is called inside create_dataloaders_AF(). Temporarily
    # provide only the arguments needed by that parser.
    loader_arguments = [
        sys.argv[0],
        "--data_dir",
        data_dir,
        "--batch-size",
        str(batch_size),
    ]

    with patch.object(sys, "argv", loader_arguments):
        (
            _train_dataloader,
            _val_dataloader,
            test_dataloader,
            _calib_dataloader,
        ) = create_dataloaders_AF()

    return test_dataloader


def get_noisy_test_dataloader(data_dir, noise_level, batch_size=64):
    """Return a noisy test dataloader for one specified noise level."""
    # The existing AF parser defines --data_dir with an underscore and
    # --batch-size with a hyphen. The --noise value is consumed inside
    # create_dataloaders_AF_noise_aug().
    loader_arguments = [
        sys.argv[0],
        "--data_dir",
        data_dir,
        "--batch-size",
        str(batch_size),
        "--noise",
        str(noise_level),
        "--addnoise",
    ]

    with patch.object(sys, "argv", loader_arguments):
        (
            _train_dataloader,
            _val_dataloader,
            test_dataloader,
            _calib_dataloader,
        ) = create_dataloaders_AF_noise_aug()

    return test_dataloader


def signal_to_numpy(signal):
    """Convert a dataset signal to a one-dimensional NumPy array."""
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()

    signal = np.asarray(signal).squeeze()

    if signal.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional signal after squeezing, "
            f"but received shape {signal.shape}."
        )

    return signal


def label_to_value(label):
    """Convert a tensor or NumPy label to a scalar where possible."""
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    label = np.asarray(label).squeeze()

    if label.size == 1:
        return label.item()

    return label


def plot_clean_and_noisy_af_signals(
    data_dir,
    sample_index=0,
    noise_levels=(0.1, 0.3, 0.6),
    sampling_frequency_hz=32.0,
    batch_size=64,
    save_path="./clean_and_noisy_af_signals.png",
):
    """Plot a clean AF test signal on top of noisy evaluation inputs.

    The clean test tensor is obtained from create_dataloaders_AF(). For every
    requested noise level, the noisy tensor is obtained from
    create_dataloaders_AF_noise_aug(). Accessing dataset[sample_index] invokes
    the original CustomDataset.__getitem__() normalisation.

    Parameters
    ----------
    data_dir : str
        Directory containing metadata.csv and signals.npy.
    sample_index : int, default=0
        Position within the AF test dataset.
    noise_levels : sequence of float
        Values passed to the existing --noise parser argument.
    sampling_frequency_hz : float, default=32.0
        DeepBeat PPG sampling frequency in hertz, used to convert samples
        to seconds.
    batch_size : int, default=64
        Batch size passed to the existing dataloader functions.
    save_path : str or None
        Figure output path. If None, the image is not saved.

    Returns
    -------
    fig, axes, plotted_signals
        Matplotlib objects and the plotted signal arrays.
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"AF data directory does not exist: {data_dir}")

    metadata_path = os.path.join(data_dir, "metadata.csv")
    signals_path = os.path.join(data_dir, "signals.npy")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.csv was not found at: {metadata_path}")

    if not os.path.isfile(signals_path):
        raise FileNotFoundError(f"signals.npy was not found at: {signals_path}")

    noise_levels = tuple(float(level) for level in noise_levels)

    if not noise_levels:
        raise ValueError("noise_levels must contain at least one value.")

    if any(level < 0 for level in noise_levels):
        raise ValueError("All noise levels must be non-negative.")

    sampling_frequency_hz = float(sampling_frequency_hz)
    if sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be greater than zero.")

    print("\nCreating the clean test dataloader...")
    print("Calling create_dataloaders_AF()")

    clean_test_dataloader = get_clean_test_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
    )
    clean_test_dataset = clean_test_dataloader.dataset

    if not 0 <= sample_index < len(clean_test_dataset):
        raise IndexError(
            f"sample_index must be between 0 and {len(clean_test_dataset) - 1}, "
            f"but received {sample_index}."
        )

    # This invokes the existing CustomDataset.__getitem__() and therefore
    # returns the signal after evaluation-time normalisation.
    clean_tensor, clean_label = clean_test_dataset[sample_index]
    clean_signal = signal_to_numpy(clean_tensor)
    clean_label = label_to_value(clean_label)

    print(f"Clean test dataset size: {len(clean_test_dataset)}")
    print(f"Selected test-set position: {sample_index}")
    print(f"Selected label: {clean_label}")
    print(f"Clean tensor shape: {tuple(clean_tensor.shape)}")

    noisy_signals = {}
    noisy_labels = {}

    for noise_level in noise_levels:
        print(f"\nCreating noisy test dataloader for noise {noise_level:g}...")
        print(
            "Calling create_dataloaders_AF_noise_aug() "
            f"with --noise {noise_level:g}"
        )

        noisy_test_dataloader = get_noisy_test_dataloader(
            data_dir=data_dir,
            noise_level=noise_level,
            batch_size=batch_size,
        )
        noisy_test_dataset = noisy_test_dataloader.dataset

        if len(noisy_test_dataset) != len(clean_test_dataset):
            raise ValueError(
                "Clean and noisy test datasets have different lengths: "
                f"{len(clean_test_dataset)} and {len(noisy_test_dataset)}."
            )

        # This invokes the noisy dataset's original __getitem__ method, after
        # raw noise was added by create_dataloaders_AF_noise_aug().
        noisy_tensor, noisy_label = noisy_test_dataset[sample_index]
        noisy_signal = signal_to_numpy(noisy_tensor)
        noisy_label = label_to_value(noisy_label)

        if noisy_signal.shape != clean_signal.shape:
            raise ValueError(
                "Clean and noisy signal shapes differ: "
                f"{clean_signal.shape} and {noisy_signal.shape}."
            )

        if not np.array_equal(np.asarray(clean_label), np.asarray(noisy_label)):
            raise ValueError(
                "Clean and noisy labels differ: "
                f"{clean_label} and {noisy_label}."
            )

        noisy_signals[noise_level] = noisy_signal
        noisy_labels[noise_level] = noisy_label

        print(f"Noisy tensor shape: {tuple(noisy_tensor.shape)}")
        print(f"Noisy label: {noisy_label}")

    number_of_rows = len(noise_levels)
    fig, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=1,
        figsize=(10, 3.5 * number_of_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # DeepBeat PPG signals are sampled at 32 Hz. Dividing each sample
    # position by 32 converts the horizontal axis to elapsed seconds.
    time_seconds = np.arange(clean_signal.size) / sampling_frequency_hz

    for row, noise_level in enumerate(noise_levels):
        axis = axes[row, 0]

        # Draw the noisy signal first so it remains underneath.
        axis.plot(
            time_seconds,
            noisy_signals[noise_level],
            color="tab:red",
            linewidth=1.0,
            alpha=0.65,
            label=f"Noisy, σ = {noise_level:g}",
            zorder=1,
        )

        # Draw the clean signal second so it appears visually on top.
        axis.plot(
            time_seconds,
            clean_signal,
            color="black",
            linewidth=1.5,
            alpha=1.0,
            label="Clean",
            zorder=2,
        )

        axis.set_title(f"Noise intensity: σ = {noise_level:g}")
        axis.set_ylabel("Normalised amplitude")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    axes[-1, 0].set_xlabel("Time (seconds)")
    fig.suptitle(
        "PPG signal augmented with different noise intensities",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        save_path = os.path.abspath(save_path)
        save_directory = os.path.dirname(save_path)

        if save_directory:
            os.makedirs(save_directory, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    plotted_signals = {
        "clean": clean_signal,
        "noisy": noisy_signals,
        "noise_levels": noise_levels,
        "time_seconds": time_seconds,
        "sampling_frequency_hz": sampling_frequency_hz,
        "test_sample_index": sample_index,
        "clean_label": clean_label,
        "noisy_labels": noisy_labels,
    }

    plt.close(fig)
    return fig, axes, plotted_signals


def main():
    """Run the plot automatically when this file is executed."""
    # Directory containing metadata.csv and signals.npy.
    data_dir = "<path>"

    # Position in the AF test dataset, not the original metadata row number.
    sample_index = 200

    noise_levels = (0.1, 0.3, 0.6)
    sampling_frequency_hz = 32.0
    batch_size = 64

    # Save the image beside this Python script.
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        script_directory,
        f"clean_and_noisy_af_signals_sample_{sample_index}.png",
    )

    print("=" * 70)
    print("AF clean and noisy signal plotting")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Selected test sample: {sample_index}")
    print(f"Noise levels: {noise_levels}")
    print(f"Sampling frequency: {sampling_frequency_hz:g} Hz")
    print(f"Output path: {output_path}")

    _fig, _axes, plotted_signals = plot_clean_and_noisy_af_signals(
        data_dir=data_dir,
        sample_index=sample_index,
        noise_levels=noise_levels,
        sampling_frequency_hz=sampling_frequency_hz,
        batch_size=batch_size,
        save_path=output_path,
    )

    print("\n" + "=" * 70)
    print("Plotting completed")
    print("=" * 70)
    print(f"Test-set position: {plotted_signals['test_sample_index']}")
    print(f"Ground-truth label: {plotted_signals['clean_label']}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
