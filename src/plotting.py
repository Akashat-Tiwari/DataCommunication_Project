
"""Waveform plotting utilities for the Data Communication assignment."""

from pathlib import Path
import matplotlib

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _save_decoder_image(signal_levels: Sequence[int]) -> Path:
    """Save a clean waveform image for image-based decoding."""
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "decoder_signal.png"

    # Keep exactly 20 pixels for every signal interval.
    pixels_per_level = 20
    width_pixels = len(signal_levels) * pixels_per_level
    height_pixels = 300
    dpi = 100

    fig = plt.figure(
        figsize=(width_pixels / dpi, height_pixels / dpi),
        dpi=dpi,
        facecolor="white",
    )
    ax = fig.add_axes([0, 0, 1, 1])

    ax.stairs(
        signal_levels,
        list(range(len(signal_levels) + 1)),
        baseline=None,
        color="#ff0000",
        linewidth=3,
    )

    # Fixed limits let the decoder infer -1, 0, and +1 from pixel positions.
    ax.set_xlim(0, len(signal_levels))
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        pad_inches=0,
    )
    plt.close(fig)

    return output_path


def _show_or_save(fig: Figure, filename: str, show: bool) -> None:
    """Save the plot as an image and optionally display it."""
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    backend = matplotlib.get_backend().lower()

    if show and backend != "agg":
        plt.show()
    elif show:
        print(f"Plot saved to: {output_path}")

def _validate_bits(bits: Sequence[int]) -> None:
    """Validate a binary sequence."""
    if not bits:
        raise ValueError("bits must not be empty")

    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")


def _validate_signal_levels(levels: Sequence[int]) -> None:
    """Validate ternary line-signal levels."""
    if not levels:
        raise ValueError("signal levels must not be empty")

    if any(level not in (-1, 0, 1) for level in levels):
        raise ValueError("signal levels must be -1, 0, or 1")


def _validate_samples(
    samples: Sequence[float], name: str
) -> list[float]:
    """Convert samples to finite floating-point values."""
    if not samples:
        raise ValueError(f"{name} must not be empty")

    try:
        values = [float(sample) for sample in samples]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")

    return values


def plot_digital_stream(
    bits: Sequence[int],
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot a binary data stream as a digital waveform."""
    _validate_bits(bits)

    fig, ax = plt.subplots()
    edges = list(range(len(bits) + 1))

    ax.stairs(bits, edges, label="Digital data")
    ax.set_title("Generated Digital Data Stream")
    ax.set_xlabel("Bit position")
    ax.set_ylabel("Bit value")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlim(0, len(bits))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    _show_or_save(fig, "digital_stream.png", show)

    return fig, ax


def plot_line_coded_signal(
    bits: Sequence[int],
    signal_levels: Sequence[int],
    scheme: str,
    samples_per_bit: int = 1,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a line-coded signal.

    samples_per_bit=1 for NRZ-L, NRZ-I, AMI, B8ZS and HDB3.
    samples_per_bit=2 for Manchester and Differential Manchester.
    """
    _validate_bits(bits)
    _validate_signal_levels(signal_levels)

    if (
        not isinstance(samples_per_bit, int)
        or isinstance(samples_per_bit, bool)
        or samples_per_bit < 1
    ):
        raise ValueError("samples_per_bit must be a positive integer")

    expected_length = len(bits) * samples_per_bit

    if len(signal_levels) != expected_length:
        raise ValueError(
            "signal length must equal len(bits) * samples_per_bit"
        )

    fig, ax = plt.subplots()

    edges = [
        index / samples_per_bit
        for index in range(len(signal_levels) + 1)
    ]

    ax.stairs(signal_levels, edges, label=scheme)

    # Mark each original bit boundary.
    for boundary in range(len(bits) + 1):
        ax.axvline(boundary, linestyle="--", linewidth=0.6, alpha=0.5)

    ax.set_title(f"{scheme} Line-Coded Signal")
    ax.set_xlabel("Bit interval")
    ax.set_ylabel("Signal level")
    ax.set_yticks([-1, 0, 1])
    ax.set_xlim(0, len(bits))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    _save_decoder_image(signal_levels)
    _show_or_save(fig, "line_coded_signal.png", show)

    return fig, ax


def plot_pcm_reconstruction(
    original_samples: Sequence[float],
    reconstructed_samples: Sequence[float],
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot original analog samples against the reconstructed PCM staircase.

    Each reconstructed value is held over one sample interval.
    """
    original = _validate_samples(original_samples, "original_samples")
    reconstructed = _validate_samples(
        reconstructed_samples, "reconstructed_samples"
    )

    if len(original) != len(reconstructed):
        raise ValueError(
            "original and reconstructed samples must have equal lengths"
        )

    fig, ax = plt.subplots()

    # Each staircase level is held for one interval.
    edges = list(range(len(reconstructed) + 1))
    ax.stairs(
        reconstructed,
        edges,
        label="Reconstructed PCM staircase",
    )

    # Plot the original samples at the centers of the intervals.
    positions = [index + 0.5 for index in range(len(original))]
    ax.plot(
        positions,
        original,
        marker="o",
        linestyle="None",
        label="Original analog samples",
    )

    ax.set_title("PCM Signal Reconstruction")
    ax.set_xlabel("Sample interval")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(original))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    _show_or_save(fig, "pcm_reconstruction.png", show)

    return fig, ax
