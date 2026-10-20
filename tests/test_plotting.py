
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.plotting import (
    plot_digital_stream,
    plot_line_coded_signal,
    plot_pcm_reconstruction,
)


def test_plot_digital_stream():
    fig, ax = plot_digital_stream([0, 1, 1, 0], show=False)

    assert ax.get_title() == "Generated Digital Data Stream"
    assert ax.get_ylabel() == "Bit value"
    assert len(ax.patches) == 1

    plt.close(fig)


def test_plot_nrz_l_signal():
    fig, ax = plot_line_coded_signal(
        [0, 1, 1, 0],
        [1, -1, -1, 1],
        "NRZ-L",
        show=False,
    )

    assert ax.get_title() == "NRZ-L Line-Coded Signal"
    assert len(ax.patches) == 1

    plt.close(fig)


def test_plot_manchester_with_two_levels_per_bit():
    fig, ax = plot_line_coded_signal(
        [0, 1],
        [1, -1, -1, 1],
        "Manchester",
        samples_per_bit=2,
        show=False,
    )

    assert ax.get_xlim() == (0.0, 2.0)
    assert len(ax.patches) == 1

    plt.close(fig)


def test_plot_pcm_reconstruction():
    fig, ax = plot_pcm_reconstruction(
        [0.0, 1.2, 2.0],
        [0.0, 1.0, 2.0],
        show=False,
    )

    assert ax.get_title() == "PCM Signal Reconstruction"
    assert len(ax.patches) == 1
    assert len(ax.lines) == 1

    plt.close(fig)


def test_digital_plot_rejects_empty_input():
    with pytest.raises(ValueError):
        plot_digital_stream([], show=False)


def test_digital_plot_rejects_invalid_bits():
    with pytest.raises(ValueError):
        plot_digital_stream([0, 2, 1], show=False)


def test_line_plot_rejects_mismatched_signal_length():
    with pytest.raises(ValueError):
        plot_line_coded_signal(
            [0, 1],
            [1, -1, 1],
            "Manchester",
            samples_per_bit=2,
            show=False,
        )


def test_line_plot_rejects_invalid_signal_level():
    with pytest.raises(ValueError):
        plot_line_coded_signal(
            [0, 1],
            [1, 2],
            "NRZ-L",
            show=False,
        )


def test_pcm_plot_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        plot_pcm_reconstruction([0.0, 1.0], [0.0], show=False)


def test_pcm_plot_rejects_non_finite_samples():
    with pytest.raises(ValueError):
        plot_pcm_reconstruction(
            [0.0, float("nan")],
            [0.0, 1.0],
            show=False,
        )
