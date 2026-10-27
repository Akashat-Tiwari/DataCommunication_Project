
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from src.line_codes import (
    encode_nrz_l,
    encode_nrz_i,
    encode_manchester,
    encode_differential_manchester,
    encode_ami,
)
from src.scrambling import scramble_b8zs, scramble_hdb3
from src.plotting import plot_line_coded_signal
from src.image_decoder import decode_signal_image


IMAGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "decoder_signal.png"
)


def check_image_round_trip(bits, encoder, scheme, samples_per_bit=1):
    signal_levels = encoder(bits)

    fig, _ = plot_line_coded_signal(
        bits,
        signal_levels,
        scheme,
        samples_per_bit=samples_per_bit,
        show=False,
    )
    plt.close(fig)

    # The decoder receives no original bits or encoded signal array.
    recovered = decode_signal_image(IMAGE_PATH, scheme)

    assert recovered == bits


def test_image_decoder_nrz_l():
    check_image_round_trip(
        [0, 1, 1, 0, 1],
        encode_nrz_l,
        "NRZ-L",
    )


def test_image_decoder_nrz_i():
    check_image_round_trip(
        [0, 1, 1, 0, 1],
        encode_nrz_i,
        "NRZ-I",
    )


def test_image_decoder_manchester():
    check_image_round_trip(
        [0, 1, 1, 0],
        encode_manchester,
        "Manchester",
        samples_per_bit=2,
    )


def test_image_decoder_differential_manchester():
    check_image_round_trip(
        [0, 1, 1, 0],
        encode_differential_manchester,
        "Differential Manchester",
        samples_per_bit=2,
    )


def test_image_decoder_ami():
    check_image_round_trip(
        [1, 0, 1, 1, 0, 1],
        encode_ami,
        "AMI",
    )


def test_image_decoder_b8zs():
    bits = [1] + [0] * 8 + [1]

    check_image_round_trip(
        bits,
        scramble_b8zs,
        "B8ZS",
    )


def test_image_decoder_hdb3():
    bits = [1, 1] + [0] * 4 + [1] + [0] * 4

    check_image_round_trip(
        bits,
        scramble_hdb3,
        "HDB3",
    )


def test_decoder_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        decode_signal_image(IMAGE_PATH, "UNKNOWN")


def test_decoder_rejects_missing_image():
    with pytest.raises(FileNotFoundError):
        decode_signal_image(
            IMAGE_PATH.parent / "missing_waveform.png",
            "NRZ-L",
        )
