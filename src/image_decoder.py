
"""Decode line-coded signals by analysing a saved waveform image."""

from pathlib import Path

import numpy as np
from matplotlib import image as mpimg


PIXELS_PER_LEVEL = 20
VALID_LEVELS = (-1, 0, 1)


def _extract_levels_from_image(image_path: str | Path) -> list[int]:
    """Recover signal levels from the red waveform in a PNG image."""
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Waveform image not found: {path}")

    image = np.asarray(mpimg.imread(path))

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected a colour waveform image")

    height, width = image.shape[:2]

    if width == 0 or height == 0 or width % PIXELS_PER_LEVEL != 0:
        raise ValueError("Image dimensions do not match the decoder waveform format")

    rgb = image[:, :, :3]

    # The plotting function draws the waveform in red on a white background.
    red_mask = (
        (rgb[:, :, 0] > 0.75)
        & (rgb[:, :, 1] < 0.40)
        & (rgb[:, :, 2] < 0.40)
    )

    level_count = width // PIXELS_PER_LEVEL

    # Expected pixel rows for +1, 0, and -1 with the plotting range [-1.5, 1.5].
    expected_rows = {
        level: (1.5 - level) * height / 3.0
        for level in VALID_LEVELS
    }

    recovered_levels = []

    for index in range(level_count):
        # Sample the middle of each signal interval, away from transitions.
        x = int((index + 0.5) * PIXELS_PER_LEVEL)
        red_rows = np.flatnonzero(red_mask[:, x])

        if red_rows.size == 0:
            raise ValueError(
                f"No red waveform detected in signal interval {index}"
            )

        observed_row = float(np.median(red_rows))

        level = min(
            VALID_LEVELS,
            key=lambda candidate: abs(
                observed_row - expected_rows[candidate]
            ),
        )

        # Reject pixels that are too far from any expected signal level.
        if abs(observed_row - expected_rows[level]) > max(8, height * 0.04):
            raise ValueError(
                f"Could not reliably identify signal level at interval {index}"
            )

        recovered_levels.append(level)

    return recovered_levels


def _decode_nrz_l(levels: list[int]) -> list[int]:
    """Decode NRZ-L using 0 -> +1 and 1 -> -1."""
    if any(level not in (-1, 1) for level in levels):
        raise ValueError("Invalid NRZ-L signal level")

    return [0 if level == 1 else 1 for level in levels]


def _decode_nrz_i(levels: list[int]) -> list[int]:
    """Decode NRZ-I; a transition represents bit 1."""
    previous_level = -1
    bits = []

    for level in levels:
        if level not in (-1, 1):
            raise ValueError("Invalid NRZ-I signal level")

        bits.append(int(level != previous_level))
        previous_level = level

    return bits


def _decode_manchester(levels: list[int]) -> list[int]:
    """Decode Manchester pairs: +1,-1 -> 0 and -1,+1 -> 1."""
    if len(levels) % 2:
        raise ValueError("Manchester signal must contain an even number of levels")

    pairs = {
        (1, -1): 0,
        (-1, 1): 1,
    }

    bits = []

    for index in range(0, len(levels), 2):
        pair = tuple(levels[index:index + 2])

        if pair not in pairs:
            raise ValueError(f"Invalid Manchester pair at index {index}")

        bits.append(pairs[pair])

    return bits


def _decode_differential_manchester(levels: list[int]) -> list[int]:
    """Decode Differential Manchester using transitions at bit boundaries."""
    if len(levels) % 2:
        raise ValueError(
            "Differential Manchester signal must contain an even number of levels"
        )

    previous_level = 1
    bits = []

    for index in range(0, len(levels), 2):
        first, second = levels[index:index + 2]

        if first != -second:
            raise ValueError(
                f"Missing mid-bit transition at signal interval {index}"
            )

        bits.append(int(first == previous_level))
        previous_level = second

    return bits


def _decode_ami(levels: list[int]) -> list[int]:
    """Decode ordinary AMI while checking alternating pulse polarity."""
    previous_pulse = -1
    bits = []

    for level in levels:
        if level == 0:
            bits.append(0)
            continue

        if level != -previous_pulse:
            raise ValueError("Invalid AMI pulse polarity")

        bits.append(1)
        previous_pulse = level

    return bits


def _decode_b8zs(levels: list[int]) -> list[int]:
    """Decode AMI with B8ZS substitutions."""
    bits = []
    previous_pulse = -1
    index = 0

    while index < len(levels):
        window = levels[index:index + 8]

        is_substitution = (
            len(window) == 8
            and window[:3] == [0, 0, 0]
            and window[3] in (-1, 1)
            and window[4] == -window[3]
            and window[5] == 0
            and window[6] == window[4]
            and window[7] == window[3]
        )

        if is_substitution:
            bits.extend([0] * 8)
            previous_pulse = window[7]
            index += 8
            continue

        level = levels[index]

        if level == 0:
            bits.append(0)
        else:
            if level != -previous_pulse:
                raise ValueError("Invalid AMI polarity outside B8ZS substitution")

            bits.append(1)
            previous_pulse = level

        index += 1

    return bits


def _decode_hdb3(levels: list[int]) -> list[int]:
    """Decode AMI with HDB3 substitutions."""
    bits = []
    previous_pulse = -1
    index = 0

    while index < len(levels):
        window = levels[index:index + 4]

        is_000v = (
            len(window) == 4
            and window[:3] == [0, 0, 0]
            and window[3] in (-1, 1)
            and window[3] == previous_pulse
        )

        is_b00v = (
            len(window) == 4
            and window[0] in (-1, 1)
            and window[1:3] == [0, 0]
            and window[3] == window[0]
            and window[0] == -previous_pulse
        )

        if is_000v or is_b00v:
            bits.extend([0] * 4)
            previous_pulse = window[3]
            index += 4
            continue

        level = levels[index]

        if level == 0:
            bits.append(0)
        else:
            if level != -previous_pulse:
                raise ValueError("Invalid AMI polarity outside HDB3 substitution")

            bits.append(1)
            previous_pulse = level

        index += 1

    return bits


_DECODERS = {
    "NRZL": _decode_nrz_l,
    "NRZI": _decode_nrz_i,
    "MANCHESTER": _decode_manchester,
    "DIFFERENTIALMANCHESTER": _decode_differential_manchester,
    "AMI": _decode_ami,
    "B8ZS": _decode_b8zs,
    "AMIB8ZS": _decode_b8zs,
    "HDB3": _decode_hdb3,
    "AMIHDB3": _decode_hdb3,
}


def decode_signal_image(
    image_path: str | Path,
    scheme: str,
) -> list[int]:
    """
    Recover the binary stream using only the saved waveform image and scheme.

    The image determines the signal levels. Original bits and encoder output
    arrays are not accepted by this function.
    """
    key = (
        scheme.strip()
        .upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("+", "")
    )

    decoder = _DECODERS.get(key)

    if decoder is None:
        raise ValueError(f"Unsupported decoding scheme: {scheme}")

    levels = _extract_levels_from_image(image_path)
    return decoder(levels)
