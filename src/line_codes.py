
"""Line coding encoders for digital communication."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable


def _validate_bits(bits: Sequence[int]) -> None:
    """Ensure the input contains only binary values."""
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")


def encode_nrz_l(bits: Sequence[int]) -> list[int]:
    """Encode bits using NRZ-L: 0 -> +1, 1 -> -1."""
    _validate_bits(bits)
    return [1 if bit == 0 else -1 for bit in bits]


def encode_nrz_i(
    bits: Sequence[int], initial_level: int = -1
) -> list[int]:
    """Encode using NRZ-I: a 1 causes a transition at the bit start."""
    _validate_bits(bits)

    if initial_level not in (-1, 1):
        raise ValueError("initial_level must be -1 or +1")

    level = initial_level
    signal = []

    for bit in bits:
        if bit == 1:
            level = -level
        signal.append(level)

    return signal


def encode_manchester(bits: Sequence[int]) -> list[int]:
    """Encode bits using Manchester coding, with two levels per bit."""
    _validate_bits(bits)

    levels = {
        0: (1, -1),
        1: (-1, 1),
    }

    return [level for bit in bits for level in levels[bit]]


def encode_differential_manchester(
    bits: Sequence[int], initial_level: int = 1
) -> list[int]:
    """
    Encode using Differential Manchester.

    Every bit has a mid-bit transition.
    A 0 causes a transition at the beginning; a 1 does not.
    """
    _validate_bits(bits)

    if initial_level not in (-1, 1):
        raise ValueError("initial_level must be -1 or +1")

    level = initial_level
    signal = []

    for bit in bits:
        if bit == 0:
            level = -level

        signal.append(level)

        level = -level
        signal.append(level)

    return signal


def encode_ami(bits: Sequence[int]) -> list[int]:
    """Encode using AMI: zeros -> 0; ones alternate +1 and -1."""
    _validate_bits(bits)

    next_polarity = 1
    signal = []

    for bit in bits:
        if bit == 0:
            signal.append(0)
        else:
            signal.append(next_polarity)
            next_polarity = -next_polarity

    return signal


# A mapping keeps scheme selection modular rather than using a long if/else chain.
_ENCODERS: dict[str, Callable[[Sequence[int]], list[int]]] = {
    "NRZ-L": encode_nrz_l,
    "NRZ-I": encode_nrz_i,
    "MANCHESTER": encode_manchester,
    "DIFFERENTIAL MANCHESTER": encode_differential_manchester,
    "DIFFERENTIAL-MANCHESTER": encode_differential_manchester,
    "AMI": encode_ami,
}


def encode_line_code(bits: Sequence[int], scheme: str) -> list[int]:
    """Encode bits using the named line-coding scheme."""
    encoder = _ENCODERS.get(scheme.strip().upper())

    if encoder is None:
        raise ValueError(f"Unsupported line-coding scheme: {scheme}")

    return encoder(bits)
