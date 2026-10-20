"""Random bitstream generation helpers."""

from __future__ import annotations

import random
from typing import List, Optional


def generate_random_bits(length: int, rng: Optional[random.Random] = None) -> List[int]:
    """Return exactly ``length`` randomly generated binary values."""
    if length < 0:
        raise ValueError("length must be non-negative")

    generator = rng if rng is not None else random
    return [generator.randint(0, 1) for _ in range(length)]


def inject_zero_run(
    bits: List[int], run_length: int, rng: Optional[random.Random] = None
) -> List[int]:
    """Insert a contiguous zero run and return a new bit list."""
    if run_length < 0:
        raise ValueError("run_length must be non-negative")
    if run_length == 0:
        return bits[:]

    generator = rng if rng is not None else random
    insertion_index = generator.randrange(len(bits) + 1)
    return (
        bits[:insertion_index]
        + [0] * run_length
        + bits[insertion_index:]
    )